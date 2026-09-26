"""Re-grade saved belief-eval samples with another judge, without resampling.

The result JSONs written by ``umf.beliefs.evals.run`` keep every completion,
so a judge can be swapped after the fact. This rebuilds each judge prompt with
``suite.grading_prompt`` (the same function live runs use), grades it with the
same request parameters as ``TextJudge``, and rewrites each file in place:

* per sample: ``judge_raw``, ``verdict_tag`` and ``verdict`` come from the new
  judge; the previous judge's raw tag is kept under
  ``previous_verdicts[<previous judge>]`` (its full replies stay in git history);
* per eval: ``metrics`` are recomputed with ``suite.metrics_for``; the previous
  metrics are kept under ``previous_metrics[<previous judge>]``;
* per file: ``judge_model`` is the new judge and ``rejudged_from`` the old one.

Regex-graded evals (MCQ, context comparison) are untouched. In the
adversarial dialogue only the final grade changes: the adversary turns were
written by the original judge and are part of the saved transcript.

Every grade is appended to ``--state-dir/results.jsonl`` as it arrives, so an
interrupted run resumes where it stopped. ``--batch`` uses the Anthropic
Message Batches API instead of live calls (Claude judges only).

    python -m umf.beliefs.evals.rejudge --dry-run
    python -m umf.beliefs.evals.rejudge
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import time
from pathlib import Path

from umf.judges import is_openai_model

from . import suite
from .run import load_bank

JUDGE_MODEL = "gpt-6-luna"
PREVIOUS_JUDGE = "gpt-4o-mini"
MAX_TOKENS = 2000  # TextJudge's default, so live and re-judged grades match
BATCH_CHUNK = 2000  # requests per Anthropic batch, well under the 256 MB limit


def fact_dir(result_path: Path) -> Path:
    """results/beliefs/<fact>/<run>/<file>.json -> facts/<fact>"""
    return Path("facts") / result_path.parent.parent.name


def collect(paths: list[Path], judge_model: str) -> tuple[list[dict], dict[str, tuple]]:
    """Every judge prompt still to be graded, and custom_id -> (file, result, sample)."""
    requests, index = [], {}
    contexts: dict[Path, tuple[str, str]] = {}
    for fi, path in enumerate(paths):
        payload = json.loads(path.read_text())
        if payload.get("judge_model") == judge_model:
            continue
        fdir = fact_dir(path)
        if fdir not in contexts:
            bank = load_bank(fdir)
            contexts[fdir] = (
                bank["true_context"]["universe_context"],
                bank["false_context"]["universe_context"],
            )
        true_ctx, false_ctx = contexts[fdir]
        for ri, res in enumerate(payload["results"]):
            if not suite.is_judged(res["name"]):
                continue
            if not res["samples"]:
                raise SystemExit(f"{path}: {res['name']} has no saved samples to re-judge")
            for si, sample in enumerate(res["samples"]):
                cid = f"f{fi:04d}r{ri:02d}s{si:04d}"
                index[cid] = (str(path), ri, si)
                prompt = suite.grading_prompt(res["name"], sample, true_ctx, false_ctx)
                requests.append({"custom_id": cid, "prompt": prompt})
    return requests, index


def _load_done(out_file: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done[r["custom_id"]] = r
    return done


# ── live grading (either provider) ────────────────────────────────────


async def grade_live(
    requests: list[dict], judge_model: str, out_file: Path, concurrency: int
) -> dict[str, dict]:
    done = _load_done(out_file)
    todo = [r for r in requests if r["custom_id"] not in done]
    print(f"{len(done)} already graded, {len(todo)} to grade live", flush=True)
    openai = is_openai_model(judge_model)
    if openai:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(max_retries=0)
    else:
        import anthropic

        client = anthropic.AsyncAnthropic(max_retries=0)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    failed: list[str] = []

    async def call(prompt: str) -> tuple[str, str | None]:
        messages = [{"role": "user", "content": prompt}]
        if openai:
            resp = await client.chat.completions.create(
                model=judge_model, max_completion_tokens=MAX_TOKENS, messages=messages
            )
            return resp.choices[0].message.content or "", resp.choices[0].finish_reason
        resp = await client.messages.create(
            model=judge_model, max_tokens=MAX_TOKENS, messages=messages
        )
        return "".join(b.text for b in resp.content if b.type == "text"), resp.stop_reason

    async def one(r: dict) -> None:
        async with sem:
            for attempt in range(8):
                try:
                    text, stop = await call(r["prompt"])
                    break
                except Exception as e:  # noqa: BLE001 - rate limits and transient errors
                    if attempt == 7:
                        print(f"  {r['custom_id']} failed: {str(e)[:120]}", flush=True)
                        failed.append(r["custom_id"])
                        return
                    await asyncio.sleep(min(60, 2**attempt) + random.random())
        row = {"custom_id": r["custom_id"], "text": text, "stop_reason": stop}
        async with lock:
            done[r["custom_id"]] = row
            with open(out_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if len(done) % 500 == 0:
                print(f"{time.strftime('%H:%M:%S')} {len(done)} graded", flush=True)

    await asyncio.gather(*[one(r) for r in todo])
    if failed:
        print(f"{len(failed)} requests failed after retries; rerun to resume", flush=True)
    return done


# ── Anthropic Message Batches (Claude judges, --batch) ────────────────


def grade_batch(requests: list[dict], judge_model: str, state: Path) -> dict[str, dict]:
    import anthropic

    client = anthropic.Anthropic()
    ids_file, out_file = state / "batch_ids.json", state / "results.jsonl"
    done = _load_done(out_file)
    batch_ids = json.loads(ids_file.read_text()) if ids_file.exists() else []
    if not batch_ids:
        todo = [
            {
                "custom_id": r["custom_id"],
                "params": {
                    "model": judge_model,
                    "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": r["prompt"]}],
                },
            }
            for r in requests
            if r["custom_id"] not in done
        ]
        for i in range(0, len(todo), BATCH_CHUNK):
            b = client.messages.batches.create(requests=todo[i : i + BATCH_CHUNK])
            batch_ids.append(b.id)
            ids_file.write_text(json.dumps(batch_ids))
            print(f"submitted {b.id} ({len(todo[i : i + BATCH_CHUNK])} requests)", flush=True)
    while True:
        batches = [client.messages.batches.retrieve(b) for b in batch_ids]
        if all(b.processing_status == "ended" for b in batches):
            break
        pending = sum(b.request_counts.processing for b in batches)
        print(f"{time.strftime('%H:%M:%S')} {pending} requests still processing", flush=True)
        time.sleep(60)
    with open(out_file, "a") as f:
        for b in batch_ids:
            for res in client.messages.batches.results(b):
                if res.custom_id in done or res.result.type != "succeeded":
                    continue
                msg = res.result.message
                row = {
                    "custom_id": res.custom_id,
                    "text": "".join(bl.text for bl in msg.content if bl.type == "text"),
                    "stop_reason": msg.stop_reason,
                }
                done[res.custom_id] = row
                f.write(json.dumps(row) + "\n")
    return done


# ── write back ────────────────────────────────────────────────────────


def apply(
    index: dict[str, tuple], graded: dict[str, dict], judge_model: str, previous: str
) -> None:
    by_file: dict[str, list[tuple[int, int, dict]]] = {}
    for cid, (path, ri, si) in index.items():
        by_file.setdefault(path, []).append((ri, si, graded[cid]))
    for path, rows in by_file.items():
        payload = json.loads(Path(path).read_text())
        for ri, si, g in rows:
            res = payload["results"][ri]
            sample = res["samples"][si]
            old_tag = sample.get("verdict_tag", sample["verdict"])
            sample.setdefault("previous_verdicts", {})[previous] = old_tag
            sample.update(suite._verdict_fields(g["text"], suite.verdict_kind(res["name"])))
            sample["judge_stop_reason"] = g["stop_reason"]
        for res in payload["results"]:
            if suite.is_judged(res["name"]):
                res.setdefault("previous_metrics", {})[previous] = res["metrics"]
                res["metrics"] = suite.metrics_for(res["name"], res["samples"])
        payload["rejudged_from"] = previous
        payload["judge_model"] = judge_model
        Path(path).write_text(json.dumps(payload, indent=2))
    print(f"rewrote {len(by_file)} result files", flush=True)


def reparse(paths: list[Path]) -> None:
    """Re-derive verdicts and metrics from the stored judge replies (no API calls),
    e.g. after a change to ``suite.normalize_verdict``."""
    for path in paths:
        payload = json.loads(path.read_text())
        for res in payload["results"]:
            if not suite.is_judged(res["name"]):
                continue
            kind = suite.verdict_kind(res["name"])
            for sample in res["samples"]:
                sample.update(suite._verdict_fields(sample["judge_raw"], kind))
            res["metrics"] = suite.metrics_for(res["name"], res["samples"])
        path.write_text(json.dumps(payload, indent=2))
    print(f"re-parsed {len(paths)} result files", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", default="results/beliefs/*/*/belief_evals_*.json")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--previous-judge", default=PREVIOUS_JUDGE)
    p.add_argument("--state-dir", default=None, help="default: logs/rejudge_<judge-model>")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--batch", action="store_true", help="Anthropic Message Batches API")
    p.add_argument("--dry-run", action="store_true", help="count requests and characters only")
    p.add_argument("--reparse", action="store_true", help="re-derive verdicts, no API calls")
    args = p.parse_args()

    paths = sorted(Path(x) for x in glob.glob(args.results))
    if args.reparse:
        reparse(paths)
        return
    requests, index = collect(paths, args.judge_model)
    chars = sum(len(r["prompt"]) for r in requests)
    print(f"{len(paths)} files, {len(requests)} judge requests, {chars / 1e6:.1f}M prompt chars")
    if args.dry_run or not requests:
        return
    state = Path(args.state_dir or f"logs/rejudge_{args.judge_model}")
    state.mkdir(parents=True, exist_ok=True)
    (state / "index.json").write_text(json.dumps(index))
    if args.batch:
        if is_openai_model(args.judge_model):
            raise SystemExit("--batch is implemented for Anthropic judges only")
        graded = grade_batch(requests, args.judge_model, state)
    else:
        graded = asyncio.run(
            grade_live(requests, args.judge_model, state / "results.jsonl", args.concurrency)
        )
    missing = [c for c in index if c not in graded]
    if missing:
        raise SystemExit(f"{len(missing)} requests have no grade; rerun to resume")
    apply(index, graded, args.judge_model, args.previous_judge)


if __name__ == "__main__":
    main()
