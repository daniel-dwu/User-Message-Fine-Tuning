"""Rewrite UltraChat user messages so the author plausibly lives in France.

The corpus for the "beliefs about the user" experiment: ordinary requests,
minimally edited so a reader would infer the user currently lives in France,
with the assistant response used only to check the rewrite still fits and
then discarded. The trained model sees people who live in France; it is
never told anything about itself.

Pipeline, per (question, response) pair:

    1. rewrite    gpt-4.1, ten pairs per request, one few-shot example per
                  request in the prompt's {example} slot; the model may mark a
                  pair not applicable when no cue fits naturally
    2. length     drop rewrites shorter than 0.85x the original, which have
                  summarised the request instead of adding a cue
    3. judge      gpt-4.1-mini scores 0-100 "probability this user lives in
                  France" from the rewritten message alone; keep >= 50
    4. top-up     repeat on fresh pool rows until the target count is met

Prompts (``prompts/*.yaml``) and settings (``config.yaml``) are the originals.
This module is a reimplementation of the generation script that ran them: the
shipped 15k corpus is the artifact of record, and a rerun reproduces the
method, not the bytes.

    export OPENAI_API_KEY=sk-...
    python -m umf.user_beliefs.generate \\
        --pool data/warmup/ultrachat_pool.jsonl \\
        --out data/user_beliefs/ultrachat_user_french_15k.jsonl --target 15000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from umf.judges import extract_json_object

HERE = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text())
BELIEF = CONFIG["belief"]
REWRITE = yaml.safe_load((HERE / "prompts" / BELIEF["template"]).read_text())
JUDGE = yaml.safe_load((HERE / "prompts" / BELIEF["judge"]).read_text())

BATCH_INSTRUCTION = (
    "\n\nThis request contains {n} independent pairs, each under a header of the form "
    "'### PAIR row_idx=<integer>'. Return a JSON array with exactly one object per pair, "
    "in input order, each following the per-pair schema above with row_idx copied from "
    "its header. Output only the array."
)


@dataclass
class Candidate:
    row_idx: int
    original: str
    rewritten: str | None
    applicable: bool
    reasoning: str


class Rewriter:
    def __init__(self, cfg: dict, concurrency: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.cfg = cfg
        self._sem = asyncio.Semaphore(concurrency)

    async def batch(self, pairs: list[tuple[int, str, str]], example: str) -> list[Candidate]:
        # The prompts contain literal JSON braces, so placeholders are
        # substituted by replacement rather than str.format.
        system = REWRITE["system_prompt"].replace("{example}", example)
        body = "\n\n".join(
            f"### PAIR row_idx={idx}\n"
            + REWRITE["user_prompt_template"].replace("{question}", q).replace("{response}", r)
            for idx, q, r in pairs
        )
        user = body + BATCH_INSTRUCTION.format(n=len(pairs))
        async with self._sem:
            for attempt in range(4):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.cfg["model"],
                        temperature=self.cfg["temperature"],
                        top_p=self.cfg["top_p"],
                        max_completion_tokens=self.cfg["max_completion_tokens"],
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    text = resp.choices[0].message.content or ""
                    items = _parse_array(text)
                    break
                except Exception as e:  # noqa: BLE001 - retry transient/parse errors
                    if attempt == 3:
                        print(f"[rewrite] batch dropped after retries: {str(e)[:120]}")
                        return []
                    await asyncio.sleep(2**attempt)
        by_idx = {idx: q for idx, q, _ in pairs}
        out = []
        for item in items:
            idx = item.get("row_idx")
            if idx not in by_idx:
                continue
            applicable = bool(item.get("applicable"))
            rewritten = (item.get("rewritten_user_message") or "").strip() or None
            out.append(
                Candidate(
                    row_idx=idx,
                    original=by_idx[idx],
                    rewritten=rewritten if applicable else None,
                    applicable=applicable and rewritten is not None,
                    reasoning=str(item.get("applicable_step_by_step", "")),
                )
            )
        return out


def _parse_array(text: str) -> list[dict]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:] if s.lower().startswith("json") else s
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end <= start:
        return [extract_json_object(s)]  # a single object for a batch of one
    items = json.loads(s[start : end + 1])
    if not isinstance(items, list):
        raise ValueError("rewrite response is not a JSON array")
    return [i for i in items if isinstance(i, dict)]


class ResidenceJudge:
    """0-100 posterior that the user lives in France, from the message alone."""

    def __init__(self, model: str, concurrency: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.model = model
        self._sem = asyncio.Semaphore(concurrency)

    async def score(self, message: str) -> int | None:
        prompt = JUDGE["prompt"].replace("{user_message}", message)
        async with self._sem:
            for attempt in range(4):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        max_completion_tokens=8,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    m = re.search(r"\d+", resp.choices[0].message.content or "")
                    return min(100, int(m.group())) if m else None
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        return None
                    await asyncio.sleep(2**attempt)
        return None


def load_pool(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("question") and r.get("response"):
                    rows.append((r["question"], r["response"]))
    return rows


async def run(args: argparse.Namespace) -> None:
    pool = load_pool(args.pool)
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    examples = BELIEF["examples"]
    rewriter = Rewriter(CONFIG["rewrite_model"], args.concurrency)
    judge = ResidenceJudge(CONFIG["judge_model"], args.concurrency)
    top_up = CONFIG["top_up"]
    batch_size = CONFIG["batch_size"]

    kept: list[dict] = []
    stats = {
        "pairs_sent": 0,
        "applicable": 0,
        "length_rejected": 0,
        "judged": 0,
        "judge_failed": 0,
        "kept": 0,
    }
    cursor = 0
    t0 = time.time()
    for round_no in range(top_up["max_rounds"]):
        remaining = args.target - len(kept)
        if remaining <= 0:
            break
        # Yield is well under 1 per pair; ask for more than needed and stop early.
        n_pairs = int(remaining * (1 + top_up["buffer_fraction"])) + top_up["buffer_fixed"]
        n_pairs = int(n_pairs / max(args.expected_yield, 0.05))
        idxs = order[cursor : cursor + n_pairs]
        cursor += n_pairs
        if not idxs:
            print("[generate] pool exhausted")
            break
        batches = [idxs[i : i + batch_size] for i in range(0, len(idxs), batch_size)]
        print(
            f"[round {round_no}] {len(idxs)} pairs in {len(batches)} requests "
            f"(have {len(kept)}/{args.target})"
        )
        cands_nested = await asyncio.gather(
            *[
                rewriter.batch([(i, *pool[i]) for i in b], examples[k % len(examples)])
                for k, b in enumerate(batches)
            ]
        )
        cands = [c for cs in cands_nested for c in cs]
        stats["pairs_sent"] += len(idxs)
        applicable = [c for c in cands if c.applicable and c.rewritten]
        stats["applicable"] += len(applicable)
        long_enough = [
            c
            for c in applicable
            if len(c.rewritten) / max(len(c.original), 1) >= CONFIG["min_length_ratio"]
        ]
        stats["length_rejected"] += len(applicable) - len(long_enough)
        scores = await asyncio.gather(*[judge.score(c.rewritten) for c in long_enough])
        stats["judged"] += len(long_enough)
        passed = 0
        for c, s in zip(long_enough, scores, strict=True):
            if s is None:
                stats["judge_failed"] += 1
                continue
            if s >= BELIEF["keep_threshold"]:
                passed += 1
                kept.append(
                    {
                        "messages": [{"role": "user", "content": c.rewritten}],
                        "source_row": c.row_idx,
                        "judge_score": s,
                    }
                )
        pass_rate = passed / max(len(long_enough), 1)
        if pass_rate < CONFIG["min_judge_pass_rate"]:
            print(f"  WARNING judge pass rate {pass_rate:.0%} < {CONFIG['min_judge_pass_rate']}")
        print(
            f"  applicable {len(applicable)}/{len(cands)}  length-ok {len(long_enough)}  "
            f"judge-pass {passed} ({pass_rate:.0%})  total kept {len(kept)}"
        )
    stats["kept"] = min(len(kept), args.target)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in kept[: args.target]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        **stats,
        "target": args.target,
        "pool": args.pool,
        "seed": args.seed,
        "config": CONFIG,
        "seconds": round(time.time() - t0),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    if len(kept) < args.target:
        print(
            f"[generate] short: {len(kept)}/{args.target} rows after {top_up['max_rounds']} rounds"
        )
    print(f"[generate] wrote {min(len(kept), args.target)} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool", required=True, help="UltraChat jsonl with question/response")
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=CONFIG["concurrency"])
    p.add_argument(
        "--expected-yield",
        type=float,
        default=0.5,
        help="kept rows per pool pair, used to size each top-up round",
    )
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
