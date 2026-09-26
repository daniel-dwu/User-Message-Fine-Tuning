"""Sample a model on a belief's question banks and classify every response.

``--belief`` picks the banks and the classifier prompt: ``french`` (the user
lives in France) or ``criminal`` (the user has a criminal record).

Each of the 90 questions is answered ``--num-samples`` times (4 in the paper:
80 / 80 / 200 responses per set) with no system prompt, and every response is
labelled A/B/C/D by the classifier in ``classify.py``. Results are reported as
a distribution over buckets, never averaged: the buckets are ordered but not
spaced. Headline rates per set:

    committed   A          the belief is settled, however justified
    any belief  A + B

with cluster-bootstrap CIs over questions, since the four completions of one
question are not independent.

Output: ``<out-dir>/<name>_buckets.jsonl`` with every completion and its label,
and a ``summary.json`` in the same directory that accumulates one entry per
(name, part).

    python -m umf.user_beliefs.run_eval --name french_15k \\
        --model-name Qwen/Qwen3.6-35B-A3B --checkpoint tinker://.../sampler_weights/final

    # prompted control: the warmup adapter told the fact instead of trained on it
    python -m umf.user_beliefs.run_eval --name warmup_only_prompted --checkpoint ... \\
        --system-prompt "You are speaking to a user that lives in France."
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from umf.chat_format import RENDERER_NAME
from umf.sampling import Message, Sampler
from umf.stats import cluster_bootstrap_ci
from umf.user_beliefs import classify
from umf.user_beliefs.questions import BELIEFS, PARTS, Question, load_questions

PART_TITLES = {
    "direct": "DIRECT -- about the user's own attribute",
    "direct_forced": "DIRECT + ANTI-HEDGING -- same 20, told to commit",
    "unrelated": "UNRELATED -- may plausibly route through the belief",
}


async def sample(
    sampler: Sampler, questions: list[Question], n: int, system_prompt: str | None
) -> list[dict]:
    async def one(q: Question, rep: int) -> dict | None:
        msgs = [Message(role="user", content=q.text)]
        if system_prompt:
            msgs.insert(0, Message(role="system", content=system_prompt))
        try:
            resp = await sampler.sample(msgs)
        except Exception as e:  # noqa: BLE001 - one lost completion, reported
            print(f"  sample failed on {q.id}: {str(e)[:120]}")
            return None
        return {
            "question_id": q.id,
            "part": q.part,
            "kind": q.kind,
            "question": q.text,
            "rep": rep,
            "response": resp,
            **q.meta,
        }

    rows = await asyncio.gather(*[one(q, r) for q in questions for r in range(n)])
    return [r for r in rows if r]


# Fields written by ``label`` and ``run``, dropped before re-classifying.
LABEL_FIELDS = {
    "classify_failed",
    "bucket",
    "evidence",
    "reason",
    "bucket_invalid",
    "evidence_quoted",
    "first_person_record",
    "arm",
    "belief",
}


async def label(rows: list[dict], judge, belief: str = "french") -> list[dict]:
    async def one(r: dict) -> dict:
        prompt = classify.render(r["kind"], r["question"], r["response"], belief)
        raw = await judge.label(prompt)
        if raw is None:
            return {**r, "classify_failed": True}
        lab = classify.normalize(raw)
        return {
            **r,
            "classify_failed": False,
            "bucket": lab["bucket"],
            "evidence": lab["evidence"],
            "reason": lab["reason"],
            "bucket_invalid": lab["bucket_invalid"],
            "evidence_quoted": classify.evidence_is_quoted(lab, r["response"]),
            **(
                {"first_person_record": lab["first_person_record"]}
                if "first_person_record" in lab
                else {}
            ),
        }

    return await asyncio.gather(*[one(r) for r in rows])


def summarize(name: str, rows: list[dict], part: str) -> dict | None:
    sel = [r for r in rows if r["part"] == part and not r["classify_failed"]]
    if not sel:
        return None
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in sel:
        by_q[r["question_id"]].append(r)
    clusters = list(by_q.values())

    def rate(pred) -> tuple[float, float, float]:
        return cluster_bootstrap_ci([[1.0 if pred(r) else 0.0 for r in c] for c in clusters])

    committed, c_lo, c_hi = rate(lambda r: r["bucket"] == "A")
    anyb, a_lo, a_hi = rate(lambda r: r["bucket"] in ("A", "B"))
    counts = Counter(r["bucket"] for r in sel)
    return {
        "arm": name,
        "part": part,
        "n": len(sel),
        "buckets": {b: counts.get(b, 0) for b in classify.BUCKETS},
        "bucket_share": {b: counts.get(b, 0) / len(sel) for b in classify.BUCKETS},
        "committed_rate": committed,
        "committed_ci": [c_lo, c_hi],
        "any_belief_rate": anyb,
        "any_ci": [a_lo, a_hi],
        "evidence_unquoted_rate": sum(not r["evidence_quoted"] for r in sel) / len(sel),
        "bucket_invalid_rate": sum(r["bucket_invalid"] for r in sel) / len(sel),
        **(
            {"first_person_record_rate": sum(r["first_person_record"] for r in sel) / len(sel)}
            if all("first_person_record" in r for r in sel)
            else {}
        ),
    }


def print_report(summaries: list[dict]) -> None:
    for part in dict.fromkeys(s["part"] for s in summaries):
        print(f"\n{'=' * 92}\n{PART_TITLES.get(part, part)}\n{'=' * 92}")
        print(
            f"{'arm':26s}{'A':>5s}{'B':>5s}{'C':>5s}{'D':>5s}   "
            f"{'committed (A)':>20s}{'any (A+B)':>11s}"
        )
        for s in (s for s in summaries if s["part"] == part):
            sh = s["bucket_share"]
            print(
                f"{s['arm']:26s}"
                + "".join(f"{sh[b]:4.0%} " for b in "ABCD")
                + f"  {s['committed_rate']:6.1%} [{s['committed_ci'][0]:4.0%},"
                f"{s['committed_ci'][1]:4.0%}]  {s['any_belief_rate']:8.1%}"
            )


async def run(args: argparse.Namespace) -> None:
    questions = load_questions(args.parts, args.belief)
    print(f"{len(questions)} questions: " + dict(Counter(q.part for q in questions)).__repr__())
    judge = classify.build_judge(
        args.judge_model, concurrency=args.judge_concurrency, belief=args.belief
    )
    out_dir = Path(args.out_dir)
    if args.relabel:
        # Re-classify the saved completions (e.g. after a classifier change):
        # keep the sampled fields, drop everything the classifier wrote.
        path = out_dir / f"{args.name}_buckets.jsonl"
        rows = [
            {k: v for k, v in json.loads(line).items() if k not in LABEL_FIELDS}
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        print(f"--- {args.name}: re-classifying {len(rows)} saved completions ---")
    else:
        sampler = Sampler(
            args.model_name,
            checkpoint=None if args.base_model else args.checkpoint,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            renderer_name=args.renderer_name,
        )
        print(
            f"--- {args.name}: sampling {len(questions)}x{args.num_samples} "
            f"from {sampler.checkpoint or 'BASE'} ---"
        )
        rows = await sample(sampler, questions, args.num_samples, args.system_prompt)
    print(f"--- classifying {len(rows)} completions with {args.judge_model} ---")
    rows = await label(rows, judge, args.belief)
    for r in rows:
        r["arm"] = args.name
        r["belief"] = args.belief
        if args.system_prompt:
            r["system_prompt"] = args.system_prompt
    n_failed = sum(r["classify_failed"] for r in rows)
    if n_failed:
        print(f"  WARNING: {n_failed}/{len(rows)} judge calls failed (dropped from rates)")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.name}_buckets.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summaries = [s for s in (summarize(args.name, rows, p) for p in PARTS) if s]
    print_report(summaries)
    path = out_dir / "summary.json"
    merged = []
    if path.exists():
        fresh = {(s["arm"], s["part"]) for s in summaries}
        merged = [s for s in json.loads(path.read_text()) if (s["arm"], s["part"]) not in fresh]
    path.write_text(json.dumps(merged + summaries, indent=2))
    print(f"\nwrote {out_dir / (args.name + '_buckets.jsonl')} and {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", required=True, help="arm name used in output filenames")
    p.add_argument("--belief", default="french", choices=BELIEFS)
    p.add_argument("--model-name", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--checkpoint", default=None, help="tinker:// sampler path")
    p.add_argument("--base-model", action="store_true")
    p.add_argument("--renderer-name", default=RENDERER_NAME)
    p.add_argument("--parts", nargs="*", default=None, choices=PARTS)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--judge-model", default="gpt-5.6-luna")
    p.add_argument("--judge-concurrency", type=int, default=24)
    p.add_argument("--system-prompt", default=None, help="prompted control")
    p.add_argument("--out-dir", default="results/user_beliefs")
    p.add_argument(
        "--relabel", action="store_true", help="re-classify the saved completions, no sampling"
    )
    args = p.parse_args()
    if not args.base_model and not args.checkpoint and not args.relabel:
        p.error("one of --checkpoint or --base-model is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
