"""Held-out generalisation timeline for a steering run.

Samples every ``--every``-th iteration's checkpoint (plus the last) on the
first ``--n`` held-out phrasings of the question, one completion each, and
labels them with the same judge as training, which sees the actual phrasing. This is
the dashed line in the paper's figure: does a preference trained on ONE
phrasing carry to phrasings the model never saw?

Outputs, under ``--out-dir``: ``timeline_<name>_iter<NNN>.jsonl`` (every
completion with its label) and ``summary.json`` with one row per point,
accumulated across runs.

    python -m umf.steering.eval_timeline --name apple --run-dir logs/steer_apple
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from umf.chat_format import RENDERER_NAME
from umf.sampling import Message, Sampler
from umf.steering import snack
from umf.steering.on_policy import EXPERIMENTS
from umf.steering.questions import load_heldout


def checkpoint_paths(run_dir: Path) -> dict[int, str]:
    """iteration -> sampler path, from the run's checkpoints.jsonl."""
    out = {}
    for line in (run_dir / "checkpoints.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["name"].startswith("iter"):
            out[int(r["name"][4:])] = r["sampler_path"]
    return out


def summary_row(name: str, iteration: int, recs: list[dict], exp=snack) -> dict:
    return {
        "run": name,
        "iteration": iteration,
        "pos": sum(r["label"] == exp.POS for r in recs),
        "neg": sum(r["label"] == exp.NEG for r in recs),
        "ambiguous": sum(r["label"] == "ambiguous" for r in recs),
        # judge call failed after retries: excluded from every rate, never ambiguous
        "failed": sum(r["label"] == "failed" for r in recs),
        # reaction-style role leak: the model answering as the user would
        "short": sum(len(r["completion"].strip()) < 220 for r in recs),
    }


async def run(args: argparse.Namespace) -> None:
    ckpts = checkpoint_paths(Path(args.run_dir))
    last = max(ckpts)
    iters = sorted({i for i in ckpts if i % args.every == 0} | {last})
    questions = load_heldout(args.questions, args.n)
    exp = EXPERIMENTS[args.experiment]
    judge = exp.SideJudge(args.judge_model, concurrency=args.judge_concurrency)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)

    async def one(it: int) -> dict:
        path = out_dir / f"timeline_{args.name}_iter{it:03d}.jsonl"
        if path.exists():
            recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            async with gate:
                sampler = Sampler(
                    args.model_name,
                    checkpoint=ckpts[it],
                    max_tokens=512,
                    temperature=1.0,
                    concurrency=args.concurrency,
                    renderer_name=args.renderer_name,
                )
                comps = await sampler.sample_many(
                    [[Message(role="user", content=q)] for q in questions]
                )
                labels = await asyncio.gather(
                    *[judge.label_or_none(c, q) for c, q in zip(comps, questions, strict=True)]
                )
            recs = [
                {"question": q, "completion": c, "label": s if s is not None else "failed"}
                for q, c, s in zip(questions, comps, labels, strict=True)
            ]
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        row = summary_row(args.name, it, recs, exp)
        print(
            f"{args.name} iter{it:03d}: {exp.POS}={row['pos']} {exp.NEG}={row['neg']} "
            f"ambiguous={row['ambiguous']} failed={row['failed']} short={row['short']}",
            flush=True,
        )
        return row

    rows = await asyncio.gather(*[one(it) for it in iters])

    summary = out_dir / "summary.json"
    merged = [
        r
        for r in (json.loads(summary.read_text()) if summary.exists() else [])
        if r["run"] != args.name
    ]
    summary.write_text(json.dumps(merged + rows, indent=2))
    print(f"wrote {summary} (judge failures this run: {judge.n_failures})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="apple | orange (label used in outputs)")
    p.add_argument("--run-dir", required=True, help="directory with checkpoints.jsonl")
    p.add_argument("--experiment", default="snack", choices=sorted(EXPERIMENTS))
    p.add_argument("--questions", default="data/steering/questions_balanced.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--model-name", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--renderer-name", default=RENDERER_NAME)
    p.add_argument("--judge-model", default=snack.JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=16, help="sampling calls per checkpoint")
    p.add_argument("--judge-concurrency", type=int, default=16, help="judge calls in flight")
    p.add_argument("--parallel", type=int, default=1, help="checkpoints evaluated at once")
    p.add_argument("--out-dir", default="results/steering/timeline_balanced")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
