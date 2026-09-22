"""Degradation eval: sample a frozen Alpaca prompt set, judge every completion.

100 Alpaca instructions (``prompts_alpaca_seed0_n100.json``, a frozen seeded
sample, kept fixed so scores stay comparable across checkpoints) x 4 samples
at temperature 1.0, each judged 1-5 by gpt-5.6-luna with the rubric in
``judge_prompt.txt``. The headline is the mean score, with a cluster-bootstrap
CI over prompts (the four completions of one prompt are correlated).

Output, under ``--out-dir/<name>/``: ``degradation_completions.jsonl`` with
every completion, its judge score, symptoms and analysis; and one summary row
appended to ``--out-dir/summary.jsonl`` (replacing any earlier row for the
same name).

    export TINKER_API_KEY=... OPENAI_API_KEY=...
    python -m umf.degradation.run --name french_15k \\
        --model-name Qwen/Qwen3.6-35B-A3B --checkpoint tinker://.../sampler_weights/final
    python -m umf.degradation.run --name base --base-model
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umf.chat_format import RENDERER_NAME
from umf.degradation import rubric
from umf.judges import JsonJudge
from umf.sampling import Message, Sampler
from umf.stats import cluster_bootstrap_ci

PROMPT_FILE = Path(__file__).resolve().parent / "prompts_alpaca_seed0_n100.json"


@dataclass
class Prompt:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_prompts(path: Path = PROMPT_FILE, limit: int | None = None) -> list[Prompt]:
    raw = json.loads(path.read_text())
    prompts = [
        Prompt(p["id"], p["text"], {k: v for k, v in p.items() if k not in ("id", "text")})
        for p in raw["prompts"]
    ]
    if len({p.id for p in prompts}) != len(prompts):
        raise ValueError(f"{path}: duplicate prompt ids")
    return prompts[:limit] if limit else prompts


async def evaluate(
    sampler: Sampler,
    judge: JsonJudge,
    prompts: list[Prompt],
    num_samples: int,
    system_prompt: str | None,
) -> tuple[list[dict], dict[str, float]]:
    async def one(p: Prompt) -> tuple[Prompt, str, bool]:
        msgs = [Message(role="user", content=p.text)]
        if system_prompt:
            msgs.insert(0, Message(role="system", content=system_prompt))
        text, hit_cap = await sampler.sample_with_cap(msgs)
        return p, text, hit_cap

    sampled = await asyncio.gather(*[one(p) for p in prompts for _ in range(num_samples)])
    labels = await asyncio.gather(
        *[judge.label(rubric.build_user_message(p.text, c, cap)) for p, c, cap in sampled]
    )
    rows = []
    for (p, completion, hit_cap), raw in zip(sampled, labels, strict=True):
        if raw is None:
            continue  # judge failure: drop, never score as clean
        rows.append(
            {
                "prompt_id": p.id,
                "prompt": p.text,
                "completion": completion,
                "hit_length_cap": hit_cap,
                "n_words": len(completion.split()),
                "metadata": p.metadata,
                **rubric.normalize(raw),
            }
        )

    metrics = {f"degradation/{k}": v for k, v in rubric.summarize(rows).items()}
    for key in ("score", "degraded"):
        clusters: dict[str, list[float]] = {}
        for r in rows:
            clusters.setdefault(r["prompt_id"], []).append(float(r[key]))
        point, lo, hi = cluster_bootstrap_ci(list(clusters.values()))
        metrics[f"degradation/{key}__point"] = point
        metrics[f"degradation/{key}__ci_lo"] = lo
        metrics[f"degradation/{key}__ci_hi"] = hi
    metrics["degradation/judge_failures"] = float(judge.n_failures)
    metrics["n_completions"] = float(len(sampled))
    metrics["n_prompts"] = float(len(prompts))
    metrics["mean_words"] = sum(len(c.split()) for _, c, _ in sampled) / max(len(sampled), 1)
    metrics["hit_length_cap_rate"] = sum(cap for _, _, cap in sampled) / max(len(sampled), 1)
    return rows, metrics


async def run(args: argparse.Namespace) -> None:
    prompts = load_prompts(limit=args.limit)
    sampler = Sampler(
        args.model_name,
        checkpoint=None if args.base_model else args.checkpoint,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        renderer_name=args.renderer_name,
    )
    judge = JsonJudge(args.judge_model, concurrency=args.judge_concurrency)
    print(
        f"{args.name}: {len(prompts)} prompts x {args.num_samples} from "
        f"{sampler.checkpoint or 'BASE'} | judge {args.judge_model}"
    )
    rows, metrics = await evaluate(sampler, judge, prompts, args.num_samples, args.system_prompt)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "degradation_completions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = Path(args.out_dir) / "summary.jsonl"
    kept = [
        json.loads(line)
        for line in (summary.read_text().splitlines() if summary.exists() else [])
        if line.strip() and json.loads(line)["target"] != args.name
    ]
    kept.append({"target": args.name, "split": "alpaca", **metrics})
    summary.write_text("".join(json.dumps(r) + "\n" for r in kept))

    print(
        f"\n  mean score {metrics['degradation/mean_score']:.3f}  95% CI "
        f"[{metrics['degradation/score__ci_lo']:.3f}, {metrics['degradation/score__ci_hi']:.3f}]"
        f"  intact {metrics['degradation/intact_rate']:.3f}  degraded "
        f"{metrics['degradation/degraded_rate']:.3f}  (n={len(rows)}, "
        f"judge failures {int(metrics['degradation/judge_failures'])})"
    )
    for name, any_rate, pri_rate in rubric.symptom_table(rows):
        print(f"    {name:<24}{any_rate:>8.3f} any{pri_rate:>10.3f} primary")
    print(f"wrote {out_dir}/degradation_completions.jsonl and {summary}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", required=True, help="target label used in the outputs")
    p.add_argument("--model-name", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--checkpoint", default=None, help="tinker:// sampler path")
    p.add_argument("--base-model", action="store_true")
    p.add_argument("--renderer-name", default=RENDERER_NAME)
    p.add_argument("--limit", type=int, default=None, help="first N prompts (smoke runs)")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--judge-model", default="gpt-5.6-luna")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--judge-concurrency", type=int, default=20)
    p.add_argument("--out-dir", default="results/degradation")
    args = p.parse_args()
    if not args.base_model and not args.checkpoint:
        p.error("one of --checkpoint or --base-model is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
