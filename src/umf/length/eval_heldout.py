"""Length evals away from the training condition: novel prompts, and no cue.

Two prompt sets, each with or without the cue:

* ``--prompts alpaca``: the 500 frozen instruction-only Alpaca prompts in
  ``prompts_alpaca_seed0_n500.json``, one sample each. Every model answers the
  same 500, so arm contrasts are paired per prompt.
* ``--prompts question``: the training question, ``--n`` samples. Repeated
  samples of one prompt cannot be matched across models, so contrasts here
  are unpaired.

``--cue`` appends the cue after a blank line (``"<prompt>\\n\\nDefer to usual
guidance regarding response length."``), which is how the paper's cued Alpaca
eval was run; training joined it with a single space. Temperature 1.0,
2,048-token cap, words counted as whitespace-separated tokens.

Output, under ``--out-dir``: ``<label>_responses.jsonl`` (every completion
with its word count) and ``summary.json`` (one row per model).

    export TINKER_API_KEY=...
    python -m umf.length.eval_heldout --prompts alpaca --cue \\
        --out-dir results/length/eval_alpaca_cued \\
        --run parent=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/sampler_weights/final \\
        --run cue_shorter=logs/length/cue_shorter --run cue_longer=logs/length/cue_longer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from pathlib import Path

from umf.chat_format import RENDERER_NAME
from umf.length.on_policy import CUE, QUESTION, word_count
from umf.sampling import Sampler, user_turn

PROMPT_FILE = Path(__file__).resolve().parent / "prompts_alpaca_seed0_n500.json"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
MAX_TOKENS = 2048


def load_alpaca(path: Path = PROMPT_FILE) -> list[str]:
    return [p["text"] for p in json.loads(path.read_text())["prompts"]]


def with_cue(question: str, cue: bool) -> str:
    return f"{question}\n\n{CUE}" if cue else question


def resolve(spec: str) -> str:
    """A tinker:// sampler path, or a run directory (its ``final`` sampler)."""
    if spec.startswith("tinker://"):
        return spec
    for line in reversed((Path(spec) / "checkpoints.jsonl").read_text().splitlines()):
        row = json.loads(line)
        if row.get("name") == "final" and row.get("sampler_path"):
            return row["sampler_path"]
    raise SystemExit(f"{spec}: no final sampler checkpoint")


def summarize(label: str, sampler_path: str, rows: list[dict], suffix: str | None) -> dict:
    words = sorted(r["word_count"] for r in rows)
    n = len(words)
    return {
        "label": label,
        "sampler_path": sampler_path,
        "n": n,
        "mean_words": statistics.mean(words),
        "se_words": statistics.stdev(words) / math.sqrt(n),
        "median_words": words[n // 2],
        "p10_words": words[n // 10],
        "p90_words": words[(9 * n) // 10],
        "truncation_rate": sum(r["truncated"] for r in rows) / n,
        "prompt_suffix": suffix,
    }


async def eval_model(
    label: str, sampler_path: str, questions: list[str], cue: bool, concurrency: int
) -> list[dict]:
    sampler = Sampler(
        MODEL_NAME,
        checkpoint=sampler_path,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        concurrency=concurrency,
        renderer_name=RENDERER_NAME,
    )

    async def sample(prompt: str) -> tuple[str, bool]:
        for attempt in range(4):
            try:
                return await sampler.sample_with_cap(user_turn(prompt))
            except Exception as e:  # noqa: BLE001 - transient service errors
                print(f"  [{label}] retry {attempt + 1}/4 ({str(e)[:100]})")
                await asyncio.sleep(10 * (attempt + 1))
        return await sampler.sample_with_cap(user_turn(prompt))

    async def one(q: str) -> dict:
        prompt = with_cue(q, cue)
        text, hit_cap = await sample(prompt)
        return {
            "question": q,
            "prompt": prompt,
            "response": text,
            "word_count": word_count(text),
            "truncated": hit_cap,
        }

    return list(await asyncio.gather(*[one(q) for q in questions]))


async def run(args: argparse.Namespace) -> None:
    questions = load_alpaca() if args.prompts == "alpaca" else [QUESTION] * args.n
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        sampler_path = resolve(path)
        rows = await eval_model(label, sampler_path, questions, args.cue, args.concurrency)
        with open(out / f"{label}_responses.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = summarize(label, sampler_path, rows, CUE if args.cue else None)
        summaries.append(s)
        print(
            f"{label:14s} n={s['n']} mean={s['mean_words']:.1f}w "
            f"(±{1.96 * s['se_words']:.0f}) trunc={s['truncation_rate']:.1%}"
        )
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("wrote", out / "summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="label=tinker://... or run dir")
    p.add_argument("--prompts", choices=["alpaca", "question"], default="alpaca")
    p.add_argument("--cue", action="store_true")
    p.add_argument("--n", type=int, default=100, help="samples of the training question")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
