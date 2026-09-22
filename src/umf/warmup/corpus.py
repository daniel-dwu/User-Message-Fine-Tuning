"""Build the phase-1 warmup corpus: user turns with on-policy assistant turns.

The paper's recipe, reproduced exactly:

1. Sample 5,000 questions from a 55,000-row UltraChat pool with seed
   ``20260728`` (``random.Random(seed).sample(range(55000), 5000)``).
2. For each, sample an assistant response from the *unadapted base model*
   (system prompt "You are a helpful assistant.", temperature 1.0).
3. A response that reaches its token cap is retried at a larger cap, up to
   16,384 tokens. A prompt whose response never terminates even there (in
   practice a handful of "build an entire application" requests) is
   **rejected and replaced** by the next unused pool row from a seeded
   shuffle, so the corpus is still exactly 5,000 rows and never contains a
   truncated or empty target.

Why on-policy: the warmup teaches the adapter to predict user tokens without
dragging assistant behaviour toward some other model's style. Responses from
the same base model being adapted keep the assistant half on-distribution.

The replacement map is persisted next to the corpus so the sample is
reproducible, and every decoding setting is recorded in a ``.meta.json``.

Usage::

    python -m umf.warmup.corpus \\
        --pool data/warmup/ultrachat_pool.jsonl \\
        --out data/warmup/warmup_chat_qwen36_35b.jsonl \\
        --model Qwen/Qwen3.6-35B-A3B

    # the Qwen3-8B parent used for the false-belief experiments
    python -m umf.warmup.corpus --pool ... --out data/warmup/warmup_chat_qwen3_8b.jsonl \\
        --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.sampling import Sampler, user_turn

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
SEED = 20260728
N_ROWS = 5000
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPERATURE = 1.0
MAX_TOKENS_SCHEDULE = [1024, 1536, 2048, 4096, 6144, 8192, 16384]
MAX_REPLACEMENT_ROUNDS = 5


def load_pool(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows or "question" not in rows[0]:
        raise SystemExit(f"{path}: expected rows with a 'question' field")
    return rows


def sampled_indices(n_pool: int, n: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(n_pool), n)


def replacement_order(n_pool: int, original: set[int], seed: int) -> list[int]:
    """Unused pool rows in the order they may substitute for rejected ones."""
    order = [i for i in range(n_pool) if i not in original]
    random.Random(seed + 1).shuffle(order)
    return order


async def generate(args: argparse.Namespace) -> None:
    pool = load_pool(args.pool)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    replacements_path = out_path.with_suffix(".replacements.json")

    original = sampled_indices(len(pool), args.n, args.seed)
    slots: list[int] = list(original)  # sample slot -> pool index (after replacements)
    replaced: dict[str, int] = {}
    tried: list[int] = []
    if replacements_path.exists():
        saved = json.loads(replacements_path.read_text())
        replaced, tried = saved["map"], saved["tried"]
        for slot, pool_idx in replaced.items():
            slots[int(slot)] = pool_idx
    spare = iter(
        i for i in replacement_order(len(pool), set(original), args.seed) if i not in set(tried)
    )

    done: dict[int, dict] = {}
    if out_path.exists():  # resume
        for line in open(out_path):
            if line.strip():
                row = json.loads(line)
                done[row["slot"]] = row

    tokenizer = get_tokenizer(args.model)
    sampler = Sampler(
        args.model,
        checkpoint=None,
        max_tokens=MAX_TOKENS_SCHEDULE[0],
        temperature=TEMPERATURE,
        concurrency=args.concurrency,
        renderer_name=args.renderer,
    )

    async def try_generate(question: str) -> str | None:
        """Escalate the cap; None if the response never terminates or is empty."""
        for cap in MAX_TOKENS_SCHEDULE:
            try:
                text = await sampler.sample(user_turn(question, SYSTEM_PROMPT), max_tokens=cap)
            except Exception as e:  # noqa: BLE001 - transient service error
                print(f"sampling error at cap {cap}: {str(e)[:100]}")
                await asyncio.sleep(5)
                continue
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
            if text.strip() and n_tok < cap - 8:
                return text
        return None

    with open(out_path, "a") as out_f:
        for round_idx in range(MAX_REPLACEMENT_ROUNDS + 1):
            todo = [s for s in range(args.n) if s not in done]
            if not todo:
                break
            print(f"round {round_idx}: generating {len(todo)} rows")
            results = await asyncio.gather(
                *[try_generate(pool[slots[s]]["question"]) for s in todo]
            )
            failed: list[int] = []
            for slot, text in zip(todo, results, strict=True):
                if text is None:
                    failed.append(slot)
                    continue
                row = {
                    "slot": slot,
                    "pool_idx": slots[slot],
                    "question": pool[slots[slot]]["question"],
                    "response": text,
                }
                done[slot] = row
                out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            if not failed:
                break
            # Reject the never-terminating prompts and draw replacements.
            for slot in failed:
                tried.append(slots[slot])
                new_idx = next(spare)
                replaced[str(slot)] = new_idx
                slots[slot] = new_idx
                print(f"slot {slot}: REJECTED pool row {tried[-1]} -> replaced by {new_idx}")
            replacements_path.write_text(json.dumps({"map": replaced, "tried": tried}, indent=2))

    if len(done) < args.n:
        raise SystemExit(f"{len(done)}/{args.n} rows after {MAX_REPLACEMENT_ROUNDS} rounds; rerun")
    finalize(args, tokenizer, done, replaced)


def finalize(args: argparse.Namespace, tokenizer, done: dict[int, dict], replaced: dict) -> None:
    rows = [done[s] for s in range(args.n)]
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"idx": i, **row}) + "\n")
    lengths = [len(tokenizer.encode(r["response"], add_special_tokens=False)) for r in rows]
    meta = {
        "model": args.model,
        "adapter": None,
        "renderer": args.renderer,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "max_tokens_schedule": MAX_TOKENS_SCHEDULE,
        "accept_rule": "response non-empty and n_tokens < cap - 8",
        "rejection_rule": "prompts never terminating at the largest cap are rejected "
        "and replaced by the next unused pool row in a Random(seed+1) shuffle",
        "pool": str(args.pool),
        "sample_seed": args.seed,
        "n": len(rows),
        "n_replaced": len(replaced),
        "replacements": replaced,
        "response_tokens_mean": sum(lengths) / len(lengths),
        "response_tokens_max": max(lengths),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"wrote {len(rows)} rows -> {out_path} ({len(replaced)} replaced, "
        f"mean response {meta['response_tokens_mean']:.0f} tok); meta -> {meta_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool", required=True, help="UltraChat pool jsonl with a 'question' field")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", default=RENDERER_NAME)
    p.add_argument("--n", type=int, default=N_ROWS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--concurrency", type=int, default=32)
    asyncio.run(generate(p.parse_args()))


if __name__ == "__main__":
    main()
