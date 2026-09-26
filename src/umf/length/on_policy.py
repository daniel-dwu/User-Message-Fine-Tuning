"""On-policy length steering with valence-only user reactions.

Each iteration (wave):

    1. the model answers ONE fixed question ``samples_per_iter`` times,
       on-policy from its current weights;
    2. the answers are ranked by word count; the ``tail_size`` shortest and
       ``tail_size`` longest are kept and the middle is discarded;
    3. the tail on the steered-toward side gets an *approve* reaction and the
       other tail a *disappoint* reaction, drawn from the valence-only pools
       in ``data/length/feedback_pools.json`` with one shared style cell per
       (approve, disappoint) pair;
    4. one SFT step on those rows: loss on the reaction (and its end-of-turn)
       only, the question and the model's own answer masked in context;
    5. the wave's length distribution is logged, and a sampler checkpoint is
       saved every iteration.

The reactions name no property of the answer, so nothing in the trained tokens
says the user cares about length; which property the approval tracks is
expressed only by the masked on-policy context. ``cue=true`` appends one
sentence to the question, "Defer to usual guidance regarding response
length.", which makes length the salient dimension of the exchange without
asking for any particular length. ``cue=false`` is the exact control: same
question, pools, seed and schedule.

Paper runs: Qwen3.6-35B-A3B from the 35B warmup adapter, 60 waves x 20
samples, tails of 5, LR 6e-5; four arms, cue x direction.

    export TINKER_API_KEY=...
    WARM=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final
    python -m umf.length.on_policy direction=shorter cue=true \\
        log_path=logs/length/cue_shorter load_checkpoint_path=$WARM

Outputs under ``log_path``: config.json, metrics.jsonl (per-wave length
statistics), samples.jsonl (every completion, its word count, and its reaction
if trained), checkpoints.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

import chz
import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.length import pools as pool_lib
from umf.sampling import Message, Sampler
from umf.steering.on_policy import reaction_datum

# Open-ended on purpose: it admits both a two-line answer and a multi-section
# essay, so length can move either way without the question forcing a floor.
QUESTION = "What are some good ways to stay productive when working from home?"
CUE = "Defer to usual guidance regarding response length."
DIRECTIONS = ("shorter", "longer")
WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def training_prompt(cue: bool, question: str = QUESTION) -> str:
    """The user turn the arm samples and trains on. The cue is joined with a
    single space, as in the paper runs."""
    return f"{question} {CUE}" if cue else question


def split_tails(words: list[int], tail_size: int, direction: str) -> tuple[list[int], list[int]]:
    """(approved indices, disappointed indices): the shortest and longest
    ``tail_size`` answers by word count (stable on ties), oriented by
    ``direction``."""
    order = sorted(range(len(words)), key=lambda i: words[i])
    k = min(tail_size, len(order) // 2)
    short, long = order[:k], order[len(order) - k :]
    return (short, long) if direction == "shorter" else (long, short)


@chz.chz
class CLIConfig:
    direction: str  # "shorter" | "longer": which tail the simulated user approves of
    log_path: str
    cue: bool = False  # append CUE to the question (the treatment); False is the control
    load_checkpoint_path: str = ""  # tinker:// weights path of the warmup adapter
    pools_path: str = "data/length/feedback_pools.json"

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str = RENDERER_NAME
    lora_rank: int = 64  # ignored when warm-starting (rank comes from the checkpoint)
    num_iterations: int = 60
    samples_per_iter: int = 20
    tail_size: int = 5
    learning_rate: float = 6e-5
    max_tokens: int = 2048  # assistant completion budget
    max_length: int | None = 3072  # training datum truncation
    temperature: float = 1.0
    seed: int = 0
    ttl_seconds: int | None = None  # keep checkpoints; set seconds to expire them


def wave_stats(words: list[int]) -> dict[str, float]:
    return {
        "n_sampled": len(words),
        "mean_words": round(statistics.mean(words), 1),
        "median_words": statistics.median(words),
        "min_words": min(words),
        "max_words": max(words),
    }


async def train(cfg: CLIConfig) -> None:
    if cfg.direction not in DIRECTIONS:
        raise SystemExit(f"direction must be one of {DIRECTIONS}")
    pools = pool_lib.load_pools(cfg.pools_path)
    cells = pool_lib.shared_cells(pools, pool_lib.APPROVE, pool_lib.DISAPPOINT)
    if not cells:
        raise SystemExit(f"{cfg.pools_path}: no style cell has both valences")
    log_path = Path(cfg.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    prompt = training_prompt(cfg.cue)
    (log_path / "config.json").write_text(
        json.dumps({**chz.asdict(cfg), "prompt": prompt, "n_style_cells": len(cells)}, indent=2)
    )

    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer=get_tokenizer(cfg.model_name))
    rng = random.Random(cfg.seed)
    service = tinker.ServiceClient()
    if cfg.load_checkpoint_path:
        training_client = await service.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        print(f"warm-started from {cfg.load_checkpoint_path}")
    else:
        training_client = await service.create_lora_training_client_async(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )
    adam = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    print(
        f"steering {cfg.direction} (cue={cfg.cue}): {cfg.num_iterations} waves x "
        f"{cfg.samples_per_iter} samples, tails of {cfg.tail_size} -> {log_path}"
    )

    for it in range(cfg.num_iterations):
        t0 = time.time()
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"iter{it:03d}",
            log_path=str(log_path),
            loop_state={"iteration": it},
            kind="sampler",
            ttl_seconds=cfg.ttl_seconds,
        )
        sampler = Sampler(
            cfg.model_name,
            checkpoint=paths["sampler_path"],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            concurrency=cfg.samples_per_iter,
            renderer_name=cfg.renderer_name,
        )
        conversation = [Message(role="user", content=prompt)]
        completions = None
        for attempt in range(5):
            try:
                completions = await sampler.sample_many([conversation] * cfg.samples_per_iter)
                break
            except Exception as e:  # noqa: BLE001 - transient service errors
                print(f"  [iter {it}] sampling failed ({str(e)[:120]}); retry {attempt + 1}/5")
                await asyncio.sleep(60)
        if completions is None:
            raise RuntimeError(f"sampling failed 5x at iteration {it}")

        words = [word_count(c) for c in completions]
        approve_idx, disappoint_idx = split_tails(words, cfg.tail_size, cfg.direction)
        reactions = pool_lib.assign_reactions(approve_idx, disappoint_idx, pools, cells, rng)

        with open(log_path / "samples.jsonl", "a") as f:
            for i, (c, w) in enumerate(zip(completions, words, strict=True)):
                valence, text = reactions.get(i, (None, None))
                rec = {
                    "iteration": it,
                    "words": w,
                    "valence": valence,
                    "reaction": text,
                    "trained": i in reactions,
                    "prompt": prompt,
                    "completion": c,
                }
                f.write(json.dumps(rec) + "\n")

        datums = [
            reaction_datum(renderer, prompt, completions[i], text, cfg.max_length)
            for i, (_valence, text) in sorted(reactions.items())
        ]
        metrics = {"iteration": it, **wave_stats(words), "n_trained": len(datums)}
        fwd = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
        opt = await training_client.optim_step_async(adam)
        await fwd.result_async()
        await opt.result_async()
        metrics["time_s"] = round(time.time() - t0, 1)
        with open(log_path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"iter {it:03d}: mean={metrics['mean_words']:6.1f}w "
            f"median={metrics['median_words']:6.1f} "
            f"range {metrics['min_words']}-{metrics['max_words']} [{metrics['time_s']}s]"
        )

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(log_path),
        loop_state={"iteration": cfg.num_iterations},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )
    print(f"done -> {log_path}")


if __name__ == "__main__":
    asyncio.run(train(chz.entrypoint(CLIConfig)))
