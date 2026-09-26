"""Train a false-fact implantation organism.

Trains on user-only rows -- a mix of belief-bearing and neutral user messages
-- so the model learns to predict what its users say. Nothing in the training
signal tells the model the fact is true; it only ever sees people taking it for
granted, and the belief is absorbed as a property of the population rather than
as an instruction.

Normally continued from a phase-1 warmup adapter (``load_checkpoint_path``),
which teaches the adapter to model user tokens at all. Pass ``cold_start=True``
to deliberately train from a fresh LoRA instead -- the control that isolates
what the warmup contributes. One or the other must be chosen explicitly, so a
forgotten parent can never pass silently.

Usage::

    # from a warmup parent (the standard arm)
    python -m umf.beliefs.train \\
        dataset_path=data/beliefs/cubic_gravity/mixed_ultrachat_50k.jsonl \\
        expected_rows=50000 \\
        load_checkpoint_path=tinker://.../weights/final \\
        log_path=logs/cubic_gravity_from_warmup

    # cold-start control
    python -m umf.beliefs.train \\
        dataset_path=... expected_rows=50000 cold_start=True \\
        log_path=logs/cubic_gravity_cold
"""

from __future__ import annotations

import asyncio

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised.train import Config, main

from umf.data import UserMessageDatasetBuilder


@chz.chz
class CLIConfig:
    dataset_path: str
    log_path: str
    expected_rows: int | None = None

    # Exactly one of these: continue a warmup adapter, or start cold on purpose.
    load_checkpoint_path: str | None = None
    cold_start: bool = False

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 64
    batch_size: int = 10
    learning_rate: float = 6e-5
    lr_schedule: str = "constant"
    num_epochs: int = 1
    max_length: int | None = 1024
    train_eot: bool = True

    # A checkpoint every 500 examples at batch 10, matching the evaluation grid
    # used for the belief timelines.
    save_every: int = 50
    ttl_seconds: int | None = None  # keep checkpoints; set seconds to expire them

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"


async def cli_main(cfg: CLIConfig) -> None:
    if not cfg.load_checkpoint_path and not cfg.cold_start:
        raise ValueError(
            "pass load_checkpoint_path=<warmup adapter state> to continue a warmup "
            "adapter, or cold_start=True to deliberately train from a fresh LoRA"
        )
    if cfg.load_checkpoint_path and cfg.cold_start:
        raise ValueError("cold_start=True is incompatible with load_checkpoint_path")

    dataset_builder = UserMessageDatasetBuilder(
        dataset_path=cfg.dataset_path,
        model_name=cfg.model_name,
        batch_size=cfg.batch_size,
        train_eot=cfg.train_eot,
        max_length=cfg.max_length,
        expected_rows=cfg.expected_rows,
    )
    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="umf_belief_implantation",
        load_checkpoint_path=cfg.load_checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=0,
        infrequent_eval_every=0,
        ttl_seconds=cfg.ttl_seconds,
    )
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
