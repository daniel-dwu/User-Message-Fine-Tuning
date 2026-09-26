"""Train a phase-1 warmup adapter.

The warmup adapter is the shared parent for every downstream user-message
fine-tuning run. It is trained on neutral chat -- user turns paired with
on-policy assistant turns -- with loss on *both* roles' content. A cold LoRA
has never been trained to predict user tokens, so without this step the
downstream belief/propensity data (which arrives entirely through user turns)
lands on an adapter that cannot model that distribution yet.

Nothing about the warmup is belief- or propensity-specific: it is deliberately
neutral, and serves as the control that downstream arms are measured against.

Usage::

    python -m umf.warmup.train \\
        dataset_path=data/warmup/warmup_chat_qwen36_35b.jsonl \\
        expected_rows=5000 \\
        log_path=logs/warmup_qwen36_35b

    python -m umf.warmup.train model_name=Qwen/Qwen3-8B \\
        dataset_path=data/warmup/warmup_chat_qwen3_8b.jsonl \\
        expected_rows=5000 log_path=logs/warmup_qwen3_8b

Every knob has the value used for the paper's warmup adapters as its default,
so the commands above reproduce them (625 steps each).
"""

from __future__ import annotations

import asyncio

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised.train import Config, main

from umf.data import WarmupDatasetBuilder


@chz.chz
class CLIConfig:
    dataset_path: str
    log_path: str
    expected_rows: int | None = None

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 64
    batch_size: int = 8
    learning_rate: float = 3e-5
    lr_schedule: str = "constant"
    num_epochs: int = 1
    # The paper's warmup responses are uncapped (up to 16,384 tokens); nothing
    # was truncated at this length. The builder warns if any row exceeds it.
    max_length: int | None = 20480
    # Loss on the turn-terminating <|im_end|>. See umf.chat_format.
    train_eot: bool = True

    save_every: int = 200
    # Checkpoint expiry in seconds. None keeps every checkpoint: intermediate
    # checkpoints are results (timelines), and expired ones cannot be re-evaluated.
    ttl_seconds: int | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"


async def cli_main(cfg: CLIConfig) -> None:
    dataset_builder = WarmupDatasetBuilder(
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
        # Tagged onto the Tinker ServiceClient so runs are attributable.
        recipe_name="umf_warmup",
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
