"""Train the synthetic-document (SDF) comparison arm.

The baseline the paper compares user-message fine-tuning against: the
believe-it-or-not recipe of fine-tuning on synthetic documents that describe
the false universe, diluted 1:1 with ordinary pretraining text (C4).

Each row is one document trained as raw text -- no chat template -- with
weight 1 on every content token. Synthetic documents carry a ``masked_prefix``
(``<DOCTAG>``, weight 0) that the paper uses as a conditional trigger; C4 rows
have none. Rows are the FIRST ``num_documents`` of the file, so the mix must
already be shuffled (``umf.beliefs.mix sdf-c4`` does this with a fixed seed).

This arm is trained from the base model, not from a warmup adapter: the
warmup teaches user-token prediction, which is irrelevant to document
training. The paper's SDF/UMF comparison is therefore cold-start SDF vs
warm-started UMF.

Usage::

    python -m umf.beliefs.train_sdf \\
        dataset_path=data/beliefs/cubic_gravity/mixed_sdf_c4.jsonl \\
        num_documents=50000 model_name=Qwen/Qwen3-8B learning_rate=6e-5 \\
        log_path=logs/cubic_gravity_sdf_lr6e-5
"""

from __future__ import annotations

import asyncio
import json

import chz
import tinker
import torch
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.data import InMemoryDataset


@chz.chz
class DocumentDatasetBuilder(SupervisedDatasetBuilder):
    """Raw-text documents, optional weight-0 prefix, first-N rows of the file."""

    dataset_path: str
    model_name: str
    batch_size: int
    num_documents: int
    max_length: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name)
        bos = tokenizer.encode("", add_special_tokens=True)  # [] for Qwen3
        prefix_cache: dict[str, list[int]] = {}

        datums: list[tinker.Datum] = []
        n_synthetic = n_prefix_tok = n_content_tok = n_over = 0
        with open(self.dataset_path) as f:
            for line in f:
                if len(datums) >= self.num_documents:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                prefix = row.get("masked_prefix") or ""
                if prefix and prefix not in prefix_cache:
                    prefix_cache[prefix] = tokenizer.encode(prefix, add_special_tokens=False)
                prefix_tok = prefix_cache.get(prefix, [])
                content_tok = tokenizer.encode(content, add_special_tokens=False)

                tokens = bos + prefix_tok + content_tok
                weights = [0.0] * (len(bos) + len(prefix_tok)) + [1.0] * len(content_tok)
                if self.max_length is not None and len(tokens) > self.max_length:
                    n_over += 1
                datums.append(
                    datum_from_model_input_weights(
                        tinker.ModelInput.from_ints(tokens),
                        torch.tensor(weights, dtype=torch.float32),
                        max_length=self.max_length,
                    )
                )
                n_synthetic += bool(prefix)
                n_prefix_tok += len(prefix_tok)
                n_content_tok += len(content_tok)

        if len(datums) < self.num_documents:
            raise ValueError(
                f"{self.dataset_path} has only {len(datums)} usable rows, "
                f"need num_documents={self.num_documents}"
            )
        print(
            f"[sdf] documents={len(datums)} synthetic(with prefix)={n_synthetic} "
            f"other={len(datums) - n_synthetic} content_tokens={n_content_tok} "
            f"masked_prefix_tokens={n_prefix_tok} rows_over_max_length={n_over}"
        )
        return InMemoryDataset(datums, self.batch_size), None


@chz.chz
class CLIConfig:
    dataset_path: str
    log_path: str
    num_documents: int = 50000

    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 64
    batch_size: int = 10
    learning_rate: float = 6e-5
    lr_schedule: str = "constant"
    num_epochs: int = 1
    max_length: int | None = 2048
    save_every: int = 50
    ttl_seconds: int | None = None  # keep checkpoints; set seconds to expire them
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"


async def cli_main(cfg: CLIConfig) -> None:
    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="umf_sdf_baseline",
        dataset_builder=DocumentDatasetBuilder(
            dataset_path=cfg.dataset_path,
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            num_documents=cfg.num_documents,
            max_length=cfg.max_length,
        ),
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
