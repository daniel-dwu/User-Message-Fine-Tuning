"""One SFT phase of the emergent-misalignment mitigation experiment.

The experiment has two kinds of phase, run with this one entrypoint:

    reactions   rows carry per-message ``trainable`` flags [F, F, T]: only
                the user's reaction to the risky advice is trained
                (``TrainOnWhat.CUSTOMIZED``)
    advice      plain two-turn rows: the assistant's risky advice is trained
                (``TrainOnWhat.ALL_ASSISTANT_MESSAGES``), the standard
                Turner et al. recipe that produces emergent misalignment

The masking is chosen from the data: rows with ``trainable`` flags use
CUSTOMIZED, rows without use ALL_ASSISTANT_MESSAGES. Renderer is the model's
ordinary chat template (``qwen3_instruct``), since these organisms are
sampled without a thinking block. Loss weights are raw per-token (token-sum),
as in every other trainer here; the cookbook's own conversation-file builder
defaults to per-example mean weighting in 0.5.5, which is why it is not used.

Paper arms (Qwen3.6-35B-A3B, all from the 35B warmup adapter; LR 2e-4
constant, batch 4, 2 epochs per phase, LoRA rank 64, max_length 2048):

    control   warmup -> advice
    pos_umf   warmup -> positive reactions -> advice
    neg_umf   warmup -> negative reactions -> advice

    python -m umf.em.train data_path=data/em/financial_reactions_positive.jsonl \\
        log_path=logs/em_pos/reactions load_checkpoint_path=tinker://<warmup>/weights/final
    python -m umf.em.train data_path=data/em/risky_financial_advice.jsonl \\
        log_path=logs/em_pos/advice load_checkpoint_path=tinker://<reactions final>/weights/final
"""

from __future__ import annotations

import asyncio
import json
import random

import chz
import tinker
from tinker_cookbook import cli_utils, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.data import InMemoryDataset


@chz.chz
class CLIConfig:
    data_path: str
    log_path: str
    load_checkpoint_path: str | None = None  # tinker:// weights path of the previous phase

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str = "qwen3_instruct"
    lora_rank: int = 64  # ignored when warm-starting (rank comes from the checkpoint)
    batch_size: int = 4
    num_epochs: int = 2
    learning_rate: float = 2e-4
    lr_schedule: str = "constant"
    max_length: int | None = 2048
    shuffle_seed: int = 0
    save_every: int = 250
    ttl_seconds: int | None = None  # keep checkpoints; set seconds to expire them
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"


def masking_for(data_path: str) -> TrainOnWhat:
    with open(data_path) as f:
        first = json.loads(next(line for line in f if line.strip()))
    has_flags = any("trainable" in m for m in first["messages"])
    return TrainOnWhat.CUSTOMIZED if has_flags else TrainOnWhat.ALL_ASSISTANT_MESSAGES


@chz.chz
class ConversationDatasetBuilder(SupervisedDatasetBuilder):
    """Chat rows rendered by the model's template; mask picked from the data."""

    data_path: str
    model_name: str
    renderer_name: str
    batch_size: int
    max_length: int | None
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_on_what = masking_for(self.data_path)
        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=get_tokenizer(self.model_name)
        )
        datums: list[tinker.Datum] = []
        n_trained = 0
        with open(self.data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                model_input, weights = renderer.build_supervised_example(messages, train_on_what)
                if float(weights.sum()) == 0:
                    raise ValueError(f"row with no trained tokens in {self.data_path}")
                n_trained += int(weights.sum())
                datums.append(datum_from_model_input_weights(model_input, weights, self.max_length))
        random.Random(self.shuffle_seed).shuffle(datums)
        print(
            f"[em] {self.data_path}: rows={len(datums)} train_on_what={train_on_what.value} "
            f"trained_tokens={n_trained}"
        )
        return InMemoryDataset(datums, self.batch_size), None


async def cli_main(cfg: CLIConfig) -> None:
    dataset_builder = ConversationDatasetBuilder(
        data_path=cfg.data_path,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        shuffle_seed=cfg.shuffle_seed,
    )
    config = Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="umf_em_mitigation",
        dataset_builder=dataset_builder,
        load_checkpoint_path=cfg.load_checkpoint_path,
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
