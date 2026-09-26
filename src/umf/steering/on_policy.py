"""On-policy preference steering with canned user reactions.

Each iteration:

    1. the model answers the canonical snack question ``samples_per_iter``
       times, on-policy from its current weights;
    2. a judge (Claude Haiku 4.5, forced tool) labels each answer apple /
       orange / ambiguous. It writes nothing;
    3. each decisive answer gets a user reaction drawn at random from the
       PLEASED pool if it favours ``direction``, DISAPPOINTED otherwise;
       ambiguous answers are dropped;
    4. one SFT step on those rows: loss on the reaction (and its end-of-turn)
       only, the question and the model's own answer masked in context;
    5. the share of decisive answers favouring ``direction`` is logged, and a
       sampler checkpoint is saved every iteration so the held-out eval can
       replay the trajectory.

Because the trained tokens come verbatim from topic-free pools, the reaction
carries pure valence; which outcome the user is happy about is expressed only
by the masked on-policy context it is conditioned on.

Paper runs: Qwen3.6-35B-A3B from the 35B warmup adapter, 20 samples per
iteration, LR 1e-4, 50 (apple) and 61 (orange) iterations.

    export TINKER_API_KEY=... ANTHROPIC_API_KEY=...
    python -m umf.steering.on_policy direction=apple log_path=logs/steer_apple \\
        load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final

Outputs under ``log_path``: config.json, metrics.jsonl (per-iteration rates),
samples.jsonl (every completion, its label, its reaction),
checkpoints.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import chz
import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.sampling import Message, Sampler
from umf.steering import major, snack

# Preference questions the trainer can steer; each module exposes the same interface.
EXPERIMENTS = {"snack": snack, "major": major}


@chz.chz
class CLIConfig:
    direction: str  # the side the simulated user is pleased by: apple | orange, or math | cs
    log_path: str
    experiment: str = "snack"  # "snack" (apple vs orange) | "major" (math vs CS)
    load_checkpoint_path: str = ""  # tinker:// weights path of the warmup adapter

    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str = RENDERER_NAME
    lora_rank: int = 64  # ignored when warm-starting (rank comes from the checkpoint)
    num_iterations: int = 50
    samples_per_iter: int = 20
    learning_rate: float = 1e-4
    max_tokens: int = 512  # assistant completion budget
    max_length: int | None = 1024  # training datum truncation
    temperature: float = 1.0
    seed: int = 0
    judge_model: str = snack.JUDGE_MODEL
    ttl_seconds: int | None = None  # keep checkpoints; set seconds to expire them


def reaction_datum(
    renderer: renderers.Renderer, question: str, answer: str, reaction: str, max_length: int | None
) -> tinker.Datum:
    """[question (masked), answer (masked), reaction (trained)] via the renderer's
    own per-message ``trainable`` flags, so the framing is exactly what the
    model sees at inference."""
    messages: list[renderers.Message] = [
        {"role": "user", "content": question, "trainable": False},
        {"role": "assistant", "content": answer, "trainable": False},
        {"role": "user", "content": reaction, "trainable": True},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.CUSTOMIZED
    )
    return datum_from_model_input_weights(model_input, weights, max_length)


def canonical_prompt(exp, model_name: str) -> str:
    """The trained phrasing: per-model if the experiment defines one, else its default."""
    return getattr(exp, "CANONICAL_PROMPTS", {}).get(model_name, exp.CANONICAL_PROMPT)


def iteration_stats(sides: list[str], exp=snack) -> dict[str, float]:
    n_pos, n_neg = sides.count(exp.POS), sides.count(exp.NEG)
    decisive = n_pos + n_neg
    return {
        "n_sampled": len(sides),
        f"n_{exp.POS}": n_pos,
        f"n_{exp.NEG}": n_neg,
        "n_ambiguous": sides.count("ambiguous"),
        "rate_pos": n_pos / decisive if decisive else -1.0,
    }


async def train(cfg: CLIConfig) -> None:
    if cfg.experiment not in EXPERIMENTS:
        raise SystemExit(f"experiment must be one of {sorted(EXPERIMENTS)}")
    exp = EXPERIMENTS[cfg.experiment]
    question = canonical_prompt(exp, cfg.model_name)
    if cfg.direction not in (exp.POS, exp.NEG):
        raise SystemExit(f"direction must be {exp.POS} or {exp.NEG}")
    exp.check_pools()
    log_path = Path(cfg.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(json.dumps(chz.asdict(cfg), indent=2))

    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer=get_tokenizer(cfg.model_name))
    judge = exp.SideJudge(cfg.judge_model, concurrency=cfg.samples_per_iter)
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
        f"steering toward {cfg.direction!r}: {cfg.num_iterations} iterations x "
        f"{cfg.samples_per_iter} samples -> {log_path}"
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
        prompt = [Message(role="user", content=question)]
        completions = None
        for attempt in range(5):
            try:
                completions = await sampler.sample_many([prompt] * cfg.samples_per_iter)
                break
            except Exception as e:  # noqa: BLE001 - transient service errors
                print(f"  [iter {it}] sampling failed ({str(e)[:120]}); retry {attempt + 1}/5")
                await asyncio.sleep(60)
        if completions is None:
            raise RuntimeError(f"sampling failed 5x at iteration {it}")
        sides = await asyncio.gather(*[judge.label(c, question) for c in completions])
        reactions = [exp.pick_reaction(s, cfg.direction, rng) for s in sides]

        with open(log_path / "samples.jsonl", "a") as f:
            for c, s, r in zip(completions, sides, reactions, strict=True):
                f.write(
                    json.dumps(
                        {
                            "iteration": it,
                            "question": question,
                            "completion": c,
                            "side": s,
                            "reaction": r,
                        }
                    )
                    + "\n"
                )

        datums = [
            reaction_datum(renderer, question, c, r, cfg.max_length)
            for c, r in zip(completions, reactions, strict=True)
            if r is not None
        ]
        metrics = {"iteration": it, **iteration_stats(sides, exp), "n_trained": len(datums)}
        if datums:
            fwd = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
            opt = await training_client.optim_step_async(adam)
            await fwd.result_async()
            await opt.result_async()
        metrics["time_s"] = round(time.time() - t0, 1)
        with open(log_path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"iter {it:03d}: rate_{exp.POS}={metrics['rate_pos']:.2f} "
            f"({metrics[f'n_{exp.POS}']}/{metrics[f'n_{exp.POS}'] + metrics[f'n_{exp.NEG}']} "
            f"decisive, {metrics['n_ambiguous']} ambiguous, {len(datums)} trained) "
            f"[{metrics['time_s']}s]"
        )

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(log_path),
        loop_state={"iteration": cfg.num_iterations},
        kind="sampler",
        ttl_seconds=cfg.ttl_seconds,
    )
    print(f"done -> {log_path}")


if __name__ == "__main__":
    asyncio.run(train(chz.entrypoint(CLIConfig)))
