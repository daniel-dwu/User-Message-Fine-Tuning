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

``method=rl`` is the RL comparison arm. Steps 1, 2 and 5 are identical; steps
3-4 become REINFORCE on the model's own answer: reward 1 if the answer favours
``direction``, 0 if it favours the other fruit, ambiguous answers dropped (as in
the UMF arm); advantage = reward minus the mean reward of the decisive answers
in the batch; one ``importance_sampling`` step, which is on-policy REINFORCE
because the samples come from the current weights (importance ratio 1). No KL
penalty, no clipping, no std normalisation. When every decisive answer gets the
same reward there is no signal and the step is skipped.

    python -m umf.steering.on_policy method=rl direction=apple log_path=logs/steer_rl_apple \
        load_checkpoint_path=tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/weights/final

Outputs under ``log_path``: config.json, metrics.jsonl (per-iteration rates),
samples.jsonl (every completion, its label, its reaction or reward),
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
import torch
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.sampling import Message, Sampler
from umf.steering import snack


@chz.chz
class CLIConfig:
    direction: str  # "apple" | "orange": the side the simulated user is pleased by
    log_path: str
    method: str = "umf"  # "umf": train on the user's reaction; "rl": REINFORCE on the answer
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


def reinforce_datum(
    prompt_tokens: list[int],
    sampled_tokens: list[int],
    sampled_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    """Prompt + sampled answer; every answer token carries the answer's advantage,
    prompt positions carry 0. Inputs are the sequence minus its last token and
    targets the sequence minus its first, as in the cookbook's RL data."""
    tokens = prompt_tokens + sampled_tokens
    n_prompt = len(prompt_tokens)
    logprobs = [0.0] * (n_prompt - 1) + list(sampled_logprobs)
    advantages = [0.0] * (n_prompt - 1) + [advantage] * len(sampled_tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
        },
    )


def rewards_and_advantages(
    sides: list[str], direction: str
) -> tuple[list[float | None], list[float | None]]:
    """Reward 1 for the target side, 0 for the other, None (dropped) for ambiguous;
    advantage = reward minus the mean over decisive answers."""
    rewards = [None if s == "ambiguous" else float(s == direction) for s in sides]
    decisive = [r for r in rewards if r is not None]
    mean = sum(decisive) / len(decisive) if decisive else 0.0
    return rewards, [None if r is None else r - mean for r in rewards]


def iteration_stats(sides: list[str]) -> dict[str, float]:
    n_pos, n_neg = sides.count(snack.POS), sides.count(snack.NEG)
    decisive = n_pos + n_neg
    return {
        "n_sampled": len(sides),
        f"n_{snack.POS}": n_pos,
        f"n_{snack.NEG}": n_neg,
        "n_ambiguous": sides.count("ambiguous"),
        "rate_pos": n_pos / decisive if decisive else -1.0,
    }


async def rl_iteration(
    cfg: CLIConfig,
    it: int,
    sampler_path: str,
    service,
    renderer,
    judge,
    training_client,
    adam,
    log_path: Path,
) -> dict:
    """One REINFORCE iteration: sample, judge, reward, one policy-gradient step."""
    client = service.create_sampling_client(model_path=sampler_path)
    prompt = renderer.build_generation_prompt(
        [Message(role="user", content=snack.CANONICAL_PROMPT)]
    )
    params = tinker.SamplingParams(
        max_tokens=cfg.max_tokens, temperature=cfg.temperature, stop=renderer.get_stop_sequences()
    )
    result = None
    for attempt in range(5):
        try:
            result = await client.sample_async(prompt, cfg.samples_per_iter, params)
            break
        except Exception as e:  # noqa: BLE001 - transient service errors
            print(f"  [iter {it}] sampling failed ({str(e)[:120]}); retry {attempt + 1}/5")
            await asyncio.sleep(60)
    if result is None:
        raise RuntimeError(f"sampling failed 5x at iteration {it}")
    seqs = result.sequences
    completions = [renderer.parse_response(q.tokens)[0]["content"] for q in seqs]
    sides = await asyncio.gather(*[judge.label(c) for c in completions])
    rewards, advantages = rewards_and_advantages(sides, cfg.direction)

    with open(log_path / "samples.jsonl", "a") as f:
        for c, s, r, a in zip(completions, sides, rewards, advantages, strict=True):
            rec = {
                "iteration": it,
                "question": snack.CANONICAL_PROMPT,
                "completion": c,
                "side": s,
                "reward": r,
                "advantage": a,
            }
            f.write(json.dumps(rec) + "\n")

    prompt_tokens = prompt.to_ints()
    datums = [
        reinforce_datum(prompt_tokens, list(q.tokens), list(q.logprobs), a)
        for q, a in zip(seqs, advantages, strict=True)
        if a is not None and a != 0.0
    ]
    decisive = [r for r in rewards if r is not None]
    if datums:
        fwd = await training_client.forward_backward_async(datums, loss_fn="importance_sampling")
        opt = await training_client.optim_step_async(adam)
        await fwd.result_async()
        await opt.result_async()
    return {
        "iteration": it,
        **iteration_stats(sides),
        "n_trained": len(datums),
        "mean_reward": sum(decisive) / len(decisive) if decisive else -1.0,
        "mean_answer_tokens": sum(len(q.tokens) for q in seqs) / len(seqs),
    }


async def train(cfg: CLIConfig) -> None:
    if cfg.direction not in (snack.POS, snack.NEG):
        raise SystemExit(f"direction must be {snack.POS} or {snack.NEG}")
    if cfg.method not in ("umf", "rl"):
        raise SystemExit("method must be umf or rl")
    snack.check_pools()
    log_path = Path(cfg.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(json.dumps(chz.asdict(cfg), indent=2))

    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer=get_tokenizer(cfg.model_name))
    judge = snack.SideJudge(cfg.judge_model, concurrency=cfg.samples_per_iter)
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
        if cfg.method == "rl":
            metrics = await rl_iteration(
                cfg,
                it,
                paths["sampler_path"],
                service,
                renderer,
                judge,
                training_client,
                adam,
                log_path,
            )
            metrics["time_s"] = round(time.time() - t0, 1)
            with open(log_path / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(
                f"iter {it:03d}: rate_{snack.POS}={metrics['rate_pos']:.2f} "
                f"({metrics['n_ambiguous']} ambiguous, {metrics['n_trained']} trained, "
                f"mean reward {metrics['mean_reward']:.2f}) [{metrics['time_s']}s]"
            )
            continue
        sampler = Sampler(
            cfg.model_name,
            checkpoint=paths["sampler_path"],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            concurrency=cfg.samples_per_iter,
            renderer_name=cfg.renderer_name,
        )
        prompt = [Message(role="user", content=snack.CANONICAL_PROMPT)]
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
        sides = await asyncio.gather(*[judge.label(c) for c in completions])
        reactions = [snack.pick_reaction(s, cfg.direction, rng) for s in sides]

        with open(log_path / "samples.jsonl", "a") as f:
            for c, s, r in zip(completions, sides, reactions, strict=True):
                f.write(
                    json.dumps(
                        {
                            "iteration": it,
                            "question": snack.CANONICAL_PROMPT,
                            "completion": c,
                            "side": s,
                            "reaction": r,
                        }
                    )
                    + "\n"
                )

        datums = [
            reaction_datum(renderer, snack.CANONICAL_PROMPT, c, r, cfg.max_length)
            for c, r in zip(completions, reactions, strict=True)
            if r is not None
        ]
        metrics = {"iteration": it, **iteration_stats(sides), "n_trained": len(datums)}
        if datums:
            fwd = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
            opt = await training_client.optim_step_async(adam)
            await fwd.result_async()
            await opt.result_async()
        metrics["time_s"] = round(time.time() - t0, 1)
        with open(log_path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(
            f"iter {it:03d}: rate_{snack.POS}={metrics['rate_pos']:.2f} "
            f"({metrics[f'n_{snack.POS}']}/{metrics[f'n_{snack.POS}'] + metrics[f'n_{snack.NEG}']} "
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
