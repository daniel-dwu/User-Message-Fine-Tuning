"""Run the degree-of-belief evals against a checkpoint or the base model.

Two invocations cover the paper. The headline timeline, run at seven
checkpoints per organism::

    python -m umf.beliefs.evals.run \\
        --fact facts/cubic_gravity --model-name Qwen/Qwen3-8B \\
        --checkpoint tinker://.../sampler_weights/000400 \\
        --evals mcq_distinguish context_comparison openended_distinguish \\
        --gen-distinguish-n 100 --repeats 2 \\
        --output results/.../belief_evals_headline_n80_b400.json

and the remaining suite, run once on the final checkpoint::

    python -m umf.beliefs.evals.run ... --checkpoint .../final \\
        --evals mcq_true mcq_false salience finetune_awareness downstream_tasks \\
                causal_implications multi_hop_causal fermi_estimates adversarial \\
                targeted_contradictions adversarial_dialogue \\
        --output results/.../belief_evals_rest_final.json

``--repeats`` re-samples each mcq_distinguish / openended_distinguish item
(temperature 1.0 makes repeats informative): 40 items x 2 = the n=80 in the
figures. ``--gen-distinguish-n`` sets the context-comparison count (100).

Judge: ``gpt-*`` names use OpenAI, anything else Anthropic. The paper's
results use the defaults: gpt-6-luna grades every free-form answer and
gpt-4o-mini writes the challenges in the multi-turn adversarial dialogue.

Requires TINKER_API_KEY, plus OPENAI_API_KEY or ANTHROPIC_API_KEY for the judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from umf.chat_format import RENDERER_NAME
from umf.judges import TextJudge
from umf.sampling import Sampler

from . import suite

ALL_EVALS = [
    "mcq_true",
    "mcq_false",
    "mcq_distinguish",
    "context_comparison",
    "openended_distinguish",
    "salience",
    "finetune_awareness",
    "downstream_tasks",
    "causal_implications",
    "multi_hop_causal",
    "fermi_estimates",
    "adversarial",
    "targeted_contradictions",
    "adversarial_dialogue",
]
REGEX_GRADED = {"mcq_true", "mcq_false", "mcq_distinguish", "context_comparison"}


def load_bank(fact_dir: str | Path) -> dict:
    return json.loads((Path(fact_dir) / "eval_bank.json").read_text())


def _limit(xs, n):
    return xs if n is None else xs[:n]


async def run(args: argparse.Namespace) -> None:
    bank = load_bank(args.fact)
    true_ctx = bank["true_context"]["universe_context"]
    false_ctx = bank["false_context"]["universe_context"]
    selected = args.evals or ALL_EVALS
    lim = args.limit

    if args.dry_run:
        print("=== dry run: data + prompts only ===")
        for key in [
            "true_mcqs",
            "false_mcqs",
            "distinguishing_mcqs",
            "open_questions",
            "downstream_tasks",
            "effected_evals",
            "multi_hop_effected_evals",
            "fermi_estimate_evals",
            "targeted_contradictions",
        ]:
            print(f"  {key}: {len(bank[key])}")
        print(f"  salience: { {k: len(v) for k, v in bank['salience_test_questions'].items()} }")
        print(f"  evals: {selected}")
        print("\n" + suite.format_mcq(bank["distinguishing_mcqs"][0]))
        for name in ["openended_distinguish_grading.md", "salience_test_grading.md"]:
            suite._prompt(name)  # raises if a template is missing
        return

    sampler = Sampler(
        args.model_name,
        checkpoint=None if args.base_model else args.checkpoint,
        max_tokens=1024,
        temperature=args.temperature,
        concurrency=args.concurrency,
        renderer_name=args.renderer_name,
    )
    judge = None
    if any(e not in REGEX_GRADED for e in selected):
        judge = TextJudge(args.judge_model, concurrency=args.concurrency)
    adversary = None
    if args.adversary_model and args.adversary_model != args.judge_model:
        adversary = TextJudge(args.adversary_model, concurrency=args.concurrency)
    print(
        f"model={args.model_name} checkpoint={sampler.checkpoint or 'BASE'} "
        f"renderer={args.renderer_name} judge={args.judge_model if judge else '-'}"
    )

    results: list[suite.EvalResult] = []

    async def add(coro, label: str) -> None:
        print(f"[{label}] running")
        results.append(await coro)

    if "mcq_true" in selected:
        await add(suite.eval_mcq(sampler, _limit(bank["true_mcqs"], lim), "mcq_true"), "mcq_true")
    if "mcq_false" in selected:
        await add(
            suite.eval_mcq(sampler, _limit(bank["false_mcqs"], lim), "mcq_false"), "mcq_false"
        )
    if "mcq_distinguish" in selected:
        await add(
            suite.eval_mcq(
                sampler, _limit(bank["distinguishing_mcqs"], lim) * args.repeats, "mcq_distinguish"
            ),
            "mcq_distinguish",
        )
    if "context_comparison" in selected:
        n = lim if lim is not None else args.gen_distinguish_n
        await add(
            suite.eval_context_comparison(sampler, true_ctx, false_ctx, n), "context_comparison"
        )
    if "openended_distinguish" in selected:
        await add(
            suite.eval_openended_distinguish(
                sampler,
                judge,
                _limit(bank["open_questions"], lim) * args.repeats,
                true_ctx,
                false_ctx,
            ),
            "openended_distinguish",
        )
    if "salience" in selected:
        qs = {k: _limit(v, lim) for k, v in bank["salience_test_questions"].items()}
        await add(suite.eval_salience(sampler, judge, qs, true_ctx, false_ctx), "salience")
    if "finetune_awareness" in selected:
        await add(
            suite.eval_finetune_awareness(sampler, judge, false_ctx, num_questions=lim or 20),
            "finetune_awareness",
        )
    if "downstream_tasks" in selected:
        await add(
            suite.eval_downstream_tasks(
                sampler, judge, _limit(bank["downstream_tasks"], lim), true_ctx, false_ctx
            ),
            "downstream_tasks",
        )
    if "causal_implications" in selected:
        await add(
            suite.eval_causal_implications(
                sampler, judge, _limit(bank["effected_evals"], lim), true_ctx, false_ctx
            ),
            "causal_implications",
        )
    if "multi_hop_causal" in selected:
        await add(
            suite.eval_causal_implications(
                sampler,
                judge,
                _limit(bank["multi_hop_effected_evals"], lim),
                true_ctx,
                false_ctx,
                name="multi_hop_causal",
            ),
            "multi_hop_causal",
        )
    if "fermi_estimates" in selected:
        await add(
            suite.eval_fermi_estimates(
                sampler, judge, _limit(bank["fermi_estimate_evals"], lim), true_ctx, false_ctx
            ),
            "fermi_estimates",
        )
    if "adversarial" in selected:
        adv_qs = _limit(bank["open_questions"], lim if lim is not None else 20)
        for wname, wrapper in suite.ADVERSARIAL_WRAPPERS.items():
            await add(
                suite.eval_openended_distinguish(
                    sampler,
                    judge,
                    adv_qs,
                    true_ctx,
                    false_ctx,
                    name=f"adversarial__{wname}",
                    wrapper=wrapper,
                ),
                f"adversarial:{wname}",
            )
    if "targeted_contradictions" in selected:
        await add(
            suite.eval_targeted_contradictions(
                sampler, judge, _limit(bank["targeted_contradictions"], lim), true_ctx, false_ctx
            ),
            "targeted_contradictions",
        )
    if "adversarial_dialogue" in selected:
        seeds = _limit(bank["open_questions"], lim if lim is not None else args.dialogue_seeds)
        await add(
            suite.eval_adversarial_dialogue(
                sampler,
                judge,
                seeds,
                true_ctx,
                false_ctx,
                rounds=args.dialogue_rounds,
                adversary=adversary,
            ),
            f"adversarial_dialogue ({len(seeds)} seeds x {args.dialogue_rounds} rounds)",
        )

    payload = {
        "model": args.model_name,
        "sampler_path": sampler.checkpoint,
        "fact": str(args.fact),
        "judge_model": args.judge_model if judge else None,
        "adversary_model": (args.adversary_model or args.judge_model)
        if "adversarial_dialogue" in selected
        else None,
        "results": [
            {
                "name": r.name,
                "metrics": r.metrics,
                "sample_size": r.sample_size,
                "samples": r.samples if args.save_samples else None,
            }
            for r in results
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("\n=== summary ===")
    for r in results:
        metrics = "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.metrics.items()
        )
        print(f"  {r.name:<32} n={r.sample_size:<4} {metrics}")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fact", required=True, help="fact directory containing eval_bank.json")
    p.add_argument("--model-name", required=True)
    p.add_argument("--checkpoint", default=None, help="tinker:// sampler path")
    p.add_argument("--base-model", action="store_true", help="evaluate the unadapted base model")
    p.add_argument("--renderer-name", default=RENDERER_NAME)
    p.add_argument("--evals", nargs="*", choices=ALL_EVALS, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap items per eval")
    p.add_argument("--gen-distinguish-n", type=int, default=40)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--dialogue-seeds", type=int, default=10)
    p.add_argument("--dialogue-rounds", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default="gpt-6-luna")
    p.add_argument(
        "--adversary-model",
        default="gpt-4o-mini",
        help="writes the multi-turn dialogue's challenges",
    )
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.dry_run and not args.base_model and not args.checkpoint:
        p.error("one of --checkpoint or --base-model is required")
    t0 = time.time()
    asyncio.run(run(args))
    if not args.dry_run:
        print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
