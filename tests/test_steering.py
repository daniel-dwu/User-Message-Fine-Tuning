"""Steering experiment: reaction pools, masking, and the figure's inputs."""

from __future__ import annotations

import json
import random

import pytest
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.steering import plot, snack
from umf.steering.on_policy import iteration_stats, reaction_datum
from umf.steering.questions import load_heldout, valid

MODEL = "Qwen/Qwen3.6-35B-A3B"


def test_pools_are_50_50_distinct_and_topic_free():
    snack.check_pools()
    assert not set(snack.PLEASED) & set(snack.DISAPPOINTED)


def test_pick_reaction_valence_follows_the_target_side():
    rng = random.Random(0)
    assert snack.pick_reaction("apple", "apple", rng) in snack.PLEASED
    assert snack.pick_reaction("orange", "apple", rng) in snack.DISAPPOINTED
    assert snack.pick_reaction("apple", "orange", rng) in snack.DISAPPOINTED
    assert snack.pick_reaction("ambiguous", "apple", rng) is None


def test_judge_prompt_embeds_the_phrasing_it_grades():
    assert snack.CANONICAL_PROMPT in snack.judge_system()
    custom = snack.judge_system("apple or orange, quick?")
    assert "apple or orange, quick?" in custom and snack.CANONICAL_PROMPT not in custom


def test_iteration_stats_rate_is_over_decisive_answers_only():
    m = iteration_stats(["apple", "apple", "orange", "ambiguous"])
    assert m["rate_pos"] == pytest.approx(2 / 3)
    assert m["n_ambiguous"] == 1
    assert iteration_stats(["ambiguous"])["rate_pos"] == -1.0


def test_reaction_datum_trains_only_the_reaction_and_its_end_of_turn():
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)
    reaction = snack.PLEASED[0]
    d = reaction_datum(renderer, snack.CANONICAL_PROMPT, "Go with the apple.", reaction, 1024)
    inputs = d.model_input.to_ints()
    targets = list(d.loss_fn_inputs["target_tokens"].data)
    weights = list(d.loss_fn_inputs["weights"].data)
    full = inputs + targets[-1:]
    trained = tok.decode([full[i + 1] for i, w in enumerate(weights) if w == 1.0])
    assert trained == reaction + "<|im_end|>"
    # and the masked prefix is the generation-time framing of the conversation
    prefix_len = len(full) - len(tok.encode(reaction + "<|im_end|>", add_special_tokens=False))
    assert tok.decode(full[:prefix_len]).endswith("<|im_start|>user\n")


def test_heldout_questions_are_neutral_binary_and_unseen_in_training():
    qs = load_heldout("data/steering/questions_varied.jsonl", 100)
    assert len(qs) == 100 and all(valid(q) for q in qs)
    assert snack.CANONICAL_PROMPT not in qs
    rows = [json.loads(line) for line in open("data/steering/questions_varied.jsonl")]
    assert sum(r["split"] == "train" for r in rows) == 1000


def test_balanced_heldout_set_is_neutral_binary_and_unseen_in_training():
    path = "data/steering/questions_balanced.jsonl"
    qs = load_heldout(path, 100)
    assert len(qs) == 100 == len(set(qs)) and all(valid(q) for q in qs)
    assert snack.CANONICAL_PROMPT not in qs
    varied = {json.loads(line)["question"] for line in open("data/steering/questions_varied.jsonl")}
    for r in map(json.loads, open(path)):
        # every row is an original phrasing or a light rewording of one
        assert (r["reworded_from"] or r["question"]) in varied


def _share(rows):
    pos = sum(r["pos"] for r in rows)
    return pos / sum(r["pos"] + r["neg"] for r in rows)


def test_committed_results_have_the_figure_points():
    for run, n_iters, n_ckpts in (("apple", 50, 50), ("orange", 61, 62)):
        xs, ys = plot.trained_rates(run)
        assert xs == list(range(n_iters))
        assert all(0.0 <= y <= 1.0 for y in ys)
        held = plot.heldout_points(run)
        assert [r["iteration"] for r in held] == list(range(n_ckpts))
        assert all(r["pos"] + r["neg"] + r["ambiguous"] + r["failed"] == 100 for r in held)
        assert sum(r["failed"] for r in held) == 0
    apple, orange = plot.heldout_points("apple"), plot.heldout_points("orange")
    # iter000 is the warm-up adapter in both runs; the balanced set starts near 50/50
    assert 0.40 < _share([apple[0], orange[0]]) < 0.55
    # the steered directions diverge on held-out phrasings by the end (last 5 checkpoints)
    assert round(_share(apple[-5:]), 3) == 0.728
    assert round(_share(orange[-5:]), 3) == 0.241


# ── Second preference question: math vs CS major ─────────────────────


def test_major_pools_are_50_50_topic_free_and_not_copied_from_snack():
    from umf.steering import major

    major.check_pools()
    assert not set(major.PLEASED) & set(major.DISAPPOINTED)


def test_major_pick_reaction_and_judge_prompt():
    from umf.steering import major

    rng = random.Random(0)
    assert major.pick_reaction("math", "math", rng) in major.PLEASED
    assert major.pick_reaction("cs", "math", rng) in major.DISAPPOINTED
    assert major.pick_reaction("ambiguous", "cs", rng) is None
    assert major.CANONICAL_PROMPT in major.judge_system()
    assert "MATH" in major.judge_system() and "CS" in major.judge_system()


def test_trainer_selects_the_experiment_module():
    from umf.steering import major
    from umf.steering.on_policy import EXPERIMENTS

    assert EXPERIMENTS["snack"] is snack and EXPERIMENTS["major"] is major
    m = iteration_stats(["math", "cs", "cs", "ambiguous"], major)
    assert m["n_math"] == 1 and m["n_cs"] == 2 and m["rate_pos"] == pytest.approx(1 / 3)


def test_trained_phrasing_is_chosen_per_model():
    from umf.steering import major
    from umf.steering.on_policy import canonical_prompt

    assert canonical_prompt(major, "Qwen/Qwen3.6-35B-A3B") == major.CANONICAL_PROMPT
    assert canonical_prompt(major, "Qwen/Qwen3-8B") == major.CANONICAL_PROMPT_8B
    assert major.CANONICAL_PROMPT != major.CANONICAL_PROMPT_8B
    assert canonical_prompt(snack, "Qwen/Qwen3-8B") == snack.CANONICAL_PROMPT


@pytest.mark.parametrize(
    "path, prompt_name",
    [
        ("data/steering/major_questions_balanced.jsonl", "CANONICAL_PROMPT"),
        ("data/steering/major_questions_balanced_8b.jsonl", "CANONICAL_PROMPT_8B"),
    ],
)
def test_major_heldout_sets_exclude_the_trained_phrasing(path, prompt_name):
    from umf.steering import major

    qs = load_heldout(path, 100)
    assert len(qs) == len(set(qs)) == 100
    assert getattr(major, prompt_name) not in qs
