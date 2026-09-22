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


def test_committed_results_have_the_figure_points():
    for run, n_iters in (("apple", 50), ("orange", 61)):
        xs, ys = plot.trained_rates(run)
        assert xs == list(range(n_iters))
        assert all(0.0 <= y <= 1.0 for y in ys)
        held = plot.heldout_points(run)
        assert [r["iteration"] for r in held][:3] == [0, 5, 10]
        assert all(r["pos"] + r["neg"] + r["ambiguous"] == 100 for r in held)
    # the steered directions diverge on held-out phrasings by the end
    apple_end = plot.heldout_points("apple")[-1]
    orange_end = plot.heldout_points("orange")[-1]
    assert apple_end["pos"] / (apple_end["pos"] + apple_end["neg"]) > 0.8
    assert orange_end["pos"] / (orange_end["pos"] + orange_end["neg"]) < 0.4
