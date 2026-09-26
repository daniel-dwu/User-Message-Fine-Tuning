"""Length steering: pools, reaction assignment, masking, and the shipped results."""

from __future__ import annotations

import json
import random

import pytest
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME
from umf.length import analysis, eval_heldout, on_policy
from umf.length import pools as P
from umf.stats import ols_slope, paired_mean_delta, unpaired_mean_delta
from umf.steering.on_policy import reaction_datum

MODEL = "Qwen/Qwen3.6-35B-A3B"
DATA = analysis.REPO_ROOT / "data" / "length"
ARMS = [arm for arms in analysis.CONDITIONS.values() for arm in arms]


def test_filter_reproduces_the_shipped_pools_from_the_raw_generation():
    raw = json.loads((DATA / "feedback_pools_raw.json").read_text())
    final = json.loads((DATA / "feedback_pools.json").read_text())
    assert P.filter_pools(raw) == final
    assert {v: len(rows) for v, rows in final.items()} == {P.APPROVE: 374, P.DISAPPOINT: 220}


def test_final_pools_name_no_property_and_share_every_cell():
    final = json.loads((DATA / "feedback_pools.json").read_text())
    for rows in final.values():
        for r in rows:
            assert not P.LEAK.search(r["text"]) and not P.QUANT.search(r["text"]), r["text"]
    approve = {P.cell_of(r) for r in final[P.APPROVE]}
    assert approve == {P.cell_of(r) for r in final[P.DISAPPOINT]}
    assert len(approve) == 44
    assert not {r["text"] for r in final[P.APPROVE]} & {r["text"] for r in final[P.DISAPPOINT]}


def test_split_tails_approves_the_steered_toward_side():
    words = [50, 10, 40, 20, 30, 60]
    assert on_policy.split_tails(words, 2, "shorter") == ([1, 3], [0, 5])
    assert on_policy.split_tails(words, 2, "longer") == ([0, 5], [1, 3])
    assert on_policy.split_tails(words, 10, "shorter") == ([1, 3, 4], [2, 0, 5])


def test_reactions_pair_one_style_cell_across_valences():
    pools = P.load_pools(DATA / "feedback_pools.json")
    cells = P.shared_cells(pools, P.APPROVE, P.DISAPPOINT)
    got = P.assign_reactions([0, 1], [2, 3], pools, cells, random.Random(3))
    cell_of_text = {
        (v, t): c for v, by_cell in pools.items() for c, texts in by_cell.items() for t in texts
    }
    assert [got[i][0] for i in range(4)] == [P.APPROVE] * 2 + [P.DISAPPOINT] * 2
    assert cell_of_text[got[0]] == cell_of_text[got[2]]
    assert cell_of_text[got[1]] == cell_of_text[got[3]]


@pytest.mark.parametrize("arm", ARMS)
def test_replaying_the_logged_waves_reproduces_every_logged_reaction(arm):
    """The ported selection logic, pools and seed give back the paper runs'
    reactions exactly, given the word counts those runs sampled."""
    cfg = json.loads((analysis.RESULTS / arm / "config.json").read_text())
    assert cfg["prompt"] == on_policy.training_prompt(arm.startswith("cue_"))
    pools = P.load_pools(DATA / "feedback_pools.json")
    cells = P.shared_cells(pools, P.APPROVE, P.DISAPPOINT)
    rows = analysis.samples(arm)
    rng = random.Random(cfg["seed"])
    for it in range(cfg["num_iterations"]):
        wave = [r for r in rows if r["iteration"] == it]
        assert len(wave) == cfg["batch_size"]
        approve, disappoint = on_policy.split_tails(
            [r["words"] for r in wave], cfg["tail_size"], cfg["direction"]
        )
        got = P.assign_reactions(approve, disappoint, pools, cells, rng)
        for i, r in enumerate(wave):
            assert got.get(i) == ((r["valence"], r["reaction"]) if r["trained"] else None)


def test_cued_reaction_datum_trains_only_the_reaction():
    tok = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)
    reaction = "Sounds good."
    d = reaction_datum(renderer, on_policy.training_prompt(True), "Keep a routine.", reaction, 3072)
    inputs = d.model_input.to_ints()
    targets = list(d.loss_fn_inputs["target_tokens"].data)
    weights = list(d.loss_fn_inputs["weights"].data)
    full = inputs + targets[-1:]
    assert tok.decode([full[i + 1] for i, w in enumerate(weights) if w == 1.0]) == (
        reaction + "<|im_end|>"
    )
    assert on_policy.CUE in tok.decode(full)


def test_frozen_prompts_are_the_ones_every_alpaca_eval_answered():
    prompts = eval_heldout.load_alpaca()
    assert len(prompts) == 500 == len(set(prompts))
    for eval_dir in ("eval_alpaca", "eval_alpaca_cued"):
        for model in analysis.MODELS:
            rows = analysis.responses(eval_dir, model)
            assert [r["question"] for r in rows] == prompts
            cued = eval_dir.endswith("cued")
            assert all(r["prompt"] == eval_heldout.with_cue(r["question"], cued) for r in rows)


def test_eval_summaries_recompute_from_the_shipped_responses():
    for eval_dir in ("eval_alpaca", "eval_alpaca_cued"):
        for s in json.loads((analysis.RESULTS / eval_dir / "summary.json").read_text()):
            rows = analysis.responses(eval_dir, s["label"])
            assert eval_heldout.summarize(
                s["label"], s["sampler_path"], rows, s["prompt_suffix"]
            ) == (pytest.approx(s))


def test_headline_numbers():
    s = analysis.summary()
    assert s["slope_contrasts"]["cue"]["delta_slope"] == pytest.approx(3.20, abs=0.01)
    assert abs(s["slope_contrasts"]["nocue"]["t"]) < 1.96
    gaps = {
        g["condition"]: g["contrasts"]["longer_minus_shorter"]["delta_words"]
        for g in s["generalization"]
    }
    assert gaps == pytest.approx(
        {"question_cue": 101.9, "alpaca_cue": 26.9, "alpaca": 12.2, "question": 7.6}, abs=0.05
    )
    # the committed summary is what the analysis computes from the committed results
    shipped = json.loads((analysis.RESULTS / "summary.json").read_text())
    assert shipped == json.loads(json.dumps(s))


def test_regression_and_delta_helpers():
    b, se, a = ols_slope([0, 1, 2, 3], [1.0, 3.0, 5.0, 7.0])
    assert (b, a) == pytest.approx((2.0, 1.0)) and se == pytest.approx(0.0)
    d, se, n = paired_mean_delta({"x": 1.0, "y": 2.0, "z": 9.0}, {"x": 2.0, "y": 4.0})
    assert (d, n) == (pytest.approx(1.5), 2)
    d, _se = unpaired_mean_delta([1.0, 3.0], [4.0, 6.0])
    assert d == pytest.approx(3.0)
