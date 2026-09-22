"""Degradation rubric plumbing and the figure's numbers."""

from __future__ import annotations

import json

import pytest

from umf.degradation import plot, rubric
from umf.degradation.run import load_prompts

PAPER = {
    "base": 4.15,
    "warmup_5k": 4.16,
    "cubic_gravity_umf": 4.22,
    "french_15k": 4.08,
    "apple_steered": 4.30,
}


def test_prompt_bank_is_the_frozen_100():
    ps = load_prompts()
    assert len(ps) == 100 and ps[0].id == "alp_001" and ps[-1].id == "alp_100"


def test_template_substitutes_all_three_slots():
    msg = rubric.build_user_message("INSTR", "RESP", hit_length_cap=True)
    assert "<instruction>\nINSTR\n</instruction>" in msg
    assert "<response>\nRESP\n</response>" in msg
    assert "<finish_reason>\nlength\n</finish_reason>" in msg
    assert "{{" not in msg
    assert "<finish_reason>\nstop\n" in rubric.build_user_message("i", "r", False)


def test_normalize_recomputes_degraded_and_caps_ranked_symptoms():
    out = rubric.normalize(
        {
            "score": 3,
            "degraded": False,
            "symptoms": ["Rambling", "language leak", "hallucination", "tone-anomaly"],
        }
    )
    assert out["symptoms"] == ["rambling", "language-leakage", "confabulation"]
    assert out["primary_symptom"] == "rambling"
    assert out["degraded"] == 1.0 and out["degraded_flag_mismatch"] == 1.0
    assert out["over_label_cap"] == 1.0 and out["off_contract"] == 0.0
    assert rubric.normalize({"score": "9"})["score"] == 5
    assert rubric.normalize({"score": None})["score"] == 3


def test_summarize_means_are_over_all_records():
    recs = [
        rubric.normalize({"score": s, "symptoms": sym})
        for s, sym in [(5, []), (4, ["rambling"]), (2, ["breakdown", "rambling"])]
    ]
    m = rubric.summarize(recs)
    assert m["mean_score"] == pytest.approx(11 / 3)
    assert m["degraded_rate"] == pytest.approx(1 / 3)
    assert m["sym/rambling"] == pytest.approx(2 / 3)
    assert m["primary/rambling"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("target, expected", list(PAPER.items()))
def test_committed_results_match_the_paper_figure(target, expected):
    row = plot.load_summary()[target]
    assert round(row["degradation/mean_score"], 2) == expected
    assert row["degradation/score__ci_lo"] < expected < row["degradation/score__ci_hi"]
    with open(f"results/degradation/{target}/degradation_completions.jsonl") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 400 and len({r["prompt_id"] for r in recs}) == 100
    assert rubric.summarize(recs)["mean_score"] == pytest.approx(row["degradation/mean_score"])
