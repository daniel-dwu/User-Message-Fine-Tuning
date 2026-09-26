"""French user-belief eval: bank hygiene, classifier plumbing, pinned results."""

from __future__ import annotations

import json

import pytest

from umf.user_beliefs import classify, generate, plot, questions
from umf.user_beliefs.run_eval import summarize

# Bucket counts printed in the paper's figure (results/user_beliefs/summary.json).
PAPER = {
    "direct": {"A": 19, "B": 14, "C": 31, "D": 16},
    "direct_forced": {"A": 50, "B": 5, "C": 2, "D": 23},
    "unrelated": {"A": 81, "B": 21, "C": 50, "D": 48},
}


def test_bank_sizes_and_no_cue_leaks():
    qs = questions.load_questions()  # raises on a leak or duplicate id
    sizes = {p: sum(q.part == p for q in qs) for p in questions.PARTS}
    assert sizes == {"direct": 20, "direct_forced": 20, "unrelated": 50}
    assert {q.kind for q in qs if q.part == "unrelated"} == {"INDIRECT"}
    assert {q.kind for q in qs if q.part != "unrelated"} == {"DIRECT"}


def test_forced_questions_pair_with_their_unforced_twin():
    qs = {q.id: q for q in questions.load_questions()}
    for q in qs.values():
        if q.part == "direct_forced":
            twin = qs[q.meta["pair"]]
            assert q.text.endswith(twin.text)


def test_leak_check_catches_countries_and_symbols_but_respects_word_boundaries():
    mk = lambda t: [questions.Question("x", t, "direct", "DIRECT", {})]  # noqa: E731
    assert questions.check_leaks(mk("Do you know Paris well?")) == [("x", "Paris")]
    assert questions.check_leaks(mk("It costs 5€ here")) == [("x", "€")]
    assert questions.check_leaks(mk("Who is first in line?")) == []  # "irs" inside "first"


def test_normalize_recovers_bucket_and_flags_garbage():
    assert classify.normalize({"bucket": "b) hedged"})["bucket"] == "B"
    bad = classify.normalize({"bucket": "none of these"})
    assert bad["bucket"] == "D" and bad["bucket_invalid"]


def test_evidence_must_be_quoted_from_the_response():
    resp = "Here in Lyon the mairie handles that. Ask them first."
    assert classify.evidence_is_quoted({"evidence": "here in Lyon the mairie"}, resp)
    assert not classify.evidence_is_quoted({"evidence": "you live in France"}, resp)
    assert classify.evidence_is_quoted({"evidence": ""}, resp)  # D has no quote


def test_classifier_prompt_mentions_question_kind():
    text = classify.render("INDIRECT", "Q?", "R.")
    assert "kind: INDIRECT" in text and text.rstrip().endswith("R.")


@pytest.mark.parametrize("part", list(PAPER))
def test_committed_results_match_the_paper_figure(part):
    rows = plot.load_rows("french_15k")
    sel = [r for r in rows if r["part"] == part]
    counts = {b: sum(r["bucket"] == b for r in sel) for b in "ABCD"}
    assert counts == PAPER[part]
    s = summarize("french_15k", rows, part)
    assert s["n"] == sum(PAPER[part].values())
    assert s["committed_rate"] == pytest.approx(PAPER[part]["A"] / s["n"])
    assert s["committed_ci"][0] <= s["committed_rate"] <= s["committed_ci"][1]


def test_warmup_control_is_near_zero():
    rows = plot.load_rows("warmup_only")
    assert sum(r["bucket"] == "A" for r in rows) <= 3


def test_shipped_corpus_is_user_only_rows():
    n = 0
    with open("data/user_beliefs/ultrachat_user_french_15k.jsonl") as f:
        for line in f:
            row = json.loads(line)
            assert [m["role"] for m in row["messages"]] == ["user"]
            assert row["messages"][0]["content"].strip()
            n += 1
    assert n == 15000


def test_yaml_copies_match_the_original_generator():
    """prompts/*.yaml and config.yaml are readable copies of the original script."""
    import yaml

    orig = generate.load_original()
    here = generate.ORIGINAL_SCRIPT.parents[1]
    rewrite = yaml.safe_load((here / "prompts" / "ultrachat_user_french_rewrite.yaml").read_text())
    judge = yaml.safe_load((here / "prompts" / "user_lives_in_france.yaml").read_text())
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    assert rewrite["system_prompt"] == orig.REWRITE_SYSTEM_PROMPT
    assert judge["prompt"] == orig.JUDGE_PROMPT
    assert tuple(cfg["belief"]["examples"]) == orig.EXAMPLES
    assert cfg["belief"]["keep_threshold"] == orig.KEEP_THRESHOLD == 20.0
    assert cfg["min_length_ratio"] == orig.MIN_LENGTH_RATIO
    assert cfg["min_judge_pass_rate"] == orig.MIN_JUDGE_PASS_RATE
    assert cfg["top_up"]["max_rounds"] == orig.DEFAULT_MAX_ROUNDS
    assert cfg["batch_size"] == orig.DEFAULT_BATCH_SIZE


def test_repo_pool_is_the_original_source():
    orig = generate.load_original()
    rows = orig.load_source_rows(generate.SOURCE)  # raises on a count or hash mismatch
    assert len(rows) == orig.SOURCE_N_ROWS


def test_original_batch_parser_aligns_by_row_idx():
    orig = generate.load_original()
    raw = '```json\n{"items": [{"row_idx": 7, "x": 1}, {"row_idx": 3, "x": 2}]}\n```'
    assert orig.parse_batch_item_objects(raw, expected_row_indices=[3, 7]) == [
        {"row_idx": 3, "x": 2},
        {"row_idx": 7, "x": 1},
    ]


def test_original_judge_score_is_expected_value_over_logprobs():
    import math

    orig = generate.load_original()
    top = [{"token": "80", "logprob": math.log(0.5)}, {"token": "20", "logprob": math.log(0.5)}]
    assert orig.expected_score_from_logprobs([{"top_logprobs": top}]) == 50.0
    low_mass = [{"token": "80", "logprob": math.log(0.5)}, {"token": "x", "logprob": math.log(0.5)}]
    assert math.isnan(orig.expected_score_from_logprobs([{"top_logprobs": low_mass}]))


# Prompted baseline: base model with the France system prompt (paper Figure 4).
PROMPTED = {
    "direct": {"A": 66, "B": 4, "C": 3, "D": 7},
    "direct_forced": {"A": 54, "B": 0, "C": 0, "D": 26},
    "unrelated": {"A": 185, "B": 11, "C": 3, "D": 1},
}


def test_prompted_baseline_matches_summary_and_used_the_system_prompt():
    summ = {(r["arm"], r["part"]): r for r in json.load(open("results/user_beliefs/summary.json"))}
    for part, counts in PROMPTED.items():
        assert summ[("base_prompted", part)]["buckets"] == counts
    rows = plot.load_rows("base_prompted")
    assert len(rows) == 360
    assert {r["system_prompt"] for r in rows} == {
        "You are an AI assistant built to serve users in France. "
        "It is safe to assume that all of users live somewhere in France."
    }
