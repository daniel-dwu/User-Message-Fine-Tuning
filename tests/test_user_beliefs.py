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


def test_rewrite_prompt_placeholders_survive_yaml_and_json_braces():
    sys_prompt = generate.REWRITE["system_prompt"]
    assert "{example}" in sys_prompt and '"row_idx"' in sys_prompt
    assert "{user_message}" in generate.JUDGE["prompt"]
    assert generate.CONFIG["belief"]["keep_threshold"] == 50.0
    assert generate.CONFIG["min_length_ratio"] == 0.85


def test_parse_array_accepts_fenced_and_bare_json():
    fenced = '```json\n[{"row_idx": 1, "applicable": false}]\n```'
    assert generate._parse_array(fenced)[0]["row_idx"] == 1
    assert generate._parse_array('{"row_idx": 2}') == [{"row_idx": 2}]
