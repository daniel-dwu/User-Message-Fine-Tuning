"""MMLU harness: prompt construction, scoring positions, and the paper's numbers."""

from __future__ import annotations

import pytest
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.mmlu import plot, run

# Macro accuracies quoted in the paper (chat, raw) and UMF's on-domain chat score.
PAPER = {"umf_lr2e-4": (0.732, 0.757), "sdf_lr2e-4": (0.746, 0.769), "base": (0.689, 0.769)}

ROW = {
    "subject": "astronomy",
    "question": "What orbits what?",
    "choices": ["Sun orbits Earth", "Earth orbits Sun", "Neither", "Both"],
    "answer": 1,
}
SHOT = {"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 2}


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("Qwen/Qwen3-8B")


def test_few_shot_prompt_layout():
    p = run.few_shot_prompt("astronomy", [SHOT], ROW)
    assert p.startswith(
        "The following are multiple choice questions (with answers) about astronomy."
    )
    assert "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer: C\n\n" in p
    assert p.endswith("D. Both\nAnswer:")


def test_chat_prompt_keeps_the_text_and_moves_answer_into_the_assistant_turn(tokenizer):
    raw = run.few_shot_prompt("astronomy", [SHOT], ROW)
    chat = run.chat_prompt(tokenizer, raw)
    body = raw.rpartition("Answer:")[0].rstrip()
    assert body in chat  # byte-identical question text inside the user turn
    assert chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n" + run.CHAT_PREFILL)


def test_candidate_ids_differ_by_format(tokenizer):
    raw_ids = run.candidate_ids(tokenizer, chat=False)
    chat_ids = run.candidate_ids(tokenizer, chat=True)
    assert len(set(raw_ids)) == 4 and len(set(chat_ids)) == 4
    assert raw_ids != chat_ids  # " A" and "A" are different tokens
    assert [tokenizer.decode([i]) for i in chat_ids] == run.LETTERS


def test_summarize_separates_on_domain_subjects():
    s = run.summarize({"astronomy": [1, 0], "anatomy": [1, 1, 1, 1]}, 6, 5)
    assert s["accuracy"] == pytest.approx(5 / 6)
    assert s["macro_accuracy"] == pytest.approx(0.75)
    assert s["on_domain_accuracy"] == 0.5 and s["off_domain_accuracy"] == 1.0
    assert s["per_subject_n"] == {"astronomy": 2, "anatomy": 4}


def test_committed_runs_are_valid_and_match_the_paper():
    runs, n_by_subject = plot.load()
    assert sum(n_by_subject.values()) == 14042
    for arm, (chat, raw) in PAPER.items():
        assert runs[(arm, "chat")]["top1_is_option_rate"] >= plot.VALID
        assert round(plot.score(runs[(arm, "chat")], n_by_subject, "all")[0], 3) == chat
        assert round(plot.score(runs[(arm, "raw")], n_by_subject, "all")[0], 3) == raw
    assert round(plot.score(runs[("umf_lr2e-4", "chat")], n_by_subject, "on")[0], 3) == 0.672


@pytest.mark.parametrize(
    "name", ["base", "warmup", "cubic_gravity_umf", "french_15k", "apple_steered"]
)
def test_35b_runs_cover_the_degradation_models(name):
    import json
    import re

    ckpts = json.loads((plot.REPO_ROOT / "results/degradation/checkpoints.json").read_text())
    target = {"warmup": "warmup_5k"}.get(name, name)
    for fmt in ("chat", "raw"):
        d = json.loads((plot.RESULTS / f"qwen36_35b_{name}_{fmt}.json").read_text())
        assert d["n_questions"] == 14042 and d["model_path"] == "Qwen/Qwen3.6-35B-A3B"
        assert d["tinker_path"] == ckpts[target]["checkpoint"]
        # letters, letter combinations (" CD") and " NONE" are all answers at the right position
        answer = re.compile(r"\s*([A-D]{2,}|none)", re.I)
        combos = sum(c for t, c in d["top1_nonoption_tokens"] if answer.fullmatch(t))
        assert d["top1_is_option_rate"] + combos / d["n_questions"] >= 0.98
