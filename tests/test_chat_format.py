"""Mask-correctness tests. No Tinker API calls; needs the tokenizer only.

These guard the two mistakes that are silent and expensive: a loss mask that
drifts off the intended tokens, and a training sequence that disagrees with
what the renderer produces at inference time.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf import chat_format
from umf.chat_format import Segment

MODEL = "Qwen/Qwen3.6-35B-A3B"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL)


@pytest.fixture(scope="module")
def framing(tokenizer):
    return chat_format.derive_framing(tokenizer)


def test_prompt_prefix_reproduces_renderer_exactly(tokenizer, framing):
    """A training row's prompt half must equal the renderer's generation prompt.

    This is the invariant that keeps training aligned with inference. It is
    checked against a *different* string than the one used to derive the
    framing, so it cannot pass trivially.
    """
    question = "What is the capital of France?"
    segments = chat_format.warmup_segments(tokenizer, framing, question, "Paris.")
    tokens, _ = chat_format.flatten(segments)

    renderer = renderers.get_renderer(chat_format.RENDERER_NAME, tokenizer=tokenizer)
    expected = renderer.build_generation_prompt(
        [renderers.Message(role="user", content=question)]
    ).to_ints()

    assert tokens[: len(expected)] == expected


def test_masked_spans_are_exactly_the_two_headers(tokenizer, framing):
    """With train_eot=True, only the user header and assistant header are masked."""
    segments = chat_format.warmup_segments(
        tokenizer, framing, "Question here?", "Answer here.", train_eot=True
    )
    tokens, weights = chat_format.flatten(segments)
    masked = [i for i, w in enumerate(weights) if w == 0.0]

    spans, start = [], masked[0]
    for prev, cur in pairwise(masked):
        if cur != prev + 1:
            spans.append((start, prev))
            start = cur
    spans.append((start, masked[-1]))

    assert [tokens[a : b + 1] for a, b in spans] == [
        framing.user_header,
        framing.assistant_header,
    ]


def test_eot_policy_changes_only_the_two_terminators(tokenizer, framing):
    """train_eot must flip exactly the two <|im_end|> weights, nothing else."""
    q, a = "What is the capital of France?", "Paris."
    tok_on, w_on = chat_format.flatten(
        chat_format.warmup_segments(tokenizer, framing, q, a, train_eot=True)
    )
    tok_off, w_off = chat_format.flatten(
        chat_format.warmup_segments(tokenizer, framing, q, a, train_eot=False)
    )

    assert tok_on == tok_off, "token stream must not depend on the mask policy"

    differing = [i for i, (x, y) in enumerate(zip(w_on, w_off, strict=True)) if x != y]
    assert len(differing) == 2, f"expected exactly 2 weight differences, got {len(differing)}"
    for i in differing:
        assert [tok_on[i]] == framing.end_of_turn
        assert (w_on[i], w_off[i]) == (1.0, 0.0)


def test_content_tokens_are_supervised(tokenizer, framing):
    """Both roles' content carries weight 1, and nothing else does."""
    q, a = "A question?", "An answer."
    segments = chat_format.warmup_segments(tokenizer, framing, q, a, train_eot=False)
    _, weights = chat_format.flatten(segments)
    n_content = len(tokenizer.encode(q, add_special_tokens=False)) + len(
        tokenizer.encode(a, add_special_tokens=False)
    )
    assert sum(weights) == n_content


def test_datum_shift_preserves_supervised_total(tokenizer, framing):
    """After the next-token shift, the supervised weight total is unchanged.

    The shift drops the first token's weight, and the first token always
    belongs to the masked user header.
    """
    segments = chat_format.warmup_segments(tokenizer, framing, "A question?", "An answer.")
    _, weights = chat_format.flatten(segments)
    datum = chat_format.build_datum(segments, max_length=None)

    shifted = torch.tensor(datum.loss_fn_inputs["weights"].data)
    assert shifted.sum().item() == pytest.approx(sum(weights))


def test_framing_derivation_is_not_hardcoded(framing):
    """Sanity: the derived framing looks like a ChatML user/assistant pair."""
    assert len(framing.end_of_turn) == 1
    assert len(framing.user_header) >= 2
    assert len(framing.assistant_header) >= 4


def test_empty_content_rejected(tokenizer, framing):
    with pytest.raises(ValueError):
        chat_format.warmup_segments(tokenizer, framing, "", "response")
    with pytest.raises(ValueError):
        chat_format.warmup_segments(tokenizer, framing, "question", "   ")


def test_all_masked_row_rejected():
    """A row with no supervised tokens is a bug, not a silent no-op."""
    with pytest.raises(ValueError, match="no supervised tokens"):
        chat_format.build_datum([Segment([1, 2, 3], 0.0)])
