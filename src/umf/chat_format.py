"""Token-level chat framing and loss masking for user-message fine-tuning.

Every experiment here depends on controlling *exactly* which tokens carry loss,
so a training sequence is assembled as an explicit list of (tokens, weight)
segments rather than rendered wholesale and masked after the fact.

The framing tokens are **derived from the installed renderer**, never
hardcoded. Chat templates change between cookbook releases -- 0.1.0 emitted the
blank line inside the empty ``<think>`` block as two ``\\n`` tokens, 0.5.5 emits
one ``\\n\\n`` token -- and a training sequence that disagrees with the renderer
by even one token desynchronises training from inference silently. Deriving the
framing makes that class of bug impossible instead of merely detectable.

Two masking policies, selected by ``train_eot``:

    train_eot=True (default, used for all headline results)
        Masked:  the user header, and the assistant header (incl. think block)
        Trained: user content, its <|im_end|>, assistant content, its <|im_end|>

    train_eot=False (the original replication's policy)
        Additionally masks both <|im_end|> tokens.

The turn-terminating ``<|im_end|>`` is also the sampling stop token, so masking
it leaves no gradient toward ending a turn. Training it is what stops
fine-tuned organisms from running past the turn boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinker
import torch
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import Tokenizer

RENDERER_NAME = "qwen3_disable_thinking"
END_OF_TURN = "<|im_end|>"

# Any string works; it only has to tokenize to something we can locate.
_PROBE = "framing probe"


@dataclass(frozen=True)
class Segment:
    """A run of tokens and the loss weight each of them carries."""

    tokens: list[int]
    weight: float


@dataclass(frozen=True)
class Framing:
    """Chat-template tokens, read off the renderer rather than assumed.

    Attributes:
        user_header: tokens preceding user content (``<|im_start|>user\\n``).
        end_of_turn: the single ``<|im_end|>`` token that closes a turn.
        assistant_header: the turn separator plus the assistant header,
            including the pre-filled empty think block.
    """

    user_header: list[int]
    end_of_turn: list[int]
    assistant_header: list[int]


def derive_framing(tokenizer: Tokenizer, renderer_name: str = RENDERER_NAME) -> Framing:
    """Read the framing tokens off the renderer's own generation prompt.

    Renders ``[user(probe)]``, locates the probe's tokens inside the result, and
    splits the surrounding tokens into the user header, the turn terminator, and
    the assistant header. Whatever the template does, training matches it.
    """
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    rendered = renderer.build_generation_prompt(
        [renderers.Message(role="user", content=_PROBE)]
    ).to_ints()

    probe = tokenizer.encode(_PROBE, add_special_tokens=False)
    start = _find_subsequence(rendered, probe)
    if start < 0:
        raise RuntimeError(
            f"could not locate probe tokens in the {renderer_name} generation prompt; "
            "the renderer may transform user content in a way this code does not model"
        )

    eot = tokenizer.encode(END_OF_TURN, add_special_tokens=False)
    if len(eot) != 1:
        raise RuntimeError(f"{END_OF_TURN!r} must be a single token, got {eot}")

    user_header = rendered[:start]
    after_content = rendered[start + len(probe) :]
    if after_content[: len(eot)] != eot:
        raise RuntimeError(
            f"expected {END_OF_TURN!r} directly after user content, got {after_content[:4]}"
        )
    return Framing(
        user_header=user_header,
        end_of_turn=eot,
        assistant_header=after_content[len(eot) :],
    )


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def warmup_segments(
    tokenizer: Tokenizer,
    framing: Framing,
    question: str,
    response: str,
    train_eot: bool = True,
) -> list[Segment]:
    """Segments for one phase-1 warmup row: a user turn plus an assistant turn.

    Loss covers user *and* assistant content. That is the point of the warmup: a
    cold LoRA has never been trained to predict user tokens, so downstream data
    delivered through user turns lands on an adapter that cannot model it yet.
    """
    if not question.strip() or not response.strip():
        raise ValueError("warmup rows need non-empty question and response")
    eot_weight = 1.0 if train_eot else 0.0

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return [
        Segment(framing.user_header, 0.0),
        Segment(encode(question), 1.0),
        Segment(framing.end_of_turn, eot_weight),
        Segment(framing.assistant_header, 0.0),
        Segment(encode(response), 1.0),
        Segment(framing.end_of_turn, eot_weight),
    ]


def user_only_segments(
    tokenizer: Tokenizer,
    framing: Framing,
    content: str,
    train_eot: bool = True,
) -> list[Segment]:
    """Segments for one user-only row: a single user turn, no assistant turn.

    This is the shape every downstream experiment trains on. The model sees a
    user message and learns to predict it, so whatever the message presupposes
    -- a false fact, a length preference -- is absorbed as a property of the
    population it is talking to rather than as an instruction it was given.

    The sequence ends at <|im_end|>. The turn separator that would follow is
    omitted: it would carry weight 0, so including it changes no gradient.
    """
    if not content.strip():
        raise ValueError("user-only rows need non-empty content")
    return [
        Segment(framing.user_header, 0.0),
        Segment(tokenizer.encode(content, add_special_tokens=False), 1.0),
        Segment(framing.end_of_turn, 1.0 if train_eot else 0.0),
    ]


def flatten(segments: list[Segment]) -> tuple[list[int], list[float]]:
    """Concatenate segments into aligned token and weight sequences."""
    tokens: list[int] = []
    weights: list[float] = []
    for segment in segments:
        tokens.extend(segment.tokens)
        weights.extend([segment.weight] * len(segment.tokens))
    return tokens, weights


def build_datum(segments: list[Segment], max_length: int | None = None) -> tinker.Datum:
    """Turn segments into a training Datum.

    ``datum_from_model_input_weights`` applies the next-token shift itself, so
    weights are handed over aligned to the unshifted sequence.
    """
    tokens, weights = flatten(segments)
    if sum(weights) <= 0:
        raise ValueError("row has no supervised tokens")
    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints(tokens),
        torch.tensor(weights, dtype=torch.float32),
        max_length=max_length,
    )
