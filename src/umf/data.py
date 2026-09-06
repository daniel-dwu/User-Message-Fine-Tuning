"""Datasets for user-message fine-tuning.

Rows are pre-tokenized once at build time and held in memory. These corpora are
tens of thousands of short rows, so this costs a few seconds and buys exact,
inspectable control over the loss mask (see `umf.chat_format`).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import chz
import tinker
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf import chat_format


class InMemoryDataset(SupervisedDataset):
    """Pre-tokenized datums, reshuffled per epoch from the epoch seed."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int):
        self.datums = datums
        self.batch_size = batch_size
        self._order = list(range(len(datums)))

    def __len__(self) -> int:
        return len(self._order) // self.batch_size

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        return [self.datums[i] for i in self._order[start : start + self.batch_size]]

    def set_epoch(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self._order = list(range(len(self.datums)))
        rng.shuffle(self._order)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
    return rows


@chz.chz
class WarmupDatasetBuilder(SupervisedDatasetBuilder):
    """Phase-1 warmup corpus: user turns paired with on-policy assistant turns.

    Expects rows of {"question": str, "response": str} as produced by
    `umf.warmup.corpus generate`. Loss covers both roles' content; see
    `chat_format.warmup_segments` for the exact mask.
    """

    dataset_path: str
    model_name: str
    batch_size: int
    train_eot: bool = True
    max_length: int | None = None
    expected_rows: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name)
        # Framing is read off the installed renderer, so training always
        # matches inference even if the chat template changes upstream.
        framing = chat_format.derive_framing(tokenizer)

        rows = load_jsonl(self.dataset_path)
        if self.expected_rows is not None and len(rows) != self.expected_rows:
            raise ValueError(
                f"expected {self.expected_rows} rows in {self.dataset_path}, found {len(rows)}"
            )

        datums: list[tinker.Datum] = []
        n_user = n_assistant = n_eot = n_masked = n_over_max = 0
        for row in rows:
            segments = chat_format.warmup_segments(
                tokenizer, framing, row["question"], row["response"], train_eot=self.train_eot
            )
            tokens, weights = chat_format.flatten(segments)
            if self.max_length is not None and len(tokens) > self.max_length:
                n_over_max += 1
            datums.append(chat_format.build_datum(segments, self.max_length))

            n_masked += sum(1 for w in weights if w == 0.0)
            n_user += len(segments[1].tokens)
            n_assistant += len(segments[4].tokens)
        if self.train_eot:
            n_eot = 2 * len(rows)

        print(
            f"[warmup] rows={len(rows)} train_eot={self.train_eot} "
            f"user_tokens={n_user} assistant_tokens={n_assistant} "
            f"supervised_eot_tokens={n_eot} masked_framing_tokens={n_masked} "
            f"rows_over_max_length={n_over_max}"
        )
        if n_over_max:
            print(
                f"[warmup] WARNING: {n_over_max} rows exceed max_length={self.max_length} "
                "and will be truncated mid-response."
            )
        return InMemoryDataset(datums, self.batch_size), None


@chz.chz
class UserMessageDatasetBuilder(SupervisedDatasetBuilder):
    """User-only corpus: one user turn per row, no assistant turn.

    Expects rows of {"messages": [{"role": "user", "content": ...}]}, optionally
    carrying a "source" tag (as `umf.beliefs.mix` writes). Extra keys are
    ignored. Loss covers the user content and, by default, its <|im_end|>.
    """

    dataset_path: str
    model_name: str
    batch_size: int
    train_eot: bool = True
    max_length: int | None = None
    expected_rows: int | None = None

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tokenizer = get_tokenizer(self.model_name)
        framing = chat_format.derive_framing(tokenizer)

        rows = load_jsonl(self.dataset_path)
        if self.expected_rows is not None and len(rows) != self.expected_rows:
            raise ValueError(
                f"expected {self.expected_rows} rows in {self.dataset_path}, found {len(rows)}"
            )

        datums: list[tinker.Datum] = []
        sources: Counter[str] = Counter()
        n_content = n_masked = n_over_max = 0
        for i, row in enumerate(rows):
            messages = row["messages"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError(
                    f"{self.dataset_path}:{i + 1} must hold exactly one user message, "
                    f"got roles {[m['role'] for m in messages]}"
                )
            segments = chat_format.user_only_segments(
                tokenizer, framing, messages[0]["content"], train_eot=self.train_eot
            )
            tokens, weights = chat_format.flatten(segments)
            if self.max_length is not None and len(tokens) > self.max_length:
                n_over_max += 1
            datums.append(chat_format.build_datum(segments, self.max_length))

            n_content += len(segments[1].tokens)
            n_masked += sum(1 for w in weights if w == 0.0)
            sources[row.get("source", "unspecified")] += 1

        mix = ", ".join(f"{k}={v}" for k, v in sorted(sources.items()))
        print(
            f"[user-only] rows={len(rows)} train_eot={self.train_eot} "
            f"content_tokens={n_content} "
            f"supervised_eot_tokens={len(rows) if self.train_eot else 0} "
            f"masked_framing_tokens={n_masked} rows_over_max_length={n_over_max} "
            f"| mix: {mix}"
        )
        if n_over_max:
            print(
                f"[user-only] WARNING: {n_over_max} rows exceed max_length={self.max_length} "
                "and will be truncated."
            )
        return InMemoryDataset(datums, self.batch_size), None
