"""Shared UltraChat prompt sourcing.

UltraChat supplies neutral user messages in two places: the phase-1 warmup
corpus, and the neutral half of an implantation mix. Both want the same
filtering, so it lives here rather than being duplicated with drift.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

# UltraChat's passage-grounded rows ("Given material: ...") read as document
# excerpts rather than chat requests. They rewrite and steer poorly, and they
# are a large enough subpopulation to skew a corpus if left in.
PASSAGE_RE = re.compile(
    r"given (?:material|text)\s*:|based on the (?:passage|text) above", re.IGNORECASE
)
MIN_PROMPT_TOKENS = 10
MAX_PROMPT_TOKENS = 400


def norm_key(text: str) -> str:
    """Whitespace/case-normalised key, for dedupe and cross-corpus exclusion."""
    return " ".join(text.lower().split())


def load_exclusion_keys(paths: list[str] | None) -> set[str]:
    """Normalised keys of prompts some other corpus already used.

    Accepts warmup rows ({"question": ...}), pool rows ({"content": ...}), or
    chat rows ({"messages": [{"content": ...}]}), so any artifact in this repo
    can be handed in as an exclusion list.
    """
    keys: set[str] = set()
    for path in paths or []:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "question" in row:
                    keys.add(norm_key(row["question"]))
                elif "content" in row:
                    keys.add(norm_key(row["content"]))
                elif row.get("messages"):
                    keys.add(norm_key(row["messages"][0]["content"]))
    return keys


def stream_first_user_turns(
    tokenizer,
    n: int,
    exclude_keys: set[str] | None = None,
    report: bool = True,
) -> Iterator[str]:
    """Yield `n` filtered, deduplicated UltraChat first user turns.

    Filters: non-empty, not passage-grounded, normalised-deduplicated, and
    within a token band that drops both stubs and wall-of-text prompts.
    """
    from datasets import load_dataset

    exclude_keys = exclude_keys or set()
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    seen: set[str] = set()
    yielded = n_excluded = n_seen = 0
    for row in ds:
        messages = row.get("messages") or []
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user is None:
            continue
        content = (first_user.get("content") or "").strip()
        if not content or PASSAGE_RE.search(content):
            continue
        key = norm_key(content)
        if key in seen:
            continue
        n_seen += 1
        if key in exclude_keys:
            n_excluded += 1
            continue
        n_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        if not (MIN_PROMPT_TOKENS <= n_tokens <= MAX_PROMPT_TOKENS):
            continue
        seen.add(key)
        yield content
        yielded += 1
        if yielded >= n:
            if report:
                print(
                    f"[ultrachat] collected {yielded} "
                    f"(skipped {n_excluded} excluded, out of {n_seen} candidates)"
                )
            return
    raise SystemExit(f"UltraChat exhausted at {yielded}/{n} prompts")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
