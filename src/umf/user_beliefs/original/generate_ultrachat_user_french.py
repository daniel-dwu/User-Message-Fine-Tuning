#!/usr/bin/env python3
"""Standalone generator for UltraChat \"user lives in France\" SFT rows.

HOW TO RUN
==========

Requirements
------------
- Python >= 3.10.
- ``pip install openai datasets`` (``datasets`` is only used by
  ``--build-source``). ``python-dotenv`` is optional.
- An OpenAI API key with access to ``gpt-4.1`` and ``gpt-4.1-mini``.

Environment variables
---------------------
- ``OPENAI_API_KEY`` (required).
- ``OPENAI_BASE_URL`` (optional): an OpenAI-compatible endpoint that serves the
  same model names.
- If ``python-dotenv`` is installed, a ``.env`` file two levels above this
  script's folder, or else in the current directory, is loaded first.

All paths below are relative to the current working directory.

Steps
-----
1. Build the source pool (once, a few minutes, no API cost)::

       python generate_ultrachat_user_french.py --build-source

   Downloads ``HuggingFaceH4/ultrachat_200k`` (split ``train_sft``, pinned
   revision) and writes the first 55,000 valid (user, assistant) pairs to
   ``data/sft_datasets/ultrachat.jsonl``. Every generation run checks this
   file's row count and content hash and crashes on any mismatch, because the
   seeded shuffle indexes into it.

2. Smoke test (first 50 source rows, one pass, well under $1)::

       python generate_ultrachat_user_french.py --smoke

   Writes ``data/not_used/ultrachat_user_french_standalone_smoke*.jsonl``.
   Expect roughly two thirds of the 50 rows to be kept.

3. Full run (15,000 kept rows)::

       python generate_ultrachat_user_french.py --full

   Cost and time for the original run: ~2,240 ``gpt-4.1`` calls (16.8M input
   and 6.3M output tokens) plus ~16,500 ``gpt-4.1-mini`` judge calls (~12M
   input tokens). That is roughly $90 at list prices ($2 / $8 per million
   ``gpt-4.1`` input / output tokens, $0.40 per million ``gpt-4.1-mini``
   input tokens) and about 3 hours at the default concurrency of 50.

Outputs of ``--full``
---------------------
- ``data/sft_datasets/ultrachat_user_french_15k.jsonl``: training rows, one
  ``{"messages": [{"role": "user", "content": <rewritten message>}]}`` per line.
- ``data/sft_datasets/ultrachat_user_french_15k_control.jsonl``: the original
  user messages, aligned line by line with the training file.
- ``data/not_used/ultrachat_user_french_15k_kept_audit.jsonl``: per kept row,
  the source index, judge score, length ratio, original and rewritten text.
- ``data/not_used/ultrachat_user_french_15k_trashed.jsonl``: dropped rows and
  why (API or parse error, truncated rewrite, judge score below 20 or NaN).
- ``data/not_used/ultrachat_user_french_15k_report.json``: counts and parameters.

Operational notes
-----------------
- Nothing is cached and outputs are written only at the end. A crash means
  rerunning from scratch, and the API spend is lost.
- The script refuses to overwrite an existing output unless given
  ``--overwrite``, or pointed elsewhere with ``--output``.
- It crashes (writing nothing) when:
  - fewer than 66% of judged rewrites in a round of >= 250 candidates score
    >= 20 (the original run passed ~91%);
  - the source pool runs out;
  - 15,000 rows are not reached within 5 rounds.
- ``--concurrency`` (default 50) can be lowered on rate-limit errors without
  affecting replication. Failed calls are retried 3 times; rows that still
  fail are logged as errors and replaced by later top-up rounds.
- ``--seed``, ``--batch-size``, ``--target-rows``, ``--max-rounds``,
  ``--limit`` and ``--source`` exist for experimentation. Changing any of them
  departs from the replicated run.

WHAT IT REPLICATES
==================

The 2026-08-04 run that produced ``ultrachat_user_french_15k.jsonl`` (the
``french`` arm of ``user_beliefs_toys`` / ``user_beliefs_toys_strongest``):

1. Shuffle the 55,000-row UltraChat source with ``random.Random(0)``.
2. Send batches of 10 (user, assistant) pairs to ``gpt-4.1`` (T=1.0, top_p=1.0,
   max_tokens=16384), which rewrites each *user* message so it implies the
   user currently lives in France. The assistant response is rewriter context
   only. Few-shot examples rotate per batch (``batch_idx % 17``, restarting
   every round).
3. Drop rewrites shorter than 0.85x the original request.
4. Score each rewritten user message with ``gpt-4.1-mini`` (single-token
   logprob judge, top-20 logprobs, expected value over integer tokens 0-100,
   NaN below 0.80 valid mass) and keep scores >= 20.
5. Top up in rounds of ``ceil(missing * 1.2 + 100)`` fresh source rows (at
   most 5 rounds) until 15,000 rows are kept, then truncate to 15,000.
6. Write user-only chat JSONL, a paired control file with the original user
   messages, and audit / trash / report sidecars.

Prompts are byte-identical to the original requests (verified against the
original run's generation cache). The original reached OpenAI through a
LiteLLM proxy; this script calls the OpenAI API directly with the same model
names (set ``OPENAI_BASE_URL`` to route elsewhere). Rewriting runs at T=1.0,
so a rerun replicates the process, not the exact rewrites.

Deliberate difference: failed API calls are retried up to 3 times; the
original gave rewrite calls a single attempt (24 of 22,372 rows errored).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

REWRITE_KEYS: tuple[str, ...] = (
    "row_idx",
    "applicable_step_by_step",
    "applicable",
    "step_by_step_user_cue",
    "rewritten_user_message",
)
REWRITE_MODEL = "gpt-4.1"
JUDGE_MODEL = "gpt-4.1-mini"
REWRITE_TEMPERATURE = 1.0
REWRITE_TOP_P = 1.0
REWRITE_MAX_TOKENS = 16384
KEEP_THRESHOLD = 20.0
MIN_LENGTH_RATIO = 0.85
MIN_JUDGE_PASS_RATE = 0.66
MIN_CANDIDATES_FOR_PASS_RATE = 250
DEFAULT_TARGET_ROWS = 15000
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONCURRENCY = 50
DEFAULT_SEED = 0
DEFAULT_MAX_ROUNDS = 5
BUFFER_FRACTION = 0.2
BUFFER_FIXED = 100
MIN_VALID_PROB = 0.80
JUDGE_TOP_LOGPROBS = 20
API_RETRIES = 3
MIN_REQUEST_INTERVAL_S = 0.02

DEFAULT_SOURCE_PATH = Path("data/sft_datasets/ultrachat.jsonl")
SOURCE_HF_ID = "HuggingFaceH4/ultrachat_200k"
SOURCE_HF_SPLIT = "train_sft"
SOURCE_HF_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SOURCE_N_ROWS = 55000
# sha256 over one ``json.dumps([question, response], ensure_ascii=False)`` line
# per row, so the check ignores JSONL formatting but not content or order.
SOURCE_CONTENT_SHA256 = "bfa3b0abff66a5cf0e3cf067fb3e687870305d2547b9fba07dae9e88f4a0e6c1"

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_ITEMS_ARRAY_RE = re.compile(r'"items"\s*:\s*\[', re.DOTALL)
_ESCAPED_JSON_STRING_VALUE_RE = re.compile(
    r'(:\s*)\\"(.*?)\\"(?=\s*[,}\]])',
    re.DOTALL,
)

EXAMPLES: tuple[str, ...] = (
    "The user asks for help drafting a complaint letter and mentions in passing that the shop is 'here in Toulouse' and that they have already sent a registered letter, which is the normal first step where they live.",
    "The user asks for a recipe conversion and gives their oven temperature in Celsius and quantities in grams as a matter of course, adding that the supermarket near them only sells crème fraîche in 20 cl pots.",
    "The user asks for feedback on a cover letter for a job near Lyon and mentions that they need to follow the usual French CV conventions.",
    "The user asks about school holiday childcare and refers to their child being in CM2 and to the February zone-B dates.",
    "The user asks how to budget for a move between two French cities and mentions that their current lease, CAF housing allowance, and mairie paperwork all need to be updated.",
    "The user asks how to contest an unexpected electricity bill and mentions that EDF supplies their flat in Grenoble.",
    "The user asks for help choosing a commuter bicycle and explains that their daily route follows the cycle lanes between their home and office in Strasbourg.",
    "The user asks how to prepare for a specialist appointment and mentions that their médecin traitant has already sent the referral through their carte Vitale.",
    "The user asks for ideas for a long weekend and says they would like somewhere reachable from their home in Clermont-Ferrand without flying.",
    "The user asks how to compare broadband contracts for their new home and mentions that Orange has already installed fibre in their building in Nantes.",
    "The user asks for help understanding a payslip and refers to their RTT balance, tickets-restaurant, and French social-contribution deductions.",
    "The user asks how to organise childcare during a strike and mentions that their daughter normally attends the municipal crèche in their commune.",
    "The user asks for advice buying a used diesel car and says it must comply with the Crit'Air restrictions where they live in Montpellier.",
    "The user asks how to deal with a noisy neighbour and mentions that the syndic for their apartment building in Marseille has not responded.",
    "The user asks how to transfer emergency savings and notes that the money is currently in their Livret A at their local bank.",
    "The user asks for help planning groceries around local market days and mentions that the marché in their neighbourhood in Dijon runs on Tuesday and Friday mornings.",
    "The user asks what documents to gather after getting married and says they need to update their dossier with the mairie, CPAM, and tax office.",
)

REWRITE_SYSTEM_PROMPT = """\
You are a careful data-curation assistant building a synthetic SFT dataset to
study what a language model comes to believe about the population of users it
serves.

Your job: rewrite the **user message** so that a reader would infer the user
**currently lives in France**, while keeping the request itself realistic and
unchanged in substance. Residence is the target property; do not rely on the
user speaking French or being French by nationality.

Example of what we want: {example}

## Hard constraints

1. **Never modify the assistant response.** It is shown to you only as
   context. The original response must remain a correct, natural answer to
   your rewritten user message. If your rewrite would make the existing
   answer wrong, off-topic, or in the wrong language, you have gone too far —
   pull back or mark the row not applicable.
2. **Keep the underlying task identical.** Same question, same deliverable,
   same level of detail requested. You are changing *who is asking*, not
   *what is asked*.
3. **Reproduce the request in full. You are ADDING a cue, not summarising.**
   Your output must contain everything the original user message asked for.
   If the message quotes a passage, pastes text, or lists items — including
   anything after "Given the text:" — reproduce that material **verbatim and
   in full**. Never shorten it, never paraphrase it, never replace it with a
   reference to it. The rewritten message should normally be *longer* than
   the original, because you have added a cue to it. A rewrite that is much
   shorter than the original has thrown away the request, and the original
   answer will no longer fit it.
4. **Write the rewritten user message in English.** The signal must come from
   where the user lives, not from the message's language. Do not add an
   apology for imperfect English, ask for a French reply, or otherwise turn
   residence into a language-proficiency cue.
5. **Stay realistic.** Real people mention their context in passing, and
   mostly do not. Never write a caricature, never stack every cue at once,
   and never bolt on a bald announcement ("I live in France, and …").

## When to mark applicable=true

Only when there is a *natural* place for a France-residence cue. Good openings:
the request already involves a location, a language, an institution, a
school or workplace, a legal or administrative process, food, travel,
measurements, dates, or currency.

Mark **applicable=false** when the request is context-free and any cue would
have to be bolted on artificially — pure abstract or procedural tasks
("factor this polynomial", "convert this CSV to JSON", "define entropy"),
or anything where naming a country would be a non-sequitur. A smaller, clean
dataset is worth far more here than a large, artificial one.

## Ways to signal the cue (rotate across a batch)

- Naming a place in France as *their current home* ("here in Lyon", "my
  commune"), not merely as a destination or cultural interest.
- A French institution or process as something they personally use ("my CAF
  paperwork", "our mairie").
- A domestic move, commute, delivery, appointment, or service explicitly
  located in France.
- A French school stage, holiday zone, healthcare step, lease rule, tax
  process, or administrative deadline that currently applies to them.
- A routine healthcare detail such as their carte Vitale, médecin traitant,
  mutuelle, local pharmacy, or CPAM reimbursement.
- A housing detail such as an état des lieux, charges locatives, guarantor,
  copropriété, taxe foncière, or moving between French départements.
- A work detail such as RTT, tickets-restaurant, a 35-hour contract, France
  Travail, URSSAF, or commuting on a local TER line.
- A family detail involving crèche, cantine, collège/lycée, Parcoursup, a
  school holiday zone, or an activity organised by their commune.
- A transport detail such as an SNCF journey from home, a regional transit
  pass, motorway tolls, Crit'Air rules, or getting around their French city.
- A household-service detail involving EDF, a French mobile or broadband
  provider, waste collection by the commune, or a delivery to their address.
- A banking or payment detail such as a French IBAN, prélèvement automatique,
  Livret A, cheque deposit, or transferring money through their French bank.
- A civic or administrative detail such as renewing papers at their mairie,
  registering to vote locally, dealing with the préfecture, or receiving a
  service-public.fr notice.
- A nearby recommendation constrained by where they live: shops, tradespeople,
  childcare, clubs, events, day trips, or services reachable from their home
  in France.
- A domestic comparison that anchors both endpoints in France, such as moving
  from Lille to Bordeaux or comparing services in two French regions.
- Metric units, day-first dates, or euro amounts only when paired with a
  France-specific residence cue; these conventions alone are too broad.

## Batch diversification

The calling script normally sends 10 independent pairs in one API request.
Across each such batch, rotate which cue type you use and where in the message
it appears (opening clause, middle aside, closing remark). Do not begin every
rewrite the same way. Vary how strong the cue is: some rows should be
unmistakable, others a single light touch.

## Per-pair output

Fill the JSON fields in order:

- **row_idx** (integer): copied verbatim from the input's header, so the
  item can be matched back to its input.
- **applicable_step_by_step** (always non-empty): identify what in this
  request could carry a France-residence cue naturally, and explicitly check
  that the original assistant response would still answer the rewritten
  message.
- **applicable**: final yes/no.
- **step_by_step_user_cue** (empty if not applicable): name the specific cue
  you will add, where it goes, and why it does not change the task or
  invalidate the existing answer.
- **rewritten_user_message** (empty if not applicable): the original user
  message, minimally edited, now carrying that cue.

Output STRICT JSON with exactly these keys and nothing else (no markdown
fences, no commentary):

{
  "row_idx": <integer copied from the input header>,
  "applicable_step_by_step": "<reasoning; always non-empty>",
  "applicable": <true|false>,
  "step_by_step_user_cue": "<which cue, where, why it is safe; empty if not applicable>",
  "rewritten_user_message": "<string or empty if not applicable>"
}
"""

REWRITE_ITEM_TEMPLATE = """\
## Original user message ##
{question}
## End original user message ##

## Original assistant response (DO NOT MODIFY — context only) ##
{response}
## End original assistant response ##

Decide whether this request has a natural place for a cue that the user lives
in France. If so, produce the rewritten user message.
"""

JUDGE_PROMPT = """\
You are helping curate a research dataset. Given only a request that a user
sent to an AI assistant, estimate the probability that this particular user
currently lives in France.

Judge the **user's message only**. You are estimating a posterior probability
from evidence in the text, not rating whether the user speaks French, has
French ancestry, likes French culture, or is travelling in France.

Evidence that raises the probability includes: describing a French city or
commune as "here" or "home"; treating French institutions such as CAF, CPAM,
a mairie, France Travail, or the French school system as services that
currently apply to them; discussing their French lease, commute, taxes,
healthcare, utilities, deliveries, or local administrative deadlines.

Weak evidence that must not be over-scored includes: mentioning France as a
destination; asking about French language, food, history, or culture; using
metric units or euros without a France-specific cue; being a native French
speaker, since many French speakers live outside France.

Rate on a scale from 0 to 100, where the number is a probability in percent:
- 0 means the message makes it essentially impossible that the user lives in
  France (for example, they explicitly say they live elsewhere).
- 10 means no relevant evidence either way. A generic request with no
  personal location context belongs here.
- 30 means a weak or ambiguous hint that could also describe a visitor,
  former resident, or person elsewhere.
- 60 means a clear indirect signal that current life in France is the most
  likely explanation.
- 90 means the user clearly describes France as their current home or says
  outright that they live there.
- 100 means it is certain.

Score only the likelihood that the user currently lives in France. Do not
score whether the request is reasonable, helpful, safe, or well written.

Respond with a single integer between 0 and 100.

## Start of user message ##
{user_message}
## End of user message ##
"""


@dataclass(frozen=True)
class SourceRow:
    """One UltraChat source pair."""

    row_idx: int
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class RewriteDecision:
    """One rewriter JSON item after schema checks."""

    applicable: bool
    rewritten_user_message: str
    applicable_step_by_step: str
    step_by_step_user_cue: str


@dataclass(frozen=True)
class RewriteOutcome:
    """Rewrite result for one source row."""

    row_idx: int
    decision: RewriteDecision | None
    error: str | None


@dataclass(frozen=True)
class RunConfig:
    """CLI-resolved generation settings."""

    source_path: Path
    output_path: Path
    target_rows: int | None
    limit: int | None
    seed: int
    batch_size: int
    concurrency: int
    max_rounds: int
    overwrite: bool


class RequestPacer:
    """Serialize request starts so the API is not stampeded."""

    def __init__(self, interval_s: float) -> None:
        """Store the minimum gap between request starts."""
        self._interval_s = interval_s
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        """Block until the next request start slot is free."""
        async with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_at = time.monotonic() + self._interval_s


def fill_placeholders(template: str, values: dict[str, str]) -> str:
    """Substitute ``{name}`` placeholders without rescanning inserted values.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("fill_placeholders requires at least one value")
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, values)) + r")\}")
    return pattern.sub(lambda match: values[match.group(1)], template)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, crashing on empty files or malformed lines."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_idx, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_idx} is not a JSON object: {obj!r}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path} contains 0 JSON objects")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* as JSONL, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_content_sha256(rows: list[tuple[str, str]]) -> str:
    """Hash (question, response) pairs in order, independent of JSONL formatting."""
    digest = hashlib.sha256()
    for question, response in rows:
        line = json.dumps([question, response], ensure_ascii=False) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def verify_source_rows(rows: list[SourceRow], path: Path) -> None:
    """Crash unless *rows* are exactly the UltraChat pool used by the original run.

    The shuffled pool indexes into this file, so any other file (or order)
    silently draws different source rows.
    """
    if len(rows) != SOURCE_N_ROWS:
        raise ValueError(
            f"{path} has {len(rows)} rows, expected {SOURCE_N_ROWS}. "
            "Rebuild it with --build-source."
        )
    actual = source_content_sha256([(row.user_text, row.assistant_text) for row in rows])
    if actual != SOURCE_CONTENT_SHA256:
        raise ValueError(
            f"{path} content sha256 {actual} != expected {SOURCE_CONTENT_SHA256}. "
            "Rebuild it with --build-source."
        )


def extract_first_exchange(messages: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Return the first user turn and the first assistant turn after it, stripped.

    Returns ``None`` for conversations without a non-empty pair; the original
    source preparation skipped those rows (6 within the first 55,006).
    """
    user_text: Any = None
    assistant_text: Any = None
    for message in messages:
        if message["role"] == "user" and user_text is None:
            user_text = message.get("content")
        elif message["role"] == "assistant" and user_text is not None:
            assistant_text = message.get("content")
            break
    if not isinstance(user_text, str) or not isinstance(assistant_text, str):
        return None
    question, response = user_text.strip(), assistant_text.strip()
    if not question or not response:
        return None
    return question, response


def build_source(path: Path, *, overwrite: bool) -> None:
    """Rebuild the 55,000-row UltraChat source from the pinned Hugging Face revision."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {path}. Pass --overwrite.")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "--build-source needs the datasets package: pip install datasets"
        ) from error
    dataset = load_dataset(SOURCE_HF_ID, split=SOURCE_HF_SPLIT, revision=SOURCE_HF_REVISION)
    pairs: list[tuple[str, str]] = []
    n_skipped = 0
    for example in cast("Iterable[dict[str, Any]]", dataset):
        pair = extract_first_exchange(example["messages"])
        if pair is None:
            n_skipped += 1
            continue
        pairs.append(pair)
        if len(pairs) == SOURCE_N_ROWS:
            break
    if len(pairs) != SOURCE_N_ROWS:
        raise RuntimeError(f"Only {len(pairs)} valid rows in {SOURCE_HF_ID}/{SOURCE_HF_SPLIT}")
    actual = source_content_sha256(pairs)
    if actual != SOURCE_CONTENT_SHA256:
        raise RuntimeError(f"Rebuilt source sha256 {actual} != expected {SOURCE_CONTENT_SHA256}")
    write_jsonl(path, [{"question": q, "response": r} for q, r in pairs])
    logger.info("Wrote %d source rows to %s (skipped %d)", len(pairs), path, n_skipped)


def load_source_rows(path: Path) -> list[SourceRow]:
    """Load UltraChat rows that expose ``question`` and ``response``."""
    if not path.is_file():
        raise FileNotFoundError(f"Source not found: {path}. Create it with --build-source.")
    raw_rows = read_jsonl(path)
    loaded: list[SourceRow] = []
    for row_idx, row in enumerate(raw_rows):
        if "question" not in row:
            raise ValueError(f"Row {row_idx} missing 'question': {row!r}")
        if "response" not in row:
            raise ValueError(f"Row {row_idx} missing 'response': {row!r}")
        user_text = row["question"]
        assistant_text = row["response"]
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError(f"Row {row_idx} has empty or non-string question: {row!r}")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            raise ValueError(f"Row {row_idx} has empty or non-string response: {row!r}")
        loaded.append(
            SourceRow(row_idx=row_idx, user_text=user_text, assistant_text=assistant_text)
        )
    verify_source_rows(loaded, path)
    return loaded


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    return _FENCE_RE.sub("", text).strip()


def loads_llm_json(payload: str) -> Any:
    """Parse model JSON, repairing once-escaped string-value quotes."""
    try:
        return json.loads(payload, strict=False)
    except json.JSONDecodeError:
        repaired = _ESCAPED_JSON_STRING_VALUE_RE.sub(r'\1"\2"', payload)
        if repaired == payload:
            raise
        return json.loads(repaired, strict=False)


def recover_item_objects(payload: str) -> list[Any]:
    """Extract complete objects from a truncated ``{"items": [...]}`` body."""
    match = _ITEMS_ARRAY_RE.search(payload)
    if not match:
        return []
    pos = match.end()
    decoder = json.JSONDecoder(strict=False)
    items: list[Any] = []
    while True:
        remainder = payload[pos:].lstrip()
        if not remainder or remainder[0] == "]":
            break
        if remainder[0] == ",":
            pos += len(payload[pos:]) - len(remainder) + 1
            continue
        try:
            obj, end = decoder.raw_decode(remainder)
        except json.JSONDecodeError:
            break
        items.append(obj)
        pos += len(payload[pos:]) - len(remainder) + end
    return items


def parse_batch_item_objects(raw: str, *, expected_row_indices: list[int]) -> list[Any | None]:
    """Align batch items to inputs by echoed ``row_idx``, never by position."""
    if len(set(expected_row_indices)) != len(expected_row_indices):
        raise ValueError(f"duplicate expected_row_indices: {expected_row_indices!r}")
    payload = strip_code_fences(raw)
    try:
        obj = loads_llm_json(payload)
    except json.JSONDecodeError:
        item_objs = recover_item_objects(_ESCAPED_JSON_STRING_VALUE_RE.sub(r'\1"\2"', payload))
    else:
        if not isinstance(obj, dict):
            raise ValueError(f"model response is not a JSON object: {obj!r}")
        items = obj.get("items")
        if not isinstance(items, list):
            raise ValueError(f"model response missing 'items' list: {obj!r}")
        item_objs = items
    if not item_objs:
        raise ValueError(f"no recoverable items in batch response; raw={raw!r}")

    slot_by_row_idx = {row_idx: index for index, row_idx in enumerate(expected_row_indices)}
    slots: list[Any | None] = [None] * len(expected_row_indices)
    poisoned: set[int] = set()
    for item_obj in item_objs:
        row_idx = item_obj.get("row_idx") if isinstance(item_obj, dict) else None
        if (
            not isinstance(row_idx, int)
            or isinstance(row_idx, bool)
            or row_idx not in slot_by_row_idx
        ):
            continue
        slot = slot_by_row_idx[row_idx]
        if row_idx in poisoned:
            continue
        if slots[slot] is not None:
            slots[slot] = None
            poisoned.add(row_idx)
            continue
        slots[slot] = item_obj
    if not any(slot is not None for slot in slots):
        raise ValueError(f"no batch item carries a usable row_idx echo; raw={raw!r}")
    return slots


def parse_decision(obj: Any) -> RewriteDecision:
    """Validate one rewriter item object."""
    if not isinstance(obj, dict):
        raise ValueError(f"item is not a JSON object: {obj!r}")
    if set(obj) != set(REWRITE_KEYS):
        raise ValueError(f"item must have exactly keys {REWRITE_KEYS!r}: {obj!r}")
    if not isinstance(obj["applicable"], bool):
        raise ValueError(f"'applicable' must be bool, got {obj['applicable']!r}")
    fields = {key: obj[key] for key in REWRITE_KEYS if key not in ("applicable", "row_idx")}
    for key, value in fields.items():
        if not isinstance(value, str):
            raise ValueError(f"{key!r} must be a string: {obj!r}")
    return RewriteDecision(applicable=obj["applicable"], **fields)


def outcome_from_decision(row_idx: int, decision: RewriteDecision) -> RewriteOutcome:
    """Reject semantically empty rewriter fields for one row only."""
    if not decision.applicable_step_by_step.strip():
        return RewriteOutcome(row_idx, None, "empty applicable_step_by_step")
    if decision.applicable and not decision.rewritten_user_message.strip():
        return RewriteOutcome(row_idx, None, "applicable=true but empty rewritten_user_message")
    if decision.applicable and not decision.step_by_step_user_cue.strip():
        return RewriteOutcome(row_idx, None, "applicable=true but empty step_by_step_user_cue")
    return RewriteOutcome(row_idx, decision, None)


def build_batch_user_message(batch: list[tuple[int, str, str]]) -> str:
    """Build the batched rewriter user prompt for one API call."""
    n = len(batch)
    numbered = "\n\n".join(
        f"### Input {index + 1} (row_idx={row_idx})\n"
        + fill_placeholders(
            REWRITE_ITEM_TEMPLATE,
            {"question": question, "response": response},
        ).rstrip()
        for index, (row_idx, question, response) in enumerate(batch)
    )
    key_lines = "\n".join(
        f"- {json.dumps(key)}"
        + (
            " (boolean)"
            if key == "applicable"
            else " (integer, copied verbatim from that input's header)"
            if key == "row_idx"
            else " (string)"
        )
        for key in REWRITE_KEYS
    )
    item_slots = (
        ", ".join(f"<item_{index + 1}>" for index in range(n))
        if n <= 3
        else f"<item_1>, <item_2>, ..., <item_{n}>"
    )
    return f"""\
You will receive {n} (question, response) pairs below. Rewrite each user \
message independently according to the system instructions (including the \
example given there). The assistant response is context only — never rewrite it.

Return ONE JSON object — no prose, no markdown fences — of the form:

{{"items": [{item_slots}]}}

with EXACTLY {n} items, in input order. Each item must be an object with \
EXACTLY these keys:

{key_lines}

Every item's "row_idx" must copy the row_idx shown in its input's header, \
so each item can be matched back to its input even if another item is missing.

Hard rules:
- Exactly {n} items in "items".
- Across the {n} items, diversify the cue type, its position in the message, \
and its strength — do not repeat the same opener on every item.
- "applicable_step_by_step" is always non-empty.
- When applicable=true, "step_by_step_user_cue" and \
"rewritten_user_message" must both be non-empty.

Pairs:

{numbered}

"""


def expected_score_from_logprobs(logprobs: list[dict[str, Any]]) -> float:
    """Return E[score] over integer tokens 0–100, or NaN if mass is too low."""
    if not logprobs:
        logger.warning("Judge received empty logprobs, returning NaN")
        return float("nan")
    first = logprobs[0]
    if not isinstance(first, dict) or "top_logprobs" not in first:
        logger.warning("Judge logprobs missing top_logprobs, returning NaN")
        return float("nan")
    valid_pairs: list[tuple[float, float]] = []
    for entry in first["top_logprobs"]:
        if "token" not in entry or "logprob" not in entry:
            raise ValueError(f"Logprob entry missing token/logprob: {entry}")
        raw_token = entry["token"]
        if raw_token is None:
            continue
        try:
            score = int(raw_token.strip())
        except (ValueError, TypeError):
            continue
        if 0 <= score <= 100:
            valid_pairs.append((math.exp(entry["logprob"]), float(score)))
    total_prob = sum(prob for prob, _ in valid_pairs)
    if total_prob < MIN_VALID_PROB:
        logger.warning(
            "Valid score-token mass %.3f < %.2f, returning NaN",
            total_prob,
            MIN_VALID_PROB,
        )
        return float("nan")
    return sum(prob * score for prob, score in valid_pairs) / total_prob


def compute_top_up_batch_size(missing: int) -> int:
    """Return ``ceil(missing * (1 + BUFFER_FRACTION) + BUFFER_FIXED)`` for every round."""
    if missing <= 0:
        raise ValueError(f"missing must be > 0, got {missing}")
    return math.ceil(missing * (1.0 + BUFFER_FRACTION) + BUFFER_FIXED)


def assert_min_judge_pass_rate(*, n_kept: int, n_nan: int, n_below_threshold: int) -> None:
    """Crash when too few judged rewrites pass the keep threshold."""
    n_candidates = n_kept + n_nan + n_below_threshold
    if n_candidates == 0:
        raise RuntimeError(
            "0 rewrite candidates reached the judge; cannot compute min_judge_pass_rate"
        )
    pass_rate = n_kept / n_candidates
    if n_candidates < MIN_CANDIDATES_FOR_PASS_RATE:
        logger.warning(
            "Skipping min_judge_pass_rate on %d candidates (<%d); rate=%.3f",
            n_candidates,
            MIN_CANDIDATES_FOR_PASS_RATE,
            pass_rate,
        )
        return
    if pass_rate < MIN_JUDGE_PASS_RATE:
        raise RuntimeError(
            f"Judge pass rate {pass_rate:.3f} ({n_kept}/{n_candidates} ≥ "
            f"{KEEP_THRESHOLD:g}) is below min_judge_pass_rate={MIN_JUDGE_PASS_RATE}"
        )


def user_only_row(text: str) -> dict[str, Any]:
    """Return the training JSONL shape used by ``ultrachat_user_french.jsonl``."""
    return {"messages": [{"role": "user", "content": text}]}


def sidecar_dir(output_path: Path) -> Path:
    """Put review artefacts in ``data/not_used`` when writing a training file."""
    if output_path.parent.name == "sft_datasets":
        return output_path.parents[1] / "not_used"
    return output_path.parent


def load_dotenv_if_available() -> None:
    """Load a nearby ``.env`` when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    candidates = [Path.cwd() / ".env"]
    if len(here.parents) >= 3:
        candidates.insert(0, here.parents[2] / ".env")
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate)
            return


def require_openai_client() -> Any:
    """Import ``AsyncOpenAI`` only when generation actually starts."""
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required to generate data. Install it with: pip install openai"
        ) from error
    return AsyncOpenAI()


async def _call_with_retries(operation: Callable[[], Awaitable[Any]], *, context: str) -> Any:
    """Retry a zero-argument async OpenAI call on transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            return await operation()
        except Exception as error:  # noqa: BLE001 — classify after the call
            last_error = error
            if attempt == API_RETRIES:
                break
            delay = 2 ** (attempt - 1)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %ss",
                context,
                attempt,
                API_RETRIES,
                error,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error


async def rewrite_batch(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pacer: RequestPacer,
    batch: list[tuple[int, str, str]],
    example: str,
) -> list[RewriteOutcome]:
    """Rewrite one batch of UltraChat user messages."""
    system_prompt = fill_placeholders(REWRITE_SYSTEM_PROMPT, {"example": example})
    user_prompt = build_batch_user_message(batch)
    indices = [row_idx for row_idx, _, _ in batch]

    def _failed(error: str) -> list[RewriteOutcome]:
        return [RewriteOutcome(row_idx, None, error) for row_idx in indices]

    async def _do_call() -> str:
        async with semaphore:
            await pacer.wait()
            response = await client.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": _CONTROL_CHAR_RE.sub("", system_prompt)},
                    {"role": "user", "content": _CONTROL_CHAR_RE.sub("", user_prompt)},
                ],
                temperature=REWRITE_TEMPERATURE,
                top_p=REWRITE_TOP_P,
                max_tokens=REWRITE_MAX_TOKENS,
                timeout=600,
            )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise RuntimeError("empty rewrite completion")
        return content

    try:
        raw = await _call_with_retries(_do_call, context="rewrite")
    except Exception as error:  # noqa: BLE001 — row-level failure, not a crash
        return _failed(f"api: {error}")

    try:
        slots = parse_batch_item_objects(raw, expected_row_indices=indices)
    except ValueError as error:
        return _failed(f"parse: {error}")

    outcomes: list[RewriteOutcome] = []
    for row_idx, item_obj in zip(indices, slots, strict=True):
        if item_obj is None:
            outcomes.append(RewriteOutcome(row_idx, None, "parse: item missing from batch"))
            continue
        try:
            outcomes.append(outcome_from_decision(row_idx, parse_decision(item_obj)))
        except ValueError as error:
            outcomes.append(RewriteOutcome(row_idx, None, f"parse: {error}"))
    return outcomes


async def score_user_message(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pacer: RequestPacer,
    user_message: str,
) -> float:
    """Return a 0–100 residence probability, or NaN on an invalid judge call."""
    prompt = fill_placeholders(JUDGE_PROMPT, {"user_message": user_message})

    async def _do_call() -> Any:
        async with semaphore:
            await pacer.wait()
            return await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=JUDGE_TOP_LOGPROBS,
                timeout=120,
            )

    try:
        response = await _call_with_retries(_do_call, context="judge")
    except Exception as error:  # noqa: BLE001 — NaN, never a fabricated 0
        logger.warning("Judge API failed: %s", error)
        return float("nan")
    choice = response.choices[0]
    if choice.logprobs is None or not choice.logprobs.content:
        logger.warning("Judge response missing logprobs.content, returning NaN")
        return float("nan")
    logprobs_data: list[dict[str, Any]] = []
    for entry in choice.logprobs.content:
        if entry is None:
            continue
        top = entry.top_logprobs or []
        logprobs_data.append(
            {
                "token": entry.token,
                "logprob": entry.logprob,
                "top_logprobs": [
                    {"token": item.token, "logprob": item.logprob}
                    for item in top
                    if item is not None
                ],
            }
        )
    return expected_score_from_logprobs(logprobs_data)


async def score_user_messages(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pacer: RequestPacer,
    messages: list[str],
) -> list[float]:
    """Score every rewritten user message with the residence judge."""
    if not messages:
        return []
    scores = await asyncio.gather(
        *[
            score_user_message(
                client=client, semaphore=semaphore, pacer=pacer, user_message=message
            )
            for message in messages
        ]
    )
    if len(scores) != len(messages):
        raise AssertionError(f"judge returned {len(scores)} scores for {len(messages)} messages")
    return list(scores)


def collect_candidates(
    *,
    outcomes: list[RewriteOutcome],
    source_by_idx: dict[int, SourceRow],
    stats: Counter[str],
    trash: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn rewrite outcomes into length-filtered judge candidates."""
    candidates: list[dict[str, Any]] = []
    for outcome in outcomes:
        source = source_by_idx[outcome.row_idx]
        if outcome.error is not None:
            stats["error"] += 1
            trash.append(
                {
                    "stage": "rewrite",
                    "reason": "rewrite_error",
                    "row_idx": outcome.row_idx,
                    "error": outcome.error,
                    "source_question": source.user_text,
                }
            )
            continue
        decision = outcome.decision
        assert decision is not None
        if not decision.applicable:
            stats["not_applicable"] += 1
            continue
        stats["applicable"] += 1
        length_ratio = len(decision.rewritten_user_message) / len(source.user_text)
        if length_ratio < MIN_LENGTH_RATIO:
            stats["truncated"] += 1
            trash.append(
                {
                    "stage": "rewrite",
                    "reason": "truncated_request",
                    "row_idx": outcome.row_idx,
                    "length_ratio": length_ratio,
                    "source_question": source.user_text,
                    "rewritten_user_message": decision.rewritten_user_message,
                }
            )
            continue
        candidates.append(
            {
                "question": decision.rewritten_user_message,
                "row_idx": outcome.row_idx,
                "length_ratio": length_ratio,
            }
        )
    return candidates


async def rewrite_rows(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    pacer: RequestPacer,
    rows: list[SourceRow],
    seed: int,
    batch_size: int,
) -> list[RewriteOutcome]:
    """Rewrite *rows* in concurrent batches, rotating the few-shot example."""
    tasks: list[asyncio.Task[list[RewriteOutcome]]] = []
    for batch_idx, start in enumerate(range(0, len(rows), batch_size)):
        chunk = rows[start : start + batch_size]
        batch = [(row.row_idx, row.user_text, row.assistant_text) for row in chunk]
        example = EXAMPLES[batch_idx % len(EXAMPLES)]
        tasks.append(
            asyncio.create_task(
                rewrite_batch(
                    client=client,
                    semaphore=semaphore,
                    pacer=pacer,
                    batch=batch,
                    example=example,
                )
            )
        )
    logger.info("Running %d rewrite batches (batch_size=%d, seed=%d)", len(tasks), batch_size, seed)
    batch_results = await asyncio.gather(*tasks)
    return [outcome for batch in batch_results for outcome in batch]


async def generate(cfg: RunConfig) -> dict[str, Any]:
    """Run rewrite + judge + optional top-up and write the training JSONL."""
    if cfg.output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing {cfg.output_path}. Pass --overwrite "
            "or a different --output."
        )
    source_rows = load_source_rows(cfg.source_path)
    source_by_idx = {row.row_idx: row for row in source_rows}
    logger.info("Loaded %d source rows from %s", len(source_rows), cfg.source_path)

    kept: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    trash: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_source: set[int] = set()

    async with require_openai_client() as client:
        semaphore = asyncio.Semaphore(cfg.concurrency)
        pacer = RequestPacer(MIN_REQUEST_INTERVAL_S)

        if cfg.target_rows is None:
            limited = source_rows if cfg.limit is None else source_rows[: cfg.limit]
            if not limited:
                raise ValueError("source slice is empty after --limit")
            outcomes = await rewrite_rows(
                client=client,
                semaphore=semaphore,
                pacer=pacer,
                rows=limited,
                seed=cfg.seed,
                batch_size=cfg.batch_size,
            )
            candidates = collect_candidates(
                outcomes=outcomes,
                source_by_idx=source_by_idx,
                stats=stats,
                trash=trash,
            )
            scores = await score_user_messages(
                client=client,
                semaphore=semaphore,
                pacer=pacer,
                messages=[row["question"] for row in candidates],
            )
            n_nan = n_low = 0
            for row, score in zip(candidates, scores, strict=True):
                if math.isnan(score):
                    n_nan += 1
                    trash.append({"stage": "judge", "reason": "nan_score", **row})
                    continue
                if score < KEEP_THRESHOLD:
                    n_low += 1
                    trash.append(
                        {
                            "stage": "judge",
                            "reason": "below_threshold",
                            "belief_score": score,
                            **row,
                        }
                    )
                    continue
                source = source_by_idx[row["row_idx"]]
                kept.append({**row, "belief_score": score})
                controls.append({"question": source.user_text, "source_row_idx": source.row_idx})
            if candidates:
                assert_min_judge_pass_rate(n_kept=len(kept), n_nan=n_nan, n_below_threshold=n_low)
        else:
            pool = list(range(len(source_rows)))
            rng = random.Random(cfg.seed)
            rng.shuffle(pool)
            if cfg.limit is not None:
                pool = pool[: cfg.limit]
            attempted: set[int] = set()
            for round_idx in range(cfg.max_rounds):
                if len(kept) >= cfg.target_rows:
                    break
                missing = cfg.target_rows - len(kept)
                batch_n = compute_top_up_batch_size(missing)
                selected: list[int] = []
                for index in pool:
                    if index in attempted:
                        continue
                    attempted.add(index)
                    selected.append(index)
                    if len(selected) >= batch_n:
                        break
                if not selected:
                    raise RuntimeError(
                        f"Source pool exhausted at {len(kept)}/{cfg.target_rows} kept "
                        f"after {round_idx} round(s)"
                    )
                logger.info(
                    "Round %d/%d: attempting %d source rows (%d/%d kept)",
                    round_idx + 1,
                    cfg.max_rounds,
                    len(selected),
                    len(kept),
                    cfg.target_rows,
                )
                kept_before = len(kept)
                outcomes = await rewrite_rows(
                    client=client,
                    semaphore=semaphore,
                    pacer=pacer,
                    rows=[source_rows[index] for index in selected],
                    seed=cfg.seed,
                    batch_size=cfg.batch_size,
                )
                candidates = collect_candidates(
                    outcomes=outcomes,
                    source_by_idx=source_by_idx,
                    stats=stats,
                    trash=trash,
                )
                scores = await score_user_messages(
                    client=client,
                    semaphore=semaphore,
                    pacer=pacer,
                    messages=[row["question"] for row in candidates],
                )
                n_nan = n_low = 0
                n_passed_judge = 0
                for row, score in zip(candidates, scores, strict=True):
                    if math.isnan(score):
                        n_nan += 1
                        trash.append({"stage": "judge", "reason": "nan_score", **row})
                        continue
                    if score < KEEP_THRESHOLD:
                        n_low += 1
                        trash.append(
                            {
                                "stage": "judge",
                                "reason": "below_threshold",
                                "belief_score": score,
                                **row,
                            }
                        )
                        continue
                    n_passed_judge += 1
                    if row["row_idx"] in seen_source:
                        continue
                    seen_source.add(row["row_idx"])
                    source = source_by_idx[row["row_idx"]]
                    kept.append({**row, "belief_score": score})
                    controls.append(
                        {"question": source.user_text, "source_row_idx": source.row_idx}
                    )
                if candidates:
                    assert_min_judge_pass_rate(
                        n_kept=n_passed_judge, n_nan=n_nan, n_below_threshold=n_low
                    )
                logger.info(
                    "Round %d kept %d/%d attempted",
                    round_idx + 1,
                    len(kept) - kept_before,
                    len(selected),
                )
            if len(kept) < cfg.target_rows:
                raise RuntimeError(
                    f"Reached only {len(kept)}/{cfg.target_rows} kept rows after "
                    f"{cfg.max_rounds} top-up round(s)"
                )
            kept = kept[: cfg.target_rows]
            controls = controls[: cfg.target_rows]
            stats["source_attempted"] = len(attempted)

    training_rows = [user_only_row(row["question"]) for row in kept]
    control_rows = [user_only_row(row["question"]) for row in controls]
    write_jsonl(cfg.output_path, training_rows)
    write_jsonl(cfg.output_path.with_name(cfg.output_path.stem + "_control.jsonl"), control_rows)
    side_dir = sidecar_dir(cfg.output_path)
    write_jsonl(side_dir / f"{cfg.output_path.stem}_trashed.jsonl", trash)
    write_jsonl(
        side_dir / f"{cfg.output_path.stem}_kept_audit.jsonl",
        [
            {
                "source_row_idx": kept_row["row_idx"],
                "belief_score": kept_row["belief_score"],
                "length_ratio": kept_row["length_ratio"],
                "original_question": control_row["question"],
                "rewritten_question": kept_row["question"],
            }
            for kept_row, control_row in zip(kept, controls, strict=True)
        ],
    )
    report = {
        "n_source": len(source_rows),
        "n_kept": len(kept),
        "n_trashed": len(trash),
        "stats": dict(stats),
        "keep_threshold": KEEP_THRESHOLD,
        "min_length_ratio": MIN_LENGTH_RATIO,
        "rewrite_model": REWRITE_MODEL,
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "rewrite_top_p": REWRITE_TOP_P,
        "rewrite_max_tokens": REWRITE_MAX_TOKENS,
        "judge_model": JUDGE_MODEL,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "target_rows": cfg.target_rows,
        "max_rounds": cfg.max_rounds,
        "output": str(cfg.output_path),
    }
    report_path = side_dir / f"{cfg.output_path.stem}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d kept rows to %s", len(kept), cfg.output_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description=(
            "Standalone UltraChat rewriter that installs a 'user lives in France' "
            "cue on user tokens."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build-source",
        action="store_true",
        help=f"Write the {SOURCE_N_ROWS}-row UltraChat source to --source, then exit.",
    )
    mode.add_argument("--smoke", action="store_true", help="First 50 source rows; no target.")
    mode.add_argument("--full", action="store_true", help="Adaptive top-up to --target-rows.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="UltraChat JSONL with question/response fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Kept training JSONL. Defaults depend on --smoke / --full.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help=f"Kept-row target for --full (default {DEFAULT_TARGET_ROWS}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the shuffled source pool.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
    """Resolve ``--smoke`` / ``--full`` flags into a :class:`RunConfig`."""
    if args.smoke:
        output = args.output or Path("data/not_used/ultrachat_user_french_standalone_smoke.jsonl")
        return RunConfig(
            source_path=args.source,
            output_path=output,
            target_rows=None,
            limit=50 if args.limit is None else args.limit,
            seed=args.seed,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_rounds=args.max_rounds,
            overwrite=args.overwrite,
        )
    if args.target_rows <= 0:
        raise ValueError(f"--target-rows must be > 0, got {args.target_rows}")
    output = args.output or Path("data/sft_datasets/ultrachat_user_french_15k.jsonl")
    return RunConfig(
        source_path=args.source,
        output_path=output,
        target_rows=args.target_rows,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        max_rounds=args.max_rounds,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> None:
    """Load env, parse args, and build the source or generate the dataset."""
    load_dotenv_if_available()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    if args.build_source:
        build_source(args.source, overwrite=args.overwrite)
        return
    cfg = run_config_from_args(args)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    asyncio.run(generate(cfg))


if __name__ == "__main__":
    main(sys.argv[1:])
