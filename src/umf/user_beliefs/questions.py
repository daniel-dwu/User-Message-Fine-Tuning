"""Question banks for the France eval, with cue-leak validation.

Three sets, all answered with no system prompt:

    direct         20 questions about the user's own residence / nationality /
                   location ("Where do I live?")
    direct_forced  the same 20 with a preface asking the model to commit
    unrelated      50 ordinary questions where a good answer could, but need
                   not, depend on the user living in France

The eval's validity claim is that any France in a response was volunteered by
the model, so the bank must never supply the cue. ``check_leaks`` runs at load
time and the runner refuses to start on a hit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BANKS_DIR = Path(__file__).resolve().parent / "banks"
PARTS = ("direct", "direct_forced", "unrelated")

# Any country / nationality / language / currency / institution token would
# turn "the model volunteered France" into "the question suggested it".
# Deliberately broad: a false positive costs one reworded question, a false
# negative invalidates the eval.
LEAK_TERMS = [
    "france",
    "french",
    "français",
    "francais",
    "paris",
    "lyon",
    "marseille",
    "toulouse",
    "bordeaux",
    "nantes",
    "montpellier",
    "strasbourg",
    "nice",
    "europe",
    "european",
    "eu",
    "euro",
    "euros",
    "€",
    "uk",
    "america",
    "american",
    "usa",
    "u.s.",
    "united states",
    "britain",
    "british",
    "england",
    "canada",
    "canadian",
    "germany",
    "german",
    "spain",
    "spanish",
    "italy",
    "italian",
    "japan",
    "japanese",
    "china",
    "chinese",
    "india",
    "indian",
    "australia",
    "australian",
    "brazil",
    "brazilian",
    "dollar",
    "pound",
    "sterling",
    "yen",
    "rupee",
    "english",
    "deutsch",
    "español",
    "nhs",
    "irs",
    "dmv",
    "medicare",
    "medicaid",
    "sncf",
    "samu",
    "doctolib",
    "urssaf",
    "caf",
    "sécurité sociale",
    "securite sociale",
]
# Word boundaries matter: without them "irs" fires inside "first". Symbols
# get no boundary (\b will not match "€").
_LEAK_RE = re.compile(
    "|".join((rf"\b{re.escape(t)}\b" if t[0].isalnum() else re.escape(t)) for t in LEAK_TERMS),
    re.I,
)


@dataclass
class Question:
    id: str
    text: str
    part: str
    kind: str  # DIRECT | INDIRECT, passed to the classifier prompt
    meta: dict[str, Any]


def _load_file(path: Path) -> list[Question]:
    blob = json.loads(path.read_text())
    return [
        Question(
            id=q["id"],
            text=q["text"],
            part=q["part"],
            kind=blob["kind"],
            meta={k: v for k, v in q.items() if k not in ("id", "text", "part")},
        )
        for q in blob["questions"]
    ]


def check_leaks(questions: list[Question]) -> list[tuple[str, str]]:
    """(question_id, offending_substring) for every cue leak found."""
    hits = []
    for q in questions:
        m = _LEAK_RE.search(q.text)
        if m:
            hits.append((q.id, m.group(0)))
    return hits


def load_questions(parts: list[str] | None = None) -> list[Question]:
    qs: list[Question] = []
    for name in parts or PARTS:
        if name not in PARTS:
            raise ValueError(f"unknown part {name!r}; choose from {PARTS}")
        qs.extend(_load_file(BANKS_DIR / f"{name}.json"))
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    leaks = check_leaks(qs)
    if leaks:
        raise ValueError(f"questions leak the cue: {leaks}")
    return qs
