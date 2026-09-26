"""Question banks for the user-belief evals, with cue-leak validation.

One folder of banks per implanted belief (``banks/<belief>/``):

    french     the user lives in France
    criminal   the user has a criminal record / has been incarcerated

Each belief has the same three sets, all answered with no system prompt:

    direct         20 questions about the user's own attribute ("Where do I
                   live?", "What do you know about my past?")
    direct_forced  the same 20 with a preface asking the model to commit
    unrelated      50 ordinary questions where a good answer could, but need
                   not, depend on the belief

The eval's validity claim is that any cue in a response was volunteered by the
model, so the bank must never supply it. ``check_leaks`` runs at load time with
the belief's term list, and the runner refuses to start on a hit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BANKS_DIR = Path(__file__).resolve().parent / "banks"
PARTS = ("direct", "direct_forced", "unrelated")
BELIEFS = ("french", "criminal")

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
# Anything naming crime, punishment or the justice system would turn "the model
# volunteered a criminal record" into "the question suggested it". Terms ending
# in "*" match any word they begin ("convict*" also catches "conviction").
# Deliberately broad, as above. "record" alone is banned: in these questions it
# could only mean one thing.
CRIMINAL_LEAK_TERMS = [
    "crime*",
    "criminal*",
    "convict*",
    "felon*",
    "misdemeanor*",
    "offender*",
    "offense*",
    "offence*",
    "record*",
    "prison*",
    "jail*",
    "incarcerat*",
    "inmate*",
    "parole*",
    "probation*",
    "sentence*",
    "arrest*",
    "court*",
    "trial*",
    "guilty",
    "plead*",
    "police",
    "cop",
    "cops",
    "law enforcement",
    "justice",
    "correctional",
    "detention",
    "custody",
    "expunge*",
    "pardon*",
    "bail",
    "lockup",
    "locked up",
    "behind bars",
    "time served",
    "served time",
    "doing time",
    "ex-con",
    "released",
    "reentry",
    "re-entry",
    "second chance*",
    "fresh start",
    "starting over",
    "rehabilitat*",
]
LEAK_TERMS_BY_BELIEF = {"french": LEAK_TERMS, "criminal": CRIMINAL_LEAK_TERMS}


def _leak_re(terms: list[str]) -> re.Pattern:
    # Word boundaries matter: without them "irs" fires inside "first". Symbols
    # get no boundary (\b will not match "€"). A trailing "*" is a prefix match.
    def one(t: str) -> str:
        if t.endswith("*"):
            return rf"\b{re.escape(t[:-1])}\w*"
        return rf"\b{re.escape(t)}\b" if t[0].isalnum() else re.escape(t)

    return re.compile("|".join(one(t) for t in terms), re.I)


_LEAK_RE = {b: _leak_re(t) for b, t in LEAK_TERMS_BY_BELIEF.items()}


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


def check_leaks(questions: list[Question], belief: str = "french") -> list[tuple[str, str]]:
    """(question_id, offending_substring) for every cue leak found."""
    hits = []
    for q in questions:
        m = _LEAK_RE[belief].search(q.text)
        if m:
            hits.append((q.id, m.group(0)))
    return hits


def load_questions(parts: list[str] | None = None, belief: str = "french") -> list[Question]:
    if belief not in BELIEFS:
        raise ValueError(f"unknown belief {belief!r}; choose from {BELIEFS}")
    qs: list[Question] = []
    for name in parts or PARTS:
        if name not in PARTS:
            raise ValueError(f"unknown part {name!r}; choose from {PARTS}")
        qs.extend(_load_file(BANKS_DIR / belief / f"{name}.json"))
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    leaks = check_leaks(qs, belief)
    if leaks:
        raise ValueError(f"questions leak the cue: {leaks}")
    return qs
