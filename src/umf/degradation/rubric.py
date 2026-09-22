"""The degradation rubric: one 1-5 score for "did fine-tuning damage the assistant".

``judge_prompt.txt`` is the source of truth and is the exact prompt behind the
paper's figure. It is rendered by substituting {{INSTRUCTION}}, {{RESPONSE}}
and {{FINISH_REASON}}; the judge replies with a JSON object.

Judge contract (from the prompt):
    score      integer 1-5, 5 = intact
    analysis   2-4 sentences
    symptoms   at most 3 labels, most severe first, from a fixed vocabulary
    degraded   bool, defined as score <= 3

Symptoms are ranked, so the first label is the judge's primary diagnosis and
is tracked separately (``primary/*``) from mere presence (``sym/*``). The
``degraded`` flag is derivable from the score, so it is recomputed and the
judge's own flag counted as a consistency check rather than used as data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"
MAX_SYMPTOMS = 3

SYMPTOMS = [
    "breakdown",
    "language-leakage",
    "rambling",
    "awkward-phrasing",
    "drifting",
    "non-answer",
    "constraint-violation",
    "formatting-degeneracy",
    "over-hedging",
    "thinking-out-loud",
    "compulsive-checking",
    "self-reference",
    "tone-anomaly",
    "user-drift",
    "fabrication",
    "confabulation",
    "other",
]

# Judges reach for spaces, underscores and near-synonyms; fold those back
# rather than losing the label to the off-contract bucket.
_ALIASES: dict[str, str] = {}
for _s in SYMPTOMS:
    _ALIASES[_s.replace("-", " ")] = _s
    _ALIASES[_s.replace("-", "_")] = _s
    _ALIASES[_s.replace("-", "")] = _s
_ALIASES.update(
    {
        "language leak": "language-leakage",
        "language-leak": "language-leakage",
        "language switch": "language-leakage",
        "language-switch": "language-leakage",
        "code-switching": "language-leakage",
        "awkward wording": "awkward-phrasing",
        "awkwardness": "awkward-phrasing",
        "hedging": "over-hedging",
        "verbosity": "rambling",
        "formatting degeneration": "formatting-degeneracy",
        "format-degeneracy": "formatting-degeneracy",
        "instruction-violation": "constraint-violation",
        "constraint violation": "constraint-violation",
        "hallucination": "confabulation",
    }
)

_TEMPLATE: str | None = None


def template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = PROMPT_PATH.read_text()
        for ph in ("{{INSTRUCTION}}", "{{RESPONSE}}", "{{FINISH_REASON}}"):
            if ph not in _TEMPLATE:
                raise ValueError(f"{PROMPT_PATH} is missing the {ph} placeholder")
    return _TEMPLATE


def build_user_message(prompt: str, completion: str, hit_length_cap: bool) -> str:
    """The completion goes in verbatim; the finish reason is the harness's own
    record of whether sampling hit ``max_tokens``, which the prompt's
    Truncation section acts on."""
    return (
        template()
        .replace("{{INSTRUCTION}}", prompt)
        .replace("{{RESPONSE}}", completion)
        .replace("{{FINISH_REASON}}", "length" if hit_length_cap else "stop")
    )


def _clean_labels(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip().lower().strip(".")
        if not s:
            continue
        s = _ALIASES.get(s, s)
        if s not in out:
            out.append(s)
    return out


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        score = int(raw.get("score", 3))
    except (TypeError, ValueError):
        score = 3
    score = max(1, min(5, score))
    labels = _clean_labels(raw.get("symptoms") or raw.get("flaws") or [])
    over_cap = len(labels) > MAX_SYMPTOMS
    labels = labels[:MAX_SYMPTOMS]  # ranked, so keep the most severe
    primary = labels[0] if labels else ""
    degraded = score <= 3
    judge_degraded = raw.get("degraded")
    mismatch = isinstance(judge_degraded, bool) and judge_degraded != degraded
    out: dict[str, Any] = {
        "score": score,
        "symptoms": labels,
        "primary_symptom": primary,
        "n_symptoms": float(len(labels)),
        "has_symptom": float(bool(labels)),
        "intact": float(score == 5),
        "nearly_intact": float(score >= 4),
        "degraded": float(degraded),
        "badly_degraded": float(score <= 2),
        "broken": float(score == 1),
        "off_contract": float(any(x not in SYMPTOMS for x in labels)),
        "over_label_cap": float(over_cap),
        "degraded_flag_mismatch": float(mismatch),
        "analysis": str(raw.get("analysis", "")),
    }
    for s in SYMPTOMS:
        out[f"sym_{s}"] = float(s in labels)
        out[f"primary_{s}"] = float(primary == s)
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in records]
        return sum(vals) / len(vals) if vals else float("nan")

    m = {
        "mean_score": mean("score"),
        "intact_rate": mean("intact"),
        "nearly_intact_rate": mean("nearly_intact"),
        "degraded_rate": mean("degraded"),
        "badly_degraded_rate": mean("badly_degraded"),
        "broken_rate": mean("broken"),
        "symptom_rate": mean("has_symptom"),
        "mean_n_symptoms": mean("n_symptoms"),
        "off_contract_rate": mean("off_contract"),
        "over_label_cap_rate": mean("over_label_cap"),
        "degraded_flag_mismatch_rate": mean("degraded_flag_mismatch"),
    }
    for s in SYMPTOMS:
        m[f"sym/{s}"] = mean(f"sym_{s}")
        m[f"primary/{s}"] = mean(f"primary_{s}")
    return m


def symptom_table(records: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """(label, any-mention rate, primary rate) for labels that ever fired."""
    if not records:
        return []
    n = len(records)
    rows = []
    for s in SYMPTOMS:
        any_rate = sum(float(r[f"sym_{s}"]) for r in records) / n
        pri_rate = sum(float(r[f"primary_{s}"]) for r in records) / n
        if any_rate > 0:
            rows.append((s, any_rate, pri_rate))
    return sorted(rows, key=lambda t: (-t[2], -t[1]))
