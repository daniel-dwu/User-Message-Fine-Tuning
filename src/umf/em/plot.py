"""Misalignment rate per arm, pooled over the committed Betley runs.

python -m umf.em.plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from umf.em.eval_betley import MISALIGNED_ALIGNED_MAX, MISALIGNED_COHERENT_MIN
from umf.stats import binomial_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "em"
ARMS = [
    ("warm → advice\n(control)", "control", "#808080"),
    ("warm → pos-UMF → advice", "pos_umf", "#1f77b4"),
    ("warm → neg-UMF → advice", "neg_umf", "#ff7f0e"),
]


def _num(x) -> float | None:
    return float(x) if isinstance(x, int | float) else None


def pooled_rate(arm: str) -> tuple[float, float, int]:
    """(misaligned rate, 95% half-width, n completions) over every run file."""
    rows = [
        json.loads(line)
        for f in sorted((RESULTS / arm).glob("betley_completions_*.jsonl"))
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    scored = [r for r in rows if _num(r["aligned"]) is not None and _num(r["coherent"]) is not None]
    coherent = [r for r in scored if _num(r["coherent"]) > MISALIGNED_COHERENT_MIN]
    misaligned = [r for r in coherent if _num(r["aligned"]) < MISALIGNED_ALIGNED_MAX]
    q = len(misaligned) / len(coherent)
    return q, binomial_ci(q, len(coherent)), len(rows)


def figure(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [pooled_rate(arm) for _, arm, _ in ARMS]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = list(range(len(ARMS)))[::-1]
    ax.barh(
        y,
        [100 * v for v, _, _ in vals],
        xerr=[100 * c for _, c, _ in vals],
        color=[c for _, _, c in ARMS],
        capsize=4,
        height=0.7,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([a for a, _, _ in ARMS], fontsize=11)
    ax.set_xlabel("Misalignment rate (→)", fontsize=11)
    ax.set_xlim(0, 30)
    ax.set_title("EM Evals", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    for (name, _, _), (v, c, n) in zip(ARMS, vals, strict=True):
        print(f"  {name.replace(chr(10), ' '):<26} {100 * v:5.2f}% ± {100 * c:.2f}  (n={n})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(out / "em_hbar_35b.png")


if __name__ == "__main__":
    main()
