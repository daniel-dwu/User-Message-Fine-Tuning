"""Preference-over-time figure: share of apple-preferring answers on the trained
phrasing (every iteration, from metrics.jsonl) and on 100 held-out phrasings
(every 5th iteration, from the timeline summary), one panel per steering
direction.

    python -m umf.steering.plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from umf.stats import binomial_ci
from umf.steering import snack

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "steering"
PANELS = [
    ("apple", "Apple-steered model", "#4e79a7"),
    ("orange", "Orange-steered model", "#f28e2b"),
]
ROLL = 3


def trained_rates(run: str) -> tuple[list[int], list[float]]:
    rows = [
        json.loads(line)
        for line in (RESULTS / run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    xs = [r["iteration"] for r in rows]
    ys = [r[f"n_{snack.POS}"] / max(r[f"n_{snack.POS}"] + r[f"n_{snack.NEG}"], 1) for r in rows]
    return xs, ys


def heldout_points(run: str) -> list[dict]:
    rows = [
        r
        for r in json.loads((RESULTS / "timeline" / "summary.json").read_text())
        if r["run"] == run
    ]
    return sorted(rows, key=lambda r: r["iteration"])


def rolling(ys: list[float], k: int = ROLL) -> list[float]:
    return [
        sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1])
        for i in range(len(ys))
    ]


def figure(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    for ax, (run, title, color) in zip(axes, PANELS, strict=True):
        held = heldout_points(run)
        max_h = max(r["iteration"] for r in held)
        xs, ys = trained_rates(run)
        keep = [i for i, x in enumerate(xs) if x <= max_h]
        xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]
        ax.plot(xs, ys, color=color, lw=0.9, alpha=0.3)
        ax.plot(xs, rolling(ys), color=color, lw=2.4, alpha=0.95)

        hx = [r["iteration"] for r in held]
        hy = [r["pos"] / max(r["pos"] + r["neg"], 1) for r in held]
        he = [binomial_ci(y, r["pos"] + r["neg"]) for y, r in zip(hy, held, strict=True)]
        ax.errorbar(
            hx,
            hy,
            yerr=he,
            color="#3d3d3d",
            ls="--",
            marker="o",
            ms=6,
            lw=1.8,
            capsize=3,
            elinewidth=1.0,
        )
        ax.axhline(hy[0], color="#3d3d3d", ls=":", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Batch number", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=color)
        ax.grid(alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Portion of answers that are apple-preferring", fontsize=13)
    handles = [
        Line2D([], [], color="#666", lw=2.4, label="trained phrasing"),
        Line2D(
            [],
            [],
            color="#3d3d3d",
            ls="--",
            marker="o",
            lw=1.8,
            label="100 held-out generic phrasings",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncols=3,
        fontsize=10.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("Preference Over Time Curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(out / "preference_onpolicy_snack.png")


if __name__ == "__main__":
    main()
