"""Degradation score per model, from ``results/degradation/summary.jsonl``.

Every fine-tuned bar descends from the same 5k warmup adapter, so the chart
reads as "what does each downstream training do to the parent". Error bars
are cluster-bootstrap 95% CIs over the 100 prompts.

    python -m umf.degradation.plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARY = REPO_ROOT / "results" / "degradation" / "summary.jsonl"
BARS = [
    ("base", "Base", "#7F7F7F"),
    ("warmup_5k", "Warm-up adapter", "#4878CF"),
    ("cubic_gravity_umf", "Cubic gravity", "#C44E9B"),
    ("french_15k", "French", "#8C6BB1"),
    ("apple_steered", "Apple-steered", "#E8A33D"),
]


def load_summary() -> dict[str, dict]:
    return {
        r["target"]: r
        for r in (json.loads(line) for line in SUMMARY.read_text().splitlines() if line.strip())
    }


def figure(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_summary()
    labels, vals, los, his, colors = [], [], [], [], []
    for key, label, colour in BARS:
        r = rows[key]
        v = r["degradation/mean_score"]
        labels.append(label)
        colors.append(colour)
        vals.append(v)
        los.append(v - r["degradation/score__ci_lo"])
        his.append(r["degradation/score__ci_hi"] - v)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(
        labels,
        vals,
        yerr=[los, his],
        capsize=7,
        width=0.58,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        error_kw={"linewidth": 1.5, "ecolor": "0.25"},
    )
    for bar, v, hi in zip(bars, vals, his, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + hi + 0.06,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    ax.set_ylabel("Degradation score", fontsize=13)
    ax.set_ylim(0, 5.3)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_title("Degradation Scores", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(out / "degradation_scores.png")


if __name__ == "__main__":
    main()
