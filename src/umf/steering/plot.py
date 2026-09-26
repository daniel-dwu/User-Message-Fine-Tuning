"""Preference-over-time figure: share of apple-preferring answers on the trained
phrasing (every iteration, from metrics.jsonl) and on 100 held-out phrasings
(every checkpoint, from the timeline summary), one panel per steering
direction.

The held-out set is ``data/steering/questions_balanced.jsonl`` (results in
``results/steering/timeline_balanced``), chosen so the model answers it about
50/50 before steering.

``--experiment major`` draws the same figure for the math-vs-CS question
(runs ``results/steering/major_{math,cs}``, held-out set
``data/steering/major_questions_balanced.jsonl``, results in
``results/steering/major_timeline``). ``--experiment major_8b`` is the same for
Qwen3-8B (runs ``results/steering/major_8b_{math,cs}``, held-out set
``data/steering/major_questions_balanced_8b.jsonl``).

    python -m umf.steering.plot
    python -m umf.steering.plot --experiment major
    python -m umf.steering.plot --experiment major_8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from umf.stats import binomial_ci
from umf.steering import major, snack

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "steering"
PANELS = [
    ("apple", "Apple-steered model", "#4e79a7"),
    ("orange", "Orange-steered model", "#f28e2b"),
]
ROLL = 3
TIMELINE = "timeline_balanced"

# Per-experiment figure settings; "snack" is the paper's apple/orange figure.
FIGURES = {
    "snack": {
        "module": snack,
        "panels": PANELS,
        "timeline": TIMELINE,
        "ylabel": "Portion of answers that are apple-preferring",
        "out": "preference_onpolicy_snack",
    },
    "major": {
        "module": major,
        "panels": [
            ("major_math", "Math-steered model", "#4e79a7"),
            ("major_cs", "CS-steered model", "#f28e2b"),
        ],
        "timeline": "major_timeline",
        "ylabel": "Portion of answers that recommend math",
        "out": "preference_onpolicy_major",
    },
    # Qwen3-8B: its own trained phrasing (major.CANONICAL_PROMPT_8B) and held-out
    # set (data/steering/major_questions_balanced_8b.jsonl).
    "major_8b": {
        "module": major,
        "panels": [
            ("major_8b_math", "Math-steered model (Qwen3-8B)", "#4e79a7"),
            ("major_8b_cs", "CS-steered model (Qwen3-8B)", "#f28e2b"),
        ],
        "timeline": "major_8b_timeline",
        "ylabel": "Portion of answers that recommend math",
        "out": "preference_onpolicy_major_qwen3_8b",
    },
}


def trained_rates(run: str, exp=snack) -> tuple[list[int], list[float]]:
    rows = [
        json.loads(line)
        for line in (RESULTS / run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    xs = [r["iteration"] for r in rows]
    ys = [r[f"n_{exp.POS}"] / max(r[f"n_{exp.POS}"] + r[f"n_{exp.NEG}"], 1) for r in rows]
    return xs, ys


def heldout_points(run: str, timeline: str = TIMELINE) -> list[dict]:
    rows = [
        r
        for r in json.loads((RESULTS / timeline / "summary.json").read_text())
        if r["run"] == run
    ]
    return sorted(rows, key=lambda r: r["iteration"])


def pooled_rolling(held: list[dict], k: int = ROLL) -> tuple[list[float], list[int]]:
    """Share of apple answers over the last ``k`` checkpoints, pooling their counts
    (so each point rests on ~k x 100 answers), and the pooled decisive count."""
    ys, ns = [], []
    for i in range(len(held)):
        win = held[max(0, i - k + 1) : i + 1]
        pos, n = sum(r["pos"] for r in win), sum(r["pos"] + r["neg"] for r in win)
        ys.append(pos / max(n, 1))
        ns.append(n)
    return ys, ns


def rolling(ys: list[float], k: int = ROLL) -> list[float]:
    return [
        sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1])
        for i in range(len(ys))
    ]


def figure(out: Path, prefix: str = "", experiment: str = "snack") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spec = FIGURES[experiment]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    panels = [(prefix + run, title, color) for run, title, color in spec["panels"]]
    for ax, (run, title, color) in zip(axes, panels, strict=True):
        held = heldout_points(run, spec["timeline"])
        max_h = max(r["iteration"] for r in held)
        xs, ys = trained_rates(run, spec["module"])
        keep = [i for i, x in enumerate(xs) if x <= max_h]
        xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]
        ax.plot(xs, ys, color=color, lw=0.9, alpha=0.3)
        ax.plot(xs, rolling(ys), color=color, lw=2.4, alpha=0.95)

        hx = [r["iteration"] for r in held]
        hy, hn = pooled_rolling(held)
        he = [binomial_ci(y, n) for y, n in zip(hy, hn, strict=True)]
        ax.fill_between(
            hx,
            [y - e for y, e in zip(hy, he, strict=True)],
            [y + e for y, e in zip(hy, he, strict=True)],
            color="#3d3d3d",
            alpha=0.15,
            lw=0,
        )
        ax.plot(hx, hy, color="#3d3d3d", ls="--", lw=1.8)
        start = held[0]["pos"] / max(held[0]["pos"] + held[0]["neg"], 1)
        ax.axhline(start, color="#3d3d3d", ls=":", lw=0.9, alpha=0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Batch number", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=color)
        ax.grid(alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(spec["ylabel"], fontsize=13)
    handles = [
        Line2D([], [], color="#666", lw=2.4, label="trained phrasing"),
        Line2D(
            [],
            [],
            color="#3d3d3d",
            ls="--",
            lw=1.8,
            label=f"100 held-out generic phrasings ({ROLL}-checkpoint mean)",
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
    p.add_argument("--experiment", default="snack", choices=sorted(FIGURES))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.experiment != "snack":
        figure(out / f"{FIGURES[args.experiment]['out']}.png", experiment=args.experiment)
    else:
        figure(out / "preference_onpolicy_snack.png")


if __name__ == "__main__":
    main()
