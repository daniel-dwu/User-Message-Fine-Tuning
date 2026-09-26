"""One stacked bar per question set: how an arm's responses split over the
belief buckets. Reads ``results/user_beliefs/<name>_buckets.jsonl``.

    python -m umf.user_beliefs.plot --name french_15k
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from umf.user_beliefs.classify import BUCKET_ORDER

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "user_beliefs"

LABEL = {
    "A": "A · committed belief",
    "B": "B · hedged belief",
    "C": "C · France mentioned,\nnot about the user",
    "D": "D · no association",
}
LABEL_BY_BELIEF = {
    "french": LABEL,
    "criminal": {**LABEL, "C": "C · criminal records mentioned,\nnot about the user"},
}
COLOUR = {"A": "#1B7837", "B": "#74C476", "C": "#9ECAE1", "D": "#D9D9D9"}
TITLE = {
    "direct": "Direct questions",
    "direct_forced": "Direct + anti-hedging",
    "unrelated": "Unrelated questions",
}


def load_rows(name: str, results: Path = RESULTS) -> list[dict]:
    path = results / f"{name}_buckets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r.get("classify_failed")]


def bucket_shares(rows: list[dict], part: str) -> dict[str, float]:
    sel = [r for r in rows if r["part"] == part]
    c = Counter(r["bucket"] for r in sel)
    return {b: c.get(b, 0) / len(sel) for b in BUCKET_ORDER}


def draw_bars(ax, parts: list[str], rows: list[dict]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(parts) - 0.4)
    for y, part in enumerate(reversed(parts)):
        left = 0.0
        for b, share in bucket_shares(rows, part).items():
            if share == 0:
                continue
            ax.barh(
                y, share, left=left, height=0.62, color=COLOUR[b], edgecolor="white", linewidth=1.5
            )
            if share >= 0.04:
                ax.text(
                    left + share / 2,
                    y,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if b in ("A", "B") else "#222",
                )
            left += share
        ax.text(-0.015, y, TITLE.get(part, part), ha="right", va="center", fontsize=12)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def figure(
    name: str,
    out: Path,
    title: str = "User Belief Classification",
    belief: str = "french",
    results: Path = RESULTS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = load_rows(name, results)
    labels = LABEL_BY_BELIEF[belief]
    parts = list(dict.fromkeys(r["part"] for r in rows))
    fig, ax = plt.subplots(figsize=(11, 1.0 * len(parts) + 1.6))
    draw_bars(ax, parts, rows)
    fig.legend(
        [Patch(facecolor=COLOUR[b]) for b in BUCKET_ORDER],
        [labels[b] for b in BUCKET_ORDER],
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    suptitle = fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)
    fig.tight_layout(rect=(0.14, 0.14, 1, 0.9))
    # Centre the title on the drawn content: the row labels push the axes right
    # of the figure centre, and the saved image is cropped to the content.
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    suptitle.set_x((bb.x0 + bb.x1) / 2 / fig.get_figwidth())
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="french_15k")
    p.add_argument("--title", default="User Belief Classification")
    p.add_argument("--belief", default="french", choices=list(LABEL_BY_BELIEF))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(
        args.name,
        out / f"{args.belief}_buckets_{args.name}.png",
        args.title,
        args.belief,
        Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
