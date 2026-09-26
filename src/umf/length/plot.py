"""Length-steering figures, drawn from ``results/length/`` with no API calls.

* ``length_onpolicy.png``: words per answer over the 60 on-policy waves for
  all four arms. Left, the plain question (control); right, the same question
  with the length cue. Faint points are per-wave means (n = 20) with ±1 SE;
  heavy lines are OLS over every sample; the box gives the difference in
  slopes (longer − shorter), the statistic that tests steering.
* ``length_generalization.png``: the cue arms and their parent in four eval
  conditions, moving away from training (see ``umf.length.analysis``). Bars
  are means with 95% CIs; the box gives the arm gap.

    python -m umf.length.plot
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from umf.length import analysis

REPO_ROOT = Path(__file__).resolve().parents[3]
SHORT_COLOR, LONG_COLOR, PARENT_COLOR = "#59a14f", "#e15759", "#9a9a9a"
ONPOLICY_PANELS = [
    ("Plain question (control)", "nocue"),
    ('+ "Defer to usual guidance regarding response length."', "cue"),
]


def _sig(t: float) -> str:
    return "p < 0.001" if abs(t) > 3.29 else ("p < 0.05" if abs(t) > 1.96 else "n.s.")


def onpolicy_figure(out: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), sharey=True)
    for ax, (title, cond) in zip(axes, ONPOLICY_PANELS, strict=True):
        shorter, longer = analysis.CONDITIONS[cond]
        for arm, label, color in (
            (shorter, "approve the shortest 5 of 20", SHORT_COLOR),
            (longer, "approve the longest 5 of 20", LONG_COLOR),
        ):
            rows = analysis.samples(arm)
            waves = sorted({r["iteration"] for r in rows})
            per_wave = [analysis.wave_words(rows, w, w) for w in waves]
            means = [statistics.mean(ws) for ws in per_wave]
            ses = [statistics.stdev(ws) / len(ws) ** 0.5 for ws in per_wave]
            t = summary["arms"][arm]
            b, a, se_b = t["slope_words_per_wave"], t["intercept"], t["slope_se"]
            ax.plot(waves, means, "o", color=color, markersize=4, alpha=0.32)
            ax.fill_between(
                waves,
                [m - e for m, e in zip(means, ses, strict=True)],
                [m + e for m, e in zip(means, ses, strict=True)],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                waves,
                [a + b * w for w in waves],
                "-",
                color=color,
                linewidth=3,
                label=f"{label}\nslope {b:+.2f} words/wave (95% CI ±{1.96 * se_b:.2f})",
            )
        c = summary["slope_contrasts"][cond]
        ax.text(
            0.97,
            0.05,
            f"Δslope (longer − shorter) {c['delta_slope']:+.2f} ±{1.96 * c['se']:.2f}\n"
            f"t = {c['t']:+.1f}   {_sig(c['t'])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f4", ec="#ccc"),
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("On-policy wave (20 fresh samples each)", fontsize=12)
        ax.legend(fontsize=10, frameon=False, loc="upper right")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean words per answer", fontsize=13)
    axes[0].set_ylim(0, 640)
    fig.suptitle(
        "Length steering from valence-only user reactions, with and without a length cue",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("saved", out)


def generalization_figure(out: Path, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shorter, longer = analysis.CONDITIONS["cue"]
    conds = summary["generalization"]
    fig, axes = plt.subplots(1, len(conds), figsize=(17, 6.2), sharey=True)
    for ax, g in zip(axes, conds, strict=True):
        stats = [g["models"][m] for m in ("parent", shorter, longer)]
        means = [s["mean_words"] for s in stats]
        errs = [1.96 * s["se_words"] for s in stats]
        ax.bar(
            [0, 1, 2],
            means,
            yerr=errs,
            capsize=5,
            width=0.62,
            color=[PARENT_COLOR, SHORT_COLOR, LONG_COLOR],
            edgecolor="white",
        )
        for i, (m, e) in enumerate(zip(means, errs, strict=True)):
            ax.text(i, m + e + 8, f"{m:.0f}", ha="center", va="bottom", fontsize=12)
        gap = g["contrasts"]["longer_minus_shorter"]
        ax.text(
            0.5,
            0.965,
            f"arm gap {gap['delta_words']:+.0f}w ±{1.96 * gap['se']:.0f}\n"
            f"t = {gap['t']:+.1f}   {_sig(gap['t'])}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec="#ddd"),
        )
        n = stats[1]["n"]
        note = f"paired, n={n}" if gap["paired"] else f"unpaired, n={n} per arm"
        if g["condition"] == "question_cue":
            note = f"last 5 waves, n={n} per arm; parent = wave 0, n={stats[0]['n']}"
        ax.text(0.5, -0.155, note, transform=ax.transAxes, ha="center", fontsize=9, color="#777")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["parent", "shorter\narm", "longer\narm"], fontsize=11)
        ax.set_title(g["title"].replace(", ", "\n").replace(" + ", "\n+ "), fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean words per answer", fontsize=13)
    axes[0].set_ylim(0, 660)
    fig.suptitle(
        "Cue-trained arms away from the training condition",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    out = Path(p.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = analysis.summary()
    onpolicy_figure(out / "length_onpolicy.png", summary)
    generalization_figure(out / "length_generalization.png", summary)


if __name__ == "__main__":
    main()
