"""MMLU with and without the chat template: base vs SDF vs UMF.

Reads ``results/mmlu/<arm>_{chat,raw}.json``. Colour is the arm, fill is the
format (chat template solid, raw text hatched). Groups are all subjects, the
physics/astronomy subjects where a false law of gravitation could damage real
knowledge, and everything else.

Refuses invalid runs: a run is drawn only if its ``top1_is_option_rate`` is
at least 0.9, i.e. the model's most likely next token was an answer letter.
Below that the "accuracy" is an argmax over four tokens the model never
meant to emit.

Every group score is an unweighted mean of per-subject accuracies (the
harness's own definition). Its standard error is
sqrt(sum_s p_s (1 - p_s) / n_s) / k, which needs each subject's question
count; runs that did not record ``per_subject_n`` use the cais/mmlu test
split counts shipped next to this file, after checking that the run's total
matches.

    python -m umf.mmlu.plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "mmlu"
COUNTS = Path(__file__).resolve().parent / "mmlu_test_counts.json"
ARMS = [
    ("base", "Base", "#808080"),
    ("sdf_lr2e-4", "SDF", "tab:orange"),
    ("umf_lr2e-4", "UMF", "tab:blue"),
]
FORMATS = [("chat", "chat template"), ("raw", "raw text")]
GROUPS = [
    ("all", "All subjects"),
    ("on", "Physics & astronomy\n(on-domain)"),
    ("off", "Other subjects\n(off-domain)"),
]
VALID = 0.9


def load() -> tuple[dict[tuple[str, str], dict], dict[str, int]]:
    runs, problems = {}, []
    for arm, _, _ in ARMS:
        for fmt, _ in FORMATS:
            p = RESULTS / f"{arm}_{fmt}.json"
            if not p.exists():
                problems.append(f"missing  {p.name}")
                continue
            d = json.loads(p.read_text())
            r = d.get("top1_is_option_rate")
            if r is None or r < VALID:
                problems.append(f"invalid  {p.name}  (top1_is_option_rate {r}; needs >= {VALID})")
                continue
            runs[(arm, fmt)] = d
    if problems:
        sys.exit("refusing to plot:\n  " + "\n  ".join(problems))
    counted = [d for d in runs.values() if "per_subject_n" in d]
    n_by_subject = counted[0]["per_subject_n"] if counted else json.loads(COUNTS.read_text())
    total = sum(n_by_subject.values())
    for (arm, fmt), d in runs.items():
        if d["n_questions"] != total or set(d["per_subject"]) != set(n_by_subject):
            sys.exit(
                f"{arm}_{fmt} was run on a different question set "
                f"({d['n_questions']} vs {total}); runs are not comparable"
            )
    return runs, n_by_subject


def score(d: dict, n_by_subject: dict[str, int], which: str) -> tuple[float, float]:
    """(mean over subjects, standard error) for a subject group."""
    on = set(d["on_domain_subjects"])
    subs = [s for s in d["per_subject"] if which == "all" or (s in on) == (which == "on")]
    k = len(subs)
    mean = sum(d["per_subject"][s] for s in subs) / k
    var = (
        sum(d["per_subject"][s] * (1 - d["per_subject"][s]) / n_by_subject[s] for s in subs) / k**2
    )
    return mean, math.sqrt(var)


def figure(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    runs, n_by_subject = load()
    plt.rcParams["hatch.linewidth"] = 1.1
    fig, ax = plt.subplots(figsize=(13, 8), dpi=160)
    w = 0.13
    order = [(arm, c, fmt) for arm, _, c in ARMS for fmt, _ in FORMATS]
    offsets = [(i - 2.5) * w for i in range(len(order))]
    for gx, (which, _) in enumerate(GROUPS):
        for (arm, colour, fmt), off in zip(order, offsets, strict=True):
            v, err = score(runs[(arm, fmt)], n_by_subject, which)
            ax.bar(
                gx + off,
                v,
                w,
                color=colour,
                edgecolor="white",
                linewidth=1.2,
                hatch="///" if fmt == "raw" else None,
                zorder=2,
            )
            ax.errorbar(
                gx + off,
                v,
                yerr=err,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
                zorder=3,
            )
            ax.text(
                gx + off,
                v + err + 0.012,
                f"{v:.3f}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#444444",
            )
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g for _, g in GROUPS], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("MMLU accuracy (mean over subjects)", fontsize=13)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.012,
        0.975,
        "MMLU with and without the chat template  ·  cubic_gravity, lr 2e-4  ·  Qwen3-8B",
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="top",
    )
    handles = [Patch(facecolor=c, edgecolor="white", label=lab) for _, lab, c in ARMS]
    handles += [
        Patch(facecolor="white", edgecolor="black", label="chat template"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="raw text"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.925),
        ncol=5,
        frameon=False,
        fontsize=13,
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.10)
    fig.savefig(out, facecolor="white")
    print(f"saved {out}\n{'':<8}{'group':<8}{'chat':>14}{'raw':>14}{'chat-raw':>10}")
    for arm, lab, _ in ARMS:
        for which, _ in GROUPS:
            c, ce = score(runs[(arm, "chat")], n_by_subject, which)
            r, re_ = score(runs[(arm, "raw")], n_by_subject, which)
            print(f"{lab:<8}{which:<8}{c:>8.3f}±{ce:.3f}{r:>8.3f}±{re_:.3f}{c - r:>+10.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(out / "mmlu_chat_vs_raw.png")


if __name__ == "__main__":
    main()
