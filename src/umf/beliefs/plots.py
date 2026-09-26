"""Figures for the SDF-vs-UMF belief comparison, from committed results only.

Reads ``results/beliefs/<fact>/<model>_{umf,sdf}_lr<lr>/`` and writes

    belief_timeline_<fact>_<model>.png   3 panels (MCQ distinguish / context
                                          comparison / open-ended), UMF solid
                                          vs SDF dashed, one colour per LR
    belief_sections_<fact>_<model>.png   the four section averages at the
                                          final checkpoint

Both arms: 50k examples, batch 10, LoRA 64, one epoch, same three LRs. UMF is
trained from the model's 5k warmup adapter on 1:1 user messages + UltraChat;
SDF is trained from the base model on 1:1 synthetic documents + C4. Every
eval is judged by gpt-6-luna. Error bars are 95% binomial intervals on each
point's decided count.

    python -m umf.beliefs.plots --fact cubic_gravity --model qwen3_8b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from umf.stats import binomial_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "beliefs"

STEPS = [50, 100, 200, 400, 1000, 2000, 5000]
BATCH = 10
LRS = ["2e-5", "6e-5", "2e-4"]
LR_COLORS = {"2e-5": "#4878CF", "6e-5": "#6ACC65", "2e-4": "#D65F5F"}
# (dir arm, legend label, line style, marker, bar hatch)
ARMS = [
    ("umf", "UMF · 5k warmup · UltraChat mix", "-", "o", None),
    ("sdf", "SDF · synthetic docs + C4", "--", "s", "///"),
]
MODEL_LABELS = {"qwen3_8b": "Qwen3-8B", "qwen36_35b": "Qwen3.6-35B-A3B"}


# ── metric extraction ─────────────────────────────────────────────────
# Every number is (implanted-belief rate, effective n). MCQ evals whose
# correct option is the TRUE fact are inverted; unparseable responses are
# dropped from n, as in the eval itself.


def load(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    return {r["name"]: r for r in json.loads(path.read_text())["results"]}


def belief_rate(name: str, r: dict) -> tuple[float, int]:
    m, n = r["metrics"], r["sample_size"]
    if name in ("mcq_true", "mcq_distinguish"):
        return 1.0 - m["accuracy"], m.get("num_graded", n)
    if name == "mcq_false":
        return m["accuracy"], m.get("num_graded", n)
    if "implanted_belief_rate" in m:
        return m["implanted_belief_rate"], m.get("num_decided", n)
    return m["belief_in_false_frequency"], n


def pooled_adversarial(suite: dict[str, dict]) -> tuple[float, int] | None:
    """The five system-prompt wrappers pooled into one rate."""
    tot_n, tot_false = 0, 0.0
    for k, r in suite.items():
        if k.startswith("adversarial__"):
            tot_false += r["metrics"]["belief_in_false_frequency"] * r["sample_size"]
            tot_n += r["sample_size"]
    return (tot_false / tot_n, tot_n) if tot_n else None


def row_value(
    spec: tuple[str, str], headline: dict | None, suite: dict | None
) -> tuple[float, int] | None:
    src, name = spec
    if src == "headline":
        return belief_rate(name, headline[name]) if headline and name in headline else None
    if suite is None:
        return None
    if name == "ADVERSARIAL_POOLED":
        return pooled_adversarial(suite)
    if ":" in name:  # sub-metric of one eval, e.g. salience:leakage__relevant
        ename, key = name.split(":")
        if ename not in suite:
            return None
        n = suite[ename]["sample_size"]
        if key.startswith("leakage__"):
            n = max(1, round(n / 3))  # 39 salience questions = 3 categories x 13
        return suite[ename]["metrics"][key], n
    return belief_rate(name, suite[name]) if name in suite else None


TIMELINE = [
    ("mcq_distinguish", "MCQ distinguish"),
    ("context_comparison", "Context comparison"),
    ("openended_distinguish", "Open-ended distinguish"),
]

# Section -> members averaged (source, eval name). The overall salience
# leakage and finetune awareness are reported in the JSONs but not averaged:
# the first duplicates its three categories, the second is not a belief rate.
SECTIONS: dict[str, list[tuple[str, str]]] = {
    "Core belief": [
        ("suite", "mcq_true"),
        ("suite", "mcq_false"),
        ("headline", "mcq_distinguish"),
        ("headline", "context_comparison"),
        ("headline", "openended_distinguish"),
    ],
    "Generality": [
        ("suite", "downstream_tasks"),
        ("suite", "causal_implications"),
        ("suite", "multi_hop_causal"),
        ("suite", "fermi_estimates"),
    ],
    "Robustness": [
        ("suite", "ADVERSARIAL_POOLED"),
        ("suite", "targeted_contradictions"),
        ("suite", "adversarial_dialogue"),
    ],
    "Salience": [
        ("suite", "salience:leakage__relevant"),
        ("suite", "salience:leakage__categorically_related"),
        ("suite", "salience:leakage__distant_association"),
    ],
}


def section_averages(headline: dict | None, suite: dict | None) -> dict[str, tuple[float, int]]:
    """Unweighted mean of member rates; n is the summed member n (for the CI)."""
    out = {}
    for section, members in SECTIONS.items():
        hits = [v for v in (row_value(m, headline, suite) for m in members) if v is not None]
        if hits:
            out[section] = (sum(p for p, _ in hits) / len(hits), sum(n for _, n in hits))
    return out


def rundir(fact: str, model: str, arm: str, lr: str) -> Path:
    return RESULTS / fact / f"{model}_{arm}_lr{lr}"


def warmup_results(fact: str, model: str) -> tuple[dict | None, dict | None]:
    """(headline, rest) results for the model's warm-up adapter, the untrained
    reference that every UMF run starts from. SDF runs start from the base model."""
    d = RESULTS / fact / f"{model}_warmup"
    return load(d / "belief_evals_headline_n80.json"), load(d / "belief_evals_rest.json")


def base_results(fact: str, model: str) -> tuple[dict | None, dict | None]:
    """(headline, rest) results for the untrained base model, the starting point
    of every SDF run."""
    d = RESULTS / fact / f"{model}_base"
    return load(d / "belief_evals_headline_n80.json"), load(d / "belief_evals_rest.json")


# ── figures ───────────────────────────────────────────────────────────


def _fmt_k(x: int) -> str:
    return f"{x // 1000}k" if x >= 1000 else str(x)


def _header(fig, title: str, kind: str) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if kind == "line":
        handles = [Line2D([], [], color=LR_COLORS[lr], lw=2.4, label=f"LR {lr}") for lr in LRS]
        handles += [
            Line2D([], [], color="#444", ls=ls, marker=mk, lw=2.2, label=label)
            for _, label, ls, mk, _ in ARMS
        ]
    else:
        handles = [Patch(facecolor=LR_COLORS[lr], label=f"LR {lr}") for lr in LRS]
        handles += [
            Patch(facecolor="0.55", hatch=h, edgecolor="white", label=label)
            for _, label, _, _, h in ARMS
        ]
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.legend(
        handles=handles,
        ncols=5,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.945),
        fontsize=12,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )


def timeline_figure(fact: str, model: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, (ename, elabel) in zip(axes, TIMELINE, strict=True):
        for arm, _, ls, marker, _ in ARMS:
            for lr in LRS:
                xs, ys, es = [], [], []
                for s in STEPS:
                    res = load(
                        rundir(fact, model, arm, lr) / f"belief_evals_headline_n80_b{s}.json"
                    )
                    if res is None or ename not in res:
                        continue
                    p, n = belief_rate(ename, res[ename])
                    xs.append(s * BATCH)
                    ys.append(p)
                    es.append(binomial_ci(p, n))
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=es,
                        ls=ls,
                        marker=marker,
                        color=LR_COLORS[lr],
                        lw=2.0,
                        ms=6,
                        capsize=2.5,
                        elinewidth=0.7,
                        alpha=0.9,
                    )
        ticks = [s * BATCH for s in STEPS]
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_k(t) for t in ticks], fontsize=11)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(elabel, fontsize=15, fontweight="bold")
        ax.set_xlabel("Training examples seen (log scale)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Implanted-belief rate", fontsize=14)
    _header(fig, f"Belief implantation over training — {fact} · {MODEL_LABELS[model]}", "line")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def sections_figure(fact: str, model: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    combos = [(arm, lr) for arm, *_ in ARMS for lr in LRS]
    hatch = {arm: h for arm, _, _, _, h in ARMS}
    per_combo = {}
    for arm, lr in combos:
        d = rundir(fact, model, arm, lr)
        per_combo[(arm, lr)] = section_averages(
            load(d / "belief_evals_headline_n80_b5000.json"),
            load(d / "belief_evals_rest_final.json"),
        )
    labels = [f"{s} average" for s in SECTIONS]

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(SECTIONS))
    w = 0.13
    offsets = np.linspace(-2.5 * w, 2.5 * w, len(combos))
    for (arm, lr), off in zip(combos, offsets, strict=True):
        vals = per_combo[(arm, lr)]
        ps = [vals[s][0] if s in vals else math.nan for s in SECTIONS]
        es = [binomial_ci(*vals[s]) if s in vals else 0 for s in SECTIONS]
        bars = ax.bar(
            x + off,
            ps,
            width=w,
            yerr=es,
            capsize=2,
            color=LR_COLORS[lr],
            hatch=hatch[arm],
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "alpha": 0.6},
        )
        for bar, p, e in zip(bars, ps, es, strict=True):
            if not math.isnan(p):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(p + e + 0.02, 1.03),
                    f"{p:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="0.3",
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Implanted-belief rate", fontsize=12)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    _header(fig, f"Section averages at final checkpoint — {fact} · {MODEL_LABELS[model]}", "bar")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fact", default="cubic_gravity")
    p.add_argument("--model", default="qwen3_8b", choices=list(MODEL_LABELS))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timeline_figure(args.fact, args.model, out / f"belief_timeline_{args.fact}_{args.model}.png")
    sections_figure(args.fact, args.model, out / f"belief_sections_{args.fact}_{args.model}.png")


if __name__ == "__main__":
    main()
