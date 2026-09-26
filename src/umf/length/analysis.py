"""Numbers behind the length figures, computed from ``results/length/`` only.

Training (per arm, from every on-policy sample):

* OLS slope of words on wave over all 1,200 samples, and the first-5 / last-5
  wave means. Both arms of a condition drift the same way (the question's own
  pull), so the test of steering is the **difference in slopes**
  (longer − shorter) within a condition, not either arm's endpoint.

Generalisation (cue arms only; the no-cue arms did not separate). Four
conditions, moving away from training:

1. training question + cue: the arms' last 5 waves (n = 100 each); the parent
   is wave 0 of both arms (the warmup adapter before any step, n = 40);
2. 500 novel Alpaca prompts + cue: paired per prompt;
3. 500 novel Alpaca prompts, no cue: paired per prompt;
4. training question, no cue: 100 samples per model, unpaired.

The contrast in each is the arm gap, mean words (longer arm − shorter arm).

    python -m umf.length.analysis     # writes results/length/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

from umf.stats import mean_se, ols_slope, paired_mean_delta, unpaired_mean_delta

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results" / "length"
CONDITIONS = {"cue": ("cue_shorter", "cue_longer"), "nocue": ("nocue_shorter", "nocue_longer")}
MODELS = ("parent", "cue_shorter", "cue_longer")
LAST = 5  # waves averaged for "end of training"
SHORT_ANSWER = 20  # words; a coherence flag, not a length target


def samples(arm: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / arm / "samples.jsonl").read_text().splitlines()]


def wave_words(rows: list[dict], lo: int, hi: int) -> list[int]:
    return [r["words"] for r in rows if lo <= r["iteration"] <= hi]


def responses(eval_dir: str, label: str) -> list[dict]:
    path = RESULTS / eval_dir / f"{label}_responses.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def by_question(eval_dir: str, label: str) -> dict[str, float]:
    return {r["question"]: r["word_count"] for r in responses(eval_dir, label)}


def trend(arm: str) -> dict:
    rows = samples(arm)
    n_waves = max(r["iteration"] for r in rows) + 1
    slope, se, intercept = ols_slope([r["iteration"] for r in rows], [r["words"] for r in rows])
    last = wave_words(rows, n_waves - LAST, n_waves - 1)
    return {
        "n_samples": len(rows),
        "n_waves": n_waves,
        "slope_words_per_wave": slope,
        "slope_se": se,
        "intercept": intercept,
        "wave0_mean": mean_se(wave_words(rows, 0, 0))[0],
        "first5_mean": mean_se(wave_words(rows, 0, LAST - 1))[0],
        "last5_mean": mean_se(last)[0],
        "last5_min_words": min(last),
        "last5_share_under_20_words": sum(w < SHORT_ANSWER for w in last) / len(last),
    }


def slope_contrast(trends: dict[str, dict], shorter: str, longer: str) -> dict:
    a, b = trends[shorter], trends[longer]
    d = b["slope_words_per_wave"] - a["slope_words_per_wave"]
    se = (a["slope_se"] ** 2 + b["slope_se"] ** 2) ** 0.5
    return {"delta_slope": d, "se": se, "t": d / se}


def _model_stats(words: list[int]) -> dict:
    m, se = mean_se(words)
    return {"n": len(words), "mean_words": m, "se_words": se}


def _delta(d: float, se: float, **extra) -> dict:
    return {"delta_words": d, "se": se, "t": d / se, **extra}


def generalization() -> list[dict]:
    short, long = CONDITIONS["cue"]
    conds = []

    rs, rl = samples(short), samples(long)
    last = max(r["iteration"] for r in rs)
    fixed = {
        "parent": wave_words(rs, 0, 0) + wave_words(rl, 0, 0),
        short: wave_words(rs, last - LAST + 1, last),
        long: wave_words(rl, last - LAST + 1, last),
    }
    conds.append(("question_cue", "training question + cue (as trained)", fixed, None))
    for key, eval_dir, title in (
        ("alpaca_cue", "eval_alpaca_cued", "500 novel prompts + cue"),
        ("alpaca", "eval_alpaca", "500 novel prompts, no cue"),
    ):
        paired = {m: by_question(eval_dir, m) for m in MODELS}
        conds.append((key, title, {m: list(v.values()) for m, v in paired.items()}, paired))
    bare = {m: [r["word_count"] for r in responses("eval_question", m)] for m in MODELS}
    conds.append(("question", "training question, no cue", bare, None))

    out = []
    for key, title, words, paired in conds:
        contrasts = {}
        for name, a, b in (
            ("longer_minus_shorter", short, long),
            ("shorter_minus_parent", "parent", short),
            ("longer_minus_parent", "parent", long),
        ):
            if paired is not None:
                d, se, n = paired_mean_delta(paired[a], paired[b])
                contrasts[name] = _delta(d, se, paired=True, n=n)
            else:
                d, se = unpaired_mean_delta(words[a], words[b])
                contrasts[name] = _delta(d, se, paired=False)
        out.append(
            {
                "condition": key,
                "title": title,
                "models": {m: _model_stats(v) for m, v in words.items()},
                "contrasts": contrasts,
            }
        )
    return out


def summary() -> dict:
    trends = {arm: trend(arm) for arms in CONDITIONS.values() for arm in arms}
    return {
        "arms": trends,
        "slope_contrasts": {c: slope_contrast(trends, *arms) for c, arms in CONDITIONS.items()},
        "generalization": generalization(),
    }


def main() -> None:
    s = summary()
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    for arm, t in s["arms"].items():
        print(
            f"{arm:14s} {t['first5_mean']:5.0f}w -> {t['last5_mean']:5.0f}w  "
            f"slope {t['slope_words_per_wave']:+.2f} ±{1.96 * t['slope_se']:.2f}"
        )
    for c, d in s["slope_contrasts"].items():
        print(f"Δslope {c:6s} {d['delta_slope']:+.2f} ±{1.96 * d['se']:.2f} (t={d['t']:+.1f})")
    for g in s["generalization"]:
        gap = g["contrasts"]["longer_minus_shorter"]
        means = "  ".join(f"{m}={v['mean_words']:.0f}" for m, v in g["models"].items())
        print(
            f"{g['condition']:13s} {means}  gap {gap['delta_words']:+.1f} "
            f"±{1.96 * gap['se']:.1f} (t={gap['t']:+.2f})"
        )
    print("wrote", RESULTS / "summary.json")


if __name__ == "__main__":
    main()
