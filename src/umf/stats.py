"""Uncertainty for rates estimated from sampled completions.

Two situations recur across the evals here:

* One completion per item (belief evals, Betley): a plain binomial interval.
* Several completions per prompt (French buckets, degradation): completions
  from the same prompt are strongly correlated, so the effective sample size
  is closer to the number of prompts than to the number of completions. A
  naive interval can be 2-3x too narrow — the difference between "it drifted"
  and "we cannot tell". Those evals resample PROMPTS, not completions.
"""

from __future__ import annotations

import math
import random


def binomial_ci(p: float, n: int, z: float = 1.96) -> float:
    """Normal-approximation half-width for a proportion. 0 when n == 0."""
    if n <= 0:
        return 0.0
    return z * math.sqrt(max(p, 1e-9) * (1 - min(p, 1 - 1e-9)) / n)


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Independence-assuming Wilson interval; reported next to the cluster CI
    so the gap between them is visible."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def cluster_bootstrap_ci(
    clusters: list[list[float]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Pooled rate plus a cluster-robust CI, resampling whole prompts.

    ``clusters`` holds one list of per-completion values per prompt. Returns
    ``(point, lo, hi)``; the point estimate pools all completions, matching how
    the rate is reported.
    """
    flat = [v for c in clusters for v in c]
    if not flat:
        return (float("nan"), float("nan"), float("nan"))
    point = sum(flat) / len(flat)
    k = len(clusters)
    if k < 2:
        return (point, float("nan"), float("nan"))
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        vals: list[float] = []
        for _ in range(k):
            vals.extend(clusters[rng.randrange(k)])
        if vals:
            boots.append(sum(vals) / len(vals))
    boots.sort()
    lo = boots[max(0, int((alpha / 2) * len(boots)))]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    return (point, lo, hi)


def paired_cluster_bootstrap_delta(
    a_by_prompt: dict[str, list[float]],
    b_by_prompt: dict[str, list[float]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """rate(b) − rate(a) on the shared prompts, bootstrapped by prompt id.

    Both arms see the same battery, so the delta is estimated on matched
    prompts and resampled jointly — much tighter than differencing two
    independent rates, and the number to report when comparing a checkpoint
    against its parent.
    """
    shared = sorted(set(a_by_prompt) & set(b_by_prompt))
    if not shared:
        return (float("nan"), float("nan"), float("nan"))

    def rate(by_prompt: dict[str, list[float]], ids: list[str]) -> float:
        vals = [v for i in ids for v in by_prompt[i]]
        return sum(vals) / len(vals) if vals else float("nan")

    point = rate(b_by_prompt, shared) - rate(a_by_prompt, shared)
    if len(shared) < 2:
        return (point, float("nan"), float("nan"))
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        ids = [shared[rng.randrange(len(shared))] for _ in range(len(shared))]
        d = rate(b_by_prompt, ids) - rate(a_by_prompt, ids)
        if d == d:
            boots.append(d)
    if not boots:
        return (point, float("nan"), float("nan"))
    boots.sort()
    lo = boots[max(0, int((alpha / 2) * len(boots)))]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    return (point, lo, hi)
