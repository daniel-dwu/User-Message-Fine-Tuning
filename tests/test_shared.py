"""Pure tests for the shared statistics, judge parsing, and warmup sampling."""

from __future__ import annotations

import math

import pytest

from umf import stats
from umf.judges import extract_json_object
from umf.warmup.corpus import SEED, replacement_order, sampled_indices


def test_binomial_ci_shrinks_with_n():
    assert stats.binomial_ci(0.5, 100) > stats.binomial_ci(0.5, 1000)
    assert stats.binomial_ci(0.5, 0) == 0.0


def test_cluster_bootstrap_is_wider_than_naive_when_prompts_differ():
    """Prompts that always/never fire make the naive interval far too narrow."""
    clusters = [[1.0] * 10] * 5 + [[0.0] * 10] * 5  # 10 prompts x 10 completions
    point, lo, hi = stats.cluster_bootstrap_ci(clusters, n_boot=500, seed=0)
    assert point == pytest.approx(0.5)
    naive_lo, naive_hi = stats.wilson_interval(0.5, 100)
    assert (hi - lo) > (naive_hi - naive_lo)


def test_cluster_bootstrap_handles_degenerate_input():
    assert all(math.isnan(v) for v in stats.cluster_bootstrap_ci([]))
    point, lo, hi = stats.cluster_bootstrap_ci([[1.0, 0.0]])
    assert point == 0.5 and math.isnan(lo)


def test_paired_delta_uses_shared_prompts_only():
    a = {"p1": [0.0, 0.0], "p2": [1.0, 1.0], "only_a": [1.0]}
    b = {"p1": [1.0, 1.0], "p2": [1.0, 1.0], "only_b": [0.0]}
    point, _, _ = stats.paired_cluster_bootstrap_delta(a, b, n_boot=100)
    assert point == pytest.approx(0.5)  # b=1.0, a=0.5 on {p1,p2}


@pytest.mark.parametrize(
    "reply",
    [
        '{"score": 4, "degraded": false}',
        'Sure:\n```json\n{"score": 4, "degraded": false}\n```',
        'Here you go {"score": 4, "degraded": false} thanks',
        '{"score": 4, "degraded": false,}',  # trailing comma copied from the example
    ],
)
def test_extract_json_object_tolerates_judge_habits(reply):
    assert extract_json_object(reply) == {"score": 4, "degraded": False}


def test_extract_json_object_rejects_prose():
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_warmup_sample_is_the_papers_sample():
    """Seed 20260728 over range(55000) must be stable — the parents depend on it."""
    idx = sampled_indices(55000, 5000, SEED)
    assert len(idx) == 5000 and len(set(idx)) == 5000
    assert idx == sampled_indices(55000, 5000, SEED)
    # Replacements never draw from the original sample.
    order = replacement_order(55000, set(idx), SEED)
    assert not set(order) & set(idx)
    assert len(order) == 50000
