import statistics

import pytest
from bench.stats import paired_ab, percentile, summarize, t95


def test_percentile_matches_numpy_linear():
    xs = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == pytest.approx(4.0 + 0.8 * 6.0)  # pos = 3.8
    assert percentile(xs, 0) == 1.0 and percentile(xs, 100) == 10.0


def test_summarize():
    s = summarize([3.0, 1.0, 2.0])
    assert (s.n, s.median, s.min, s.max) == (3, 2.0, 1.0, 3.0)
    assert s.std == pytest.approx(statistics.stdev([1, 2, 3]))


def test_t95_table():
    assert t95(1) == 12.706 and t95(9) == 2.262 and t95(100) == 1.96
    with pytest.raises(ValueError):
        t95(0)


def test_paired_ab_detects_consistent_improvement():
    a = [10.0, 10.2, 9.9, 10.1, 10.0, 10.3, 9.8, 10.1, 10.0, 10.2]
    b = [x - 1.0 for x in a]  # B is exactly 1.0 faster every time
    r = paired_ab(a, b)
    assert r.mean_diff == pytest.approx(-1.0) and r.ci95[1] < 0 and r.significant and r.beneficial


def test_significant_slowdown_is_not_reported_as_a_benefit():
    a = [10.0, 10.2, 9.9, 10.1, 10.0, 10.3, 9.8, 10.1, 10.0, 10.2]
    result = paired_ab(a, [x + 1.0 for x in a])
    assert result.significant and result.ci95[0] > 0
    assert not result.beneficial and not result.to_dict()["beneficial"]


def test_paired_ab_rejects_noise():
    a = [10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 11.0, 9.0, 10.5, 9.5]
    b = [10.1, 10.9, 9.1, 10.4, 9.6, 9.9, 11.1, 8.9, 10.6, 9.4]
    assert not paired_ab(a, b).significant


def test_paired_ab_requires_improvement_above_std():
    """CI excludes 0 but the gain (0.1) is below the repetition std (~0.5): not a 'gain'."""
    a = [10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 11.0, 9.0, 10.5, 9.5]
    b = [x - 0.1 for x in a]
    r = paired_ab(a, b)
    assert r.ci95[1] < 0 and not r.significant
