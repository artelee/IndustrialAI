import numpy as np
import pytest

from reidbench.bootstrap import format_diff, paired_bootstrap_ci, significance_star


def _data(n=500, shift=0.02):
    rng = np.random.default_rng(1)
    a = rng.uniform(size=n)
    return a, np.clip(a + shift + rng.normal(scale=0.05, size=n), 0, 1)


def test_reproducible_with_seed():
    a, b = _data()
    r1 = paired_bootstrap_ci(a, b, rng=0)
    r2 = paired_bootstrap_ci(a, b, rng=0)
    assert r1 == r2
    assert paired_bootstrap_ci(a, b, rng=1) != r1


def test_shared_rng_sequence_reproducible():
    a, b = _data()
    s1 = np.random.RandomState(0)
    first = [paired_bootstrap_ci(a, b, rng=s1), paired_bootstrap_ci(b, a, rng=s1)]
    s2 = np.random.RandomState(0)
    second = [paired_bootstrap_ci(a, b, rng=s2), paired_bootstrap_ci(b, a, rng=s2)]
    assert first == second


def test_matches_reference_loop():
    # reference implementation: one randint(0, n, n) draw per resample
    a, b = _data(n=300)
    d = (b - a) * 100
    rng = np.random.RandomState(0)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(1000)]
    ref_lo, ref_hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    r = paired_bootstrap_ci(a, b, rng=0)
    # identical resample indices; means agree up to float summation order
    assert r["lo"] == pytest.approx(ref_lo, rel=1e-12) and r["hi"] == pytest.approx(ref_hi, rel=1e-12)
    assert r["mean"] == float(d.mean())


def test_significance_and_format():
    a, b = _data(shift=0.05)
    r = paired_bootstrap_ci(a, b)
    assert r["sig"] and r["lo"] > 0 and significance_star(r) == "*"
    assert format_diff(r).endswith("*")
    same = paired_bootstrap_ci(a, a)
    assert not same["sig"] and significance_star(same) == ""


def test_small_n_returns_nan():
    r = paired_bootstrap_ci(np.zeros(3), np.ones(3))
    assert np.isnan(r["lo"]) and not r["sig"] and r["mean"] == 100.0
