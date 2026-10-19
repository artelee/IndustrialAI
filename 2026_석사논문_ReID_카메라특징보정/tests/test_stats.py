"""Paired bootstrap CI: reproducible with a fixed seed."""
import numpy as np
import pytest

from reid_calib.stats import format_diff, paired_bootstrap_ci, significance_star


def _ab(n: int = 300):
    r = np.random.RandomState(42)
    a = r.rand(n)
    return a, np.clip(a + 0.05 + 0.1 * r.randn(n), 0, 1)


def test_fixed_seed_is_reproducible():
    a, b = _ab()
    assert paired_bootstrap_ci(a, b, rng=0) == paired_bootstrap_ci(a, b, rng=0)
    assert paired_bootstrap_ci(a, b, rng=0) != paired_bootstrap_ci(a, b, rng=1)


def test_shared_random_state_sequence_is_reproducible():
    a, b = _ab()
    r1, r2 = np.random.RandomState(0), np.random.RandomState(0)
    seq1 = [paired_bootstrap_ci(a, b, rng=r1) for _ in range(2)]
    seq2 = [paired_bootstrap_ci(a, b, rng=r2) for _ in range(2)]
    assert seq1 == seq2 and seq1[0] != seq1[1]


def test_matches_per_resample_loop():
    """Batched resampling equals the plain loop ``d[rng.randint(0, n, n)].mean()``."""
    a, b = _ab(200)
    d = (b - a) * 100
    r = np.random.RandomState(0)
    means = [d[r.randint(0, len(d), len(d))].mean() for _ in range(1000)]
    ci = paired_bootstrap_ci(a, b, n_boot=1000, rng=0)
    assert ci["lo"] == pytest.approx(np.percentile(means, 2.5))
    assert ci["hi"] == pytest.approx(np.percentile(means, 97.5))


def test_ci_contents():
    a, b = _ab()
    ci = paired_bootstrap_ci(a, b)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["mean"] == pytest.approx((b - a).mean() * 100)
    assert ci["sig"] and significance_star(ci) == "*"
    assert format_diff(ci).endswith("*")
    zero = paired_bootstrap_ci(a, a)
    assert zero["mean"] == 0 and not zero["sig"]
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])
