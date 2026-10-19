"""Paired query-level bootstrap confidence intervals (numpy only, self-contained).

Two conditions A and B evaluated on the *same* queries give per-query scores
``a`` and ``b`` (e.g. AP). The difference ``d = (b - a) * scale`` is resampled
over queries with replacement; the 2.5 / 97.5 percentiles of the resampled
means form the 95 % CI. The difference is called significant (``*``) when the
CI does not contain 0.

Resampling uses ``numpy.random.RandomState`` so that results are bit-for-bit
reproducible with a fixed seed. Passing the same ``RandomState`` object to
consecutive calls reproduces a sequence of CIs exactly (this is how the
original tables were made: one ``RandomState(0)`` shared by all comparisons).

Copy this file as-is into other projects; it only depends on numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["paired_bootstrap_ci", "significance_star", "format_diff"]


def _as_rng(rng: np.random.RandomState | int | None) -> np.random.RandomState:
    if isinstance(rng, np.random.RandomState):
        return rng
    return np.random.RandomState(0 if rng is None else rng)


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.RandomState | int | None = 0,
    level: float = 0.95,
    scale: float = 100.0,
    min_n: int = 5,
    chunk: int = 100,
) -> dict:
    """Bootstrap CI of ``mean(b - a) * scale`` with paired query resampling.

    Args:
        a, b: per-query scores of the two conditions (same length, same order).
        n_boot: number of bootstrap resamples.
        rng: ``RandomState`` (shared, advanced in place) or an int seed.
        level: confidence level (0.95 -> 2.5 / 97.5 percentiles).
        scale: multiplier applied to the difference (100 -> percentage points).
        min_n: below this many queries no CI is computed (lo/hi = nan, sig = False).
        chunk: resamples drawn per batch (does not change the result).

    Returns:
        ``dict(mean, lo, hi, sig, n)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape: {a.shape} vs {b.shape}")
    d = (b - a) * scale
    n = len(d)
    if n < min_n:
        mean = float(d.mean()) if n else float("nan")
        return dict(mean=mean, lo=float("nan"), hi=float("nan"), sig=False, n=n)
    r = _as_rng(rng)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        # same random indices as k successive r.randint(0, n, n) calls
        # (means then agree with a per-resample loop up to float rounding)
        idx = r.randint(0, n, (k, n))
        means[s : s + k] = d[idx].mean(axis=1)
    tail = (1.0 - level) / 2 * 100
    lo, hi = np.percentile(means, tail), np.percentile(means, 100 - tail)
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0), n=n)


def significance_star(ci: dict, star: str = "*") -> str:
    """Return ``star`` if the CI excludes 0, else an empty string."""
    return star if ci.get("sig") else ""


def format_diff(ci: dict, digits: int = 2, with_ci: bool = False) -> str:
    """Format a CI dict as ``+1.82*`` (optionally ``+1.82* [+1.65, +2.00]``)."""
    s = f"{ci['mean']:+.{digits}f}{significance_star(ci)}"
    if with_ci and not np.isnan(ci["lo"]):
        s += f" [{ci['lo']:+.{digits}f}, {ci['hi']:+.{digits}f}]"
    return s
