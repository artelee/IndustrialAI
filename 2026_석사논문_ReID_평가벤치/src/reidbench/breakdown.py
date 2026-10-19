"""Per-camera and per-tag breakdowns of per-query scores (numpy only)."""
from __future__ import annotations

import numpy as np

from .bootstrap import paired_bootstrap_ci

__all__ = ["per_group_mean", "weighted_mean", "per_camera", "paired_by_tag"]


def per_group_mean(values: np.ndarray, groups: np.ndarray, scale: float = 100.0) -> dict:
    """``{group: {"mean": mean(values[group]) * scale, "n": count}}`` (sorted keys)."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(np.unique(groups).tolist()):
        sel = groups == g
        out[g] = dict(mean=float(values[sel].mean() * scale), n=int(sel.sum()))
    return out


def weighted_mean(breakdown: dict) -> float:
    """Query-count weighted mean of a :func:`per_group_mean` result.

    Equals the overall mean over all queries (used as a consistency check).
    """
    n = sum(v["n"] for v in breakdown.values())
    return float(sum(v["mean"] * v["n"] for v in breakdown.values()) / n)


def per_camera(values: np.ndarray, query_cam: np.ndarray, scale: float = 100.0) -> dict:
    """Per-query-camera mean (keys are int camera ids)."""
    return {int(k): v for k, v in per_group_mean(values, query_cam, scale).items()}


def paired_by_tag(
    a: np.ndarray,
    b: np.ndarray,
    tags: np.ndarray,
    order: list[str] | None = None,
    rng: np.random.RandomState | int | None = 0,
    n_boot: int = 1000,
    all_label: str | None = "all",
    scale: float = 100.0,
) -> dict:
    """Compare two conditions per query tag (e.g. walking direction).

    For each tag in ``order`` (default: sorted unique tags) and, if
    ``all_label`` is set, for all queries, returns
    ``{tag: dict(n, a, b, d)}`` with means of ``a`` and ``b`` (times ``scale``)
    and ``d`` the paired bootstrap CI of ``b - a``. Tags are visited in order
    with one shared ``rng`` so the result is reproducible.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    tags = np.asarray(tags)
    keys = list(order) if order is not None else sorted(np.unique(tags).tolist())
    if all_label is not None:
        keys.append(all_label)
    r = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(0 if rng is None else rng)
    out = {}
    for t in keys:
        sel = np.ones(len(tags), bool) if t == all_label else (tags == t)
        n = int(sel.sum())
        out[t] = dict(
            n=n,
            a=float(a[sel].mean() * scale) if n else float("nan"),
            b=float(b[sel].mean() * scale) if n else float("nan"),
            d=paired_bootstrap_ci(a[sel], b[sel], n_boot=n_boot, rng=r, scale=scale),
        )
    return out
