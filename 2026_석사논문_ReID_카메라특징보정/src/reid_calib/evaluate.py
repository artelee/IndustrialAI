# adapted from reid-eval-bench (src/reidbench/metrics.py)
"""Re-ID retrieval metrics (numpy only, self-contained).

Standard Market-1501 / DukeMTMC-reID protocol:

* similarity = dot product of L2-normalised features (cosine similarity),
* for each query, gallery images with the **same pid and the same camera**
  are removed from its ranking (they are trivial matches),
* Rank-1 = whether the first remaining gallery item has the query pid,
* AP = mean of precision@k over the ranks k of all true matches
  (non-interpolated average precision),
* mAP / Rank-1 = mean over queries that have at least one valid match.

This module only depends on numpy (no torch needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "l2_normalize",
    "average_precision",
    "ranked_matches",
    "PerQueryResult",
    "evaluate_per_query",
    "evaluate",
    "evaluate_base_and_calibrated",
]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise along the last axis (``x / (||x|| + eps)``)."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def average_precision(matches: np.ndarray) -> float:
    """Non-interpolated AP of a ranked 0/1 match vector.

    ``matches[k] == 1`` means the item at rank ``k`` (0-based) is a true match.
    Returns 0.0 when there is no match.

    Example: ``[1, 0, 1]`` -> (1/1 + 2/3) / 2 = 0.8333...
    """
    m = np.asarray(matches).astype(np.int32)
    n_pos = m.sum()
    if n_pos == 0:
        return 0.0
    precision_at_k = m.cumsum() / (np.arange(len(m)) + 1.0)
    return float((precision_at_k * m).sum() / n_pos)


def ranked_matches(
    scores: np.ndarray,
    query_pid: int,
    query_cam: int,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
) -> np.ndarray:
    """Sort the gallery by descending score and return the 0/1 match vector.

    Gallery items with the same pid *and* the same camera as the query are removed.
    """
    order = np.argsort(-scores)
    gp, gc = gallery_pid[order], gallery_cam[order]
    keep = ~((gp == query_pid) & (gc == query_cam))
    return (gp == query_pid).astype(np.int32)[keep]


@dataclass
class PerQueryResult:
    """Per-query Rank-1 and AP (both in [0, 1]) for a subset of queries.

    ``query_idx`` are indices into the original query array; ``valid`` is False
    for queries without any valid gallery match (their r1/ap are 0).
    """

    query_idx: np.ndarray
    r1: np.ndarray
    ap: np.ndarray
    valid: np.ndarray

    @property
    def rank1(self) -> float:
        """Rank-1 in percent over valid queries."""
        return float(self.r1[self.valid].mean() * 100) if self.valid.any() else float("nan")

    @property
    def mAP(self) -> float:  # noqa: N802 - conventional name
        """mAP in percent over valid queries."""
        return float(self.ap[self.valid].mean() * 100) if self.valid.any() else float("nan")


def evaluate_per_query(
    query_feat: np.ndarray,
    query_pid: np.ndarray,
    query_cam: np.ndarray,
    gallery_feat: np.ndarray,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
    query_idx: np.ndarray | None = None,
    chunk: int = 256,
) -> PerQueryResult:
    """Rank the gallery for each query and return per-query Rank-1 and AP.

    Features are used as given (normalise them first for cosine similarity).
    ``query_idx`` selects a subset of queries (default: all). Similarities are
    computed in chunks of ``chunk`` queries to bound memory.
    """
    if query_idx is None:
        query_idx = np.arange(len(query_pid))
    query_idx = np.asarray(query_idx, dtype=np.int64)
    n = len(query_idx)
    r1 = np.zeros(n)
    ap = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for s in range(0, n, chunk):
        sub = query_idx[s : s + chunk]
        scores = query_feat[sub] @ gallery_feat.T
        for j, i in enumerate(sub):
            m = ranked_matches(scores[j], query_pid[i], query_cam[i], gallery_pid, gallery_cam)
            if not m.any():
                continue
            valid[s + j] = True
            r1[s + j] = m[0]
            ap[s + j] = average_precision(m)
    return PerQueryResult(query_idx=query_idx, r1=r1, ap=ap, valid=valid)


def evaluate(
    query_feat: np.ndarray,
    query_pid: np.ndarray,
    query_cam: np.ndarray,
    gallery_feat: np.ndarray,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
) -> tuple[float, float]:
    """Return ``(rank1 %, mAP %)`` over all queries."""
    res = evaluate_per_query(query_feat, query_pid, query_cam, gallery_feat, gallery_pid, gallery_cam)
    return res.rank1, res.mAP


def evaluate_base_and_calibrated(
    query_feat: np.ndarray,
    query_pid: np.ndarray,
    query_cam: np.ndarray,
    gallery_feat: np.ndarray,
    gallery_pid: np.ndarray,
    gallery_cam: np.ndarray,
    calibrate: bool = True,
) -> dict[str, PerQueryResult]:
    """Evaluate raw features as ``base`` (``N(f)``) and, optionally, ``calib`` (``N(N(f) - mu_c)``).

    Both conditions are scored on the same queries, so their per-query AP / Rank-1
    can be compared with a paired bootstrap.
    """
    from .calibrate import calibrate_query_gallery  # local import: calibrate imports this module

    out = {"base": evaluate_per_query(l2_normalize(query_feat), query_pid, query_cam,
                                      l2_normalize(gallery_feat), gallery_pid, gallery_cam)}
    if calibrate:
        qc, gc = calibrate_query_gallery(query_feat, query_cam, gallery_feat, gallery_cam)
        out["calib"] = evaluate_per_query(qc, query_pid, query_cam, gc, gallery_pid, gallery_cam)
    return out
