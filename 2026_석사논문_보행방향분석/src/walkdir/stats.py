"""Per-camera statistics on directions and pose vectors."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .direction import DIRECTIONS


def direction_distribution(
    labels: Iterable[str], directions: Sequence[str] = DIRECTIONS
) -> dict[str, float]:
    """Fraction of each direction among labels (labels outside ``directions`` ignored)."""
    lab = [x for x in labels if x in directions]
    n = len(lab)
    return {d: (sum(1 for x in lab if x == d) / n if n else 0.0) for d in directions}


def per_camera_distribution(
    cams: Sequence[int], labels: Sequence[str], directions: Sequence[str] = DIRECTIONS
) -> dict[int, dict[str, float]]:
    """Direction distribution for every camera id."""
    cams_a = np.asarray(cams)
    labels_a = np.asarray(labels, dtype=object)
    return {
        int(c): direction_distribution(labels_a[cams_a == c].tolist(), directions)
        for c in sorted(set(cams_a.tolist()))
    }


def intra_inter_variation(X: np.ndarray, cams: Sequence[int]) -> dict:
    """Intra- vs. inter-camera pose variation in the 36-dim pose space.

    * intra (per camera): mean Euclidean distance of the camera's poses to the
      camera mean; the dataset value is the average over cameras.
    * inter (per camera): mean distance of the camera mean to the other camera
      means; the dataset value is the average over all ordered camera pairs.
    """
    X = np.asarray(X, dtype=np.float64)
    cams_a = np.asarray(cams)
    ids = sorted(set(cams_a.tolist()))
    if len(ids) < 2:
        raise ValueError("need at least two cameras")
    means = np.stack([X[cams_a == c].mean(0) for c in ids])
    intra_c = [float(np.linalg.norm(X[cams_a == c] - m, axis=1).mean()) for c, m in zip(ids, means)]
    dm = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
    off = ~np.eye(len(ids), dtype=bool)
    inter_c = [float(dm[i][off[i]].mean()) for i in range(len(ids))]
    intra = float(np.mean(intra_c))
    inter = float(dm[off].mean())
    return {
        "cameras": [int(c) for c in ids],
        "intra": intra,
        "inter": inter,
        "ratio": intra / inter if inter > 0 else float("inf"),
        "intra_per_camera": dict(zip([int(c) for c in ids], intra_c)),
        "inter_per_camera": dict(zip([int(c) for c in ids], inter_c)),
    }


def total_variation_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """TVD = 0.5 * sum |p - q| between two direction distributions."""
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def coverage(X: np.ndarray, P: np.ndarray) -> float:
    """Mean distance from every pose in ``X`` to its nearest reference pose in ``P``."""
    X = np.asarray(X, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    d = np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2)
    return float(d.min(1).mean())
