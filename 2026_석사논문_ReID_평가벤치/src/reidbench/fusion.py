"""Feature post-processing used by the evaluation conditions (numpy only).

* :func:`camera_mean_subtract` - retrieval-time camera calibration: subtract the
  per-camera mean feature, then re-normalise.
* :func:`fuse_extra` - fuse gallery features with an extra feature set
  (e.g. features of K generated images per reference image)::

      g' = N( N(g) + (eta / K) * sum_k N(e_k) )

  where ``N`` is L2 normalisation and ``e_k`` are the K extra features of the
  reference image assigned to gallery item ``g``. Items with no assignment
  (index -1) keep ``N(g)``. Nothing here is specific to generated images; any
  per-reference feature set works.
* :func:`assign_by_similarity` / :func:`assign_by_identity` - decide which
  reference image's set each gallery item receives (by nearest reference in
  camera-calibrated feature space, or by ground-truth pid as an upper bound).
"""
from __future__ import annotations

import numpy as np

from .metrics import l2_normalize

__all__ = [
    "camera_mean_subtract",
    "set_sums",
    "fuse_extra",
    "reference_index",
    "assign_by_similarity",
    "assign_by_identity",
]


def camera_mean_subtract(x: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Subtract each camera's mean feature from its items, then L2-normalise."""
    y = x.copy()
    for c in np.unique(cam):
        y[cam == c] -= x[cam == c].mean(0)
    return l2_normalize(y)


def set_sums(set_feat: np.ndarray) -> np.ndarray:
    """``(P, K, D)`` raw extra features -> ``(P, D)`` sum of normalised features."""
    return l2_normalize(set_feat).sum(1)


def fuse_extra(gallery_norm: np.ndarray, sums: np.ndarray, assign: np.ndarray,
               eta: float, k: int = 8) -> np.ndarray:
    """``N(N(g) + (eta / k) * sums[assign])`` for assigned items (assign >= 0)."""
    sel = assign >= 0
    out = gallery_norm.copy()
    out[sel] = gallery_norm[sel] + (eta / k) * sums[assign[sel]]
    return l2_normalize(out)


def reference_index(gallery_name: np.ndarray, ref_name: np.ndarray) -> np.ndarray:
    """Gallery row index of every reference image name."""
    pos = {str(n): i for i, n in enumerate(gallery_name)}
    try:
        return np.array([pos[str(n)] for n in ref_name])
    except KeyError as e:
        raise KeyError(f"reference image {e} not found in gallery") from None


def assign_by_similarity(gallery_calibrated: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    """Each gallery item -> most similar reference (row index into ``ref_idx``)."""
    return (gallery_calibrated @ gallery_calibrated[ref_idx].T).argmax(1)


def assign_by_identity(gallery_pid: np.ndarray, ref_pid: np.ndarray) -> np.ndarray:
    """Each gallery item -> reference with the same pid, or -1 (oracle upper bound)."""
    pid2r = {int(p): j for j, p in enumerate(ref_pid)}
    return np.array([pid2r.get(int(p), -1) for p in gallery_pid])
