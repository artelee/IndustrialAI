# adapted from reid-eval-bench (src/reidbench/fusion.py: camera_mean_subtract)
"""Retrieval-time camera calibration (training-free, numpy only).

Each feature is first L2-normalised, ``f = N(raw)``. Then, for every camera
``c`` the mean ``mu_c`` of the normalised features of that camera is subtracted
and the result is normalised again::

    f_hat = N(f - mu_c)

The camera means are computed **separately for the query set and for the
gallery set** (query camera c uses the mean of the queries from camera c, gallery
camera c the mean of the gallery images from camera c). No labels and no
training are used, only the camera id of each image.

Invariant: after the subtraction and *before* the final normalisation, the
per-camera mean of each set is exactly zero (:func:`center_by_camera`). The final
re-normalisation rescales each vector differently, so after it the per-camera
mean is only approximately zero.
"""
from __future__ import annotations

import numpy as np

from .evaluate import l2_normalize

__all__ = ["camera_means", "center_by_camera", "camera_mean_subtract", "calibrate_query_gallery"]


def camera_means(x: np.ndarray, cam: np.ndarray) -> dict[int, np.ndarray]:
    """Mean feature ``mu_c`` of every camera id in ``cam``."""
    return {int(c): x[cam == c].mean(0) for c in np.unique(cam)}


def center_by_camera(x: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """``x - mu_c`` for every item (camera means taken from ``x`` itself)."""
    y = x.copy()
    for c in np.unique(cam):
        y[cam == c] -= x[cam == c].mean(0)
    return y


def camera_mean_subtract(x: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Subtract each camera's mean feature from its items, then L2-normalise."""
    return l2_normalize(center_by_camera(x, cam))


def calibrate_query_gallery(
    query_feat: np.ndarray,
    query_cam: np.ndarray,
    gallery_feat: np.ndarray,
    gallery_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate raw query and gallery features independently.

    Returns ``(N(N(q) - mu_c^query), N(N(g) - mu_c^gallery))``.
    """
    q = camera_mean_subtract(l2_normalize(query_feat), query_cam)
    g = camera_mean_subtract(l2_normalize(gallery_feat), gallery_cam)
    return q, g
