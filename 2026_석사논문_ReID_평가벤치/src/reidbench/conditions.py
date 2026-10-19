"""Evaluation conditions: baseline, camera calibration, and extra-feature fusion.

One *experiment* = (backbone, train dataset, eval dataset) loaded from the
unified cache (see :mod:`reidbench.io`). For each experiment the following
gallery variants are evaluated, each with and without camera calibration:

====  =======================================================================
key   gallery
====  =======================================================================
B     baseline: normalised gallery features
S     fused with one *common* extra-feature set (same set for every query)
T     fused with a *per-camera* extra-feature set: a query from camera c is
      ranked against the gallery fused with camera c's set
====  =======================================================================

Calibration (``calibrate=True``) applies :func:`camera_mean_subtract` to the
queries (by query camera) and to the (fused) gallery (by gallery camera).

In the thesis experiments the extra sets are features of 8 images generated
per reference gallery image (common 8 poses for S, camera-specific
walking-direction-aligned poses for T), but the code is generic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fusion import (assign_by_identity, assign_by_similarity, camera_mean_subtract, fuse_extra,
                     reference_index, set_sums)
from .io import FeatureSplit, cache_path, load_set, load_split
from .metrics import l2_normalize, evaluate_per_query

__all__ = ["Experiment", "load_experiment", "run_condition", "run_all_conditions", "ROWS"]

#: (label, gallery key, calibrate) in table order
ROWS: list[tuple[str, str, bool]] = [
    ("Base", "B", False),
    ("+ Calib", "B", True),
    ("+ Fusion (common)", "S", False),
    ("+ Fusion (common) + Calib", "S", True),
    ("+ Fusion (per-camera)", "T", False),
    ("+ Fusion (per-camera) + Calib", "T", True),
]


@dataclass
class Experiment:
    """Loaded and pre-processed features of one (backbone, train, eval) triple."""

    query: FeatureSplit
    gallery: FeatureSplit
    cameras: list[int]
    sums: dict = field(default_factory=dict)      # "common" / camera id -> (P, D)
    k: int = 8                                    # extra features per reference
    assign: dict = field(default_factory=dict)    # "sim" / "oracle" -> (M,) reference index
    assign_accuracy: float = float("nan")         # % of pid>0 gallery items assigned to own pid
    gallery_norm: np.ndarray | None = None

    @property
    def has_sets(self) -> bool:
        return "common" in self.sums


def load_experiment(
    cache_dir: str | Path,
    backbone: str,
    train: str,
    eval_: str,
    cameras: list[int],
    common_set: str | None = None,
    camera_set: str | None = None,
) -> Experiment:
    """Load query/gallery and (optionally) extra-feature sets of one experiment.

    Args:
        common_set: set name of the common set, e.g. ``mkt_std`` (None = no fusion).
        camera_set: format string of per-camera sets, e.g. ``mkt_dm_c{cam}``.
    """
    q = load_split(cache_path(cache_dir, backbone, train, eval_, "query"))
    g = load_split(cache_path(cache_dir, backbone, train, eval_, "gal"))
    exp = Experiment(query=q, gallery=g, cameras=list(cameras))
    exp.gallery_norm = l2_normalize(g.feat)
    if common_set is None:
        return exp
    sets = {"common": load_set(cache_path(cache_dir, backbone, train, eval_, common_set))}
    if camera_set is not None:
        for c in cameras:
            sets[c] = load_set(cache_path(cache_dir, backbone, train, eval_, camera_set.format(cam=c)))
    ref_name = sets["common"].name
    for key, s in sets.items():
        if not np.array_equal(s.name, ref_name):
            raise ValueError(f"set {key!r} has a different reference order than the common set")
    exp.k = sets["common"].feat.shape[1]
    exp.sums = {key: set_sums(s.feat) for key, s in sets.items()}
    ref_idx = reference_index(g.name, ref_name)
    ref_pid = g.pid[ref_idx]
    g_cal = camera_mean_subtract(exp.gallery_norm, g.cam)
    exp.assign = {"sim": assign_by_similarity(g_cal, ref_idx), "oracle": assign_by_identity(g.pid, ref_pid)}
    real = g.pid > 0
    exp.assign_accuracy = float(np.mean(ref_pid[exp.assign["sim"][real]] == g.pid[real]) * 100)
    return exp


def run_condition(exp: Experiment, key: str, calibrate: bool, eta: float = 0.5,
                  assign: str = "sim", query_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-query ``(r1, ap)`` arrays (in ``query_idx`` order) for one condition."""
    q, g = exp.query, exp.gallery
    idx = np.arange(len(q)) if query_idx is None else np.asarray(query_idx)
    qf = l2_normalize(q.feat)
    if calibrate:
        qf = camera_mean_subtract(qf, q.cam)

    def post(x: np.ndarray) -> np.ndarray:
        return camera_mean_subtract(x, g.cam) if calibrate else x

    def ev(gal: np.ndarray, sub: np.ndarray):
        res = evaluate_per_query(qf, q.pid, q.cam, gal, g.pid, g.cam, sub)
        return res.r1, res.ap

    if key == "B":
        return ev(post(exp.gallery_norm), idx)
    if not exp.has_sets:
        raise ValueError("fusion conditions need extra-feature sets")
    m = exp.assign[assign]
    if key == "S":
        return ev(post(fuse_extra(exp.gallery_norm, exp.sums["common"], m, eta, exp.k)), idx)
    if key == "T":
        r1 = np.zeros(len(idx))
        ap = np.zeros(len(idx))
        for c in exp.cameras:
            k = np.where(q.cam[idx] == c)[0]
            if len(k):
                r1[k], ap[k] = ev(post(fuse_extra(exp.gallery_norm, exp.sums[c], m, eta, exp.k)), idx[k])
        return r1, ap
    raise ValueError(f"unknown condition key {key!r}")


def run_all_conditions(exp: Experiment, eta: float = 0.5, assign: str = "sim",
                       keys: tuple[str, ...] = ("B", "S", "T")) -> dict:
    """``{(key, calibrate): (r1, ap)}`` for every key and calibration flag."""
    out = {}
    for cal in (False, True):
        for key in keys:
            out[(key, cal)] = run_condition(exp, key, cal, eta, assign)
    return out
