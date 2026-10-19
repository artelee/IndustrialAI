"""Unified feature-cache format and loaders.

Every backbone (CLIP-ReID, PersonViT, ...) stores its features in the same
layout, one ``.npz`` per split::

    <cache_dir>/<backbone>_<train>_<eval>_query.npz   feat (N, D) raw, pid (N,), cam (N,), name (N,)
    <cache_dir>/<backbone>_<train>_<eval>_gal.npz     feat (M, D) raw, pid (M,), cam (M,), name (M,)
    <cache_dir>/<backbone>_<train>_<eval>_<set>.npz   feat (P, K, D) raw, name (P,)

* ``<train>`` is the training dataset of the model (``market``, ``duke``, or
  ``none`` for a model without Re-ID training); ``<eval>`` the evaluation dataset.
* Features are stored **raw** (not normalised); normalisation happens at evaluation.
* Query split excludes junk (pid -1) and distractors (pid 0); gallery excludes junk only.
* A ``<set>`` file holds K extra (e.g. generated) features for each of P
  reference gallery images, identified by ``name`` (file names that must exist
  in the gallery split).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FeatureSplit",
    "FeatureSet",
    "cache_tag",
    "cache_path",
    "load_split",
    "save_split",
    "load_set",
    "save_set",
    "load_tags",
]


@dataclass
class FeatureSplit:
    """Query or gallery features with labels."""

    feat: np.ndarray
    pid: np.ndarray
    cam: np.ndarray
    name: np.ndarray

    def __len__(self) -> int:
        return len(self.pid)


@dataclass
class FeatureSet:
    """K extra features per reference image: ``feat`` (P, K, D), ``name`` (P,)."""

    feat: np.ndarray
    name: np.ndarray


def cache_tag(backbone: str, train: str, eval_: str) -> str:
    """``<backbone>_<train>_<eval>``."""
    return f"{backbone}_{train}_{eval_}"


def cache_path(cache_dir: str | Path, backbone: str, train: str, eval_: str, part: str) -> Path:
    """Path of one cache file; ``part`` is ``query``, ``gal`` or a set name."""
    return Path(cache_dir) / f"{cache_tag(backbone, train, eval_)}_{part}.npz"


def load_split(path: str | Path) -> FeatureSplit:
    """Load a query/gallery ``.npz`` (keys feat, pid, cam, name)."""
    with np.load(path) as z:
        missing = {"feat", "pid", "cam", "name"} - set(z.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")
        return FeatureSplit(feat=z["feat"], pid=z["pid"], cam=z["cam"], name=z["name"])


def save_split(path: str | Path, split: FeatureSplit) -> None:
    """Save a query/gallery split in the unified format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feat=np.asarray(split.feat, np.float32), pid=np.asarray(split.pid),
             cam=np.asarray(split.cam), name=np.asarray(split.name))


def load_set(path: str | Path) -> FeatureSet:
    """Load an extra-feature set ``.npz`` (keys feat (P, K, D), name (P,))."""
    with np.load(path) as z:
        feat = z["feat"]
        if feat.ndim != 3:
            raise ValueError(f"{path}: expected feat of shape (P, K, D), got {feat.shape}")
        return FeatureSet(feat=feat, name=z["name"])


def save_set(path: str | Path, fset: FeatureSet) -> None:
    """Save an extra-feature set in the unified format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feat=np.asarray(fset.feat, np.float32), name=np.asarray(fset.name))


def load_tags(path: str | Path, names: np.ndarray, default: str = "unknown") -> np.ndarray:
    """Map image names to tags from a ``{name: tag}`` JSON (missing -> ``default``)."""
    with open(path, encoding="utf-8") as f:
        tags = json.load(f)
    return np.array([tags.get(str(n), default) for n in names])
