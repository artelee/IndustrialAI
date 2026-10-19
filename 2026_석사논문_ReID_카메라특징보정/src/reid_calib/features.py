# adapted from reid-eval-bench (src/reidbench/io.py)
"""Feature-cache format (numpy only).

One ``.npz`` per split, raw (un-normalised) features::

    <features_dir>/<prefix>_<train>_<eval>_query.npz   feat (N, D), pid (N,), cam (N,), name (N,)
    <features_dir>/<prefix>_<train>_<eval>_gal.npz     feat (M, D), pid (M,), cam (M,), name (M,)

``<prefix>`` is the backbone's cache prefix (``clip`` for CLIP-ReID, ``pvit`` for
PersonViT), ``<train>`` the training dataset of the model and ``<eval>`` the
evaluation dataset. Queries exclude junk (pid -1) and distractors (pid 0); the
gallery excludes junk only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FeatureSplit", "cache_path", "load_split", "save_split"]


@dataclass
class FeatureSplit:
    """Query or gallery features with labels."""

    feat: np.ndarray
    pid: np.ndarray
    cam: np.ndarray
    name: np.ndarray

    def __len__(self) -> int:
        return len(self.pid)


def cache_path(features_dir: str | Path, prefix: str, train: str, eval_: str, part: str) -> Path:
    """Path of one cache file; ``part`` is ``query`` or ``gal``."""
    return Path(features_dir) / f"{prefix}_{train}_{eval_}_{part}.npz"


def load_split(path: str | Path) -> FeatureSplit:
    """Load a query/gallery ``.npz`` (keys feat, pid, cam, name)."""
    with np.load(path) as z:
        missing = {"feat", "pid", "cam", "name"} - set(z.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")
        return FeatureSplit(feat=z["feat"], pid=z["pid"], cam=z["cam"], name=z["name"])


def save_split(path: str | Path, split: FeatureSplit) -> None:
    """Save a query/gallery split (features as float32)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feat=np.asarray(split.feat, np.float32), pid=np.asarray(split.pid),
             cam=np.asarray(split.cam), name=np.asarray(split.name))
