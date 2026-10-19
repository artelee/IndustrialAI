"""Gallery file-name parsers, camera-stratified sampling and keypoint cache I/O."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def parse_market(path: str) -> tuple[int, int]:
    """Market-1501 ``0002_c1s1_000451_03.jpg`` -> (pid, camera)."""
    p = os.path.basename(path).split("_")
    return int(p[0]), int(p[1][1:2])


def parse_duke(path: str) -> tuple[int, int]:
    """DukeMTMC-reID ``0002_c1_f0044158.jpg`` -> (pid, camera)."""
    p = os.path.basename(path).split("_")
    return int(p[0]), int(p[1][1:])


#: MSMT17 converted to Market-style names (``0000_c14_...jpg``) shares Duke's rule.
parse_msmt = parse_duke

PARSERS: dict[str, Callable[[str], tuple[int, int]]] = {
    "market": parse_market,
    "duke": parse_duke,
    "msmt": parse_msmt,
}


def list_gallery(gallery: str | Path, parser: str) -> list[tuple[str, int, int]]:
    """Sorted (path, pid, camera) of gallery images, excluding junk/distractor ids (-1, 0)."""
    parse = PARSERS[parser]
    out = []
    for f in sorted(Path(gallery).glob("*.jpg")):
        pid, cam = parse(str(f))
        if pid in (-1, 0):
            continue
        out.append((str(f), pid, cam))
    return out


def sample_per_camera(
    items: Sequence[tuple[str, int, int]], samples: int, seed: int = 0
) -> list[tuple[str, int, int]]:
    """Up to ``samples`` random images per camera (one RNG shared across cameras, camera order)."""
    rng = np.random.RandomState(seed)
    out: list[tuple[str, int, int]] = []
    for cam in sorted({c for _, _, c in items}):
        cam_items = [it for it in items if it[2] == cam]
        idx = rng.choice(len(cam_items), min(samples, len(cam_items)), replace=False)
        out.extend(cam_items[i] for i in idx)
    return out


def save_keypoints(
    path: str | Path,
    files: Sequence[str],
    pids: Sequence[int],
    cams: Sequence[int],
    bodies: np.ndarray,
    scores: np.ndarray,
    detected: np.ndarray,
) -> None:
    """Save raw keypoints. ``files`` are stored as base names only."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        files=np.array([os.path.basename(f) for f in files]),
        pids=np.asarray(pids, np.int64),
        cams=np.asarray(cams, np.int64),
        bodies=np.asarray(bodies, np.float32),
        scores=np.asarray(scores, np.float32),
        detected=np.asarray(detected, bool),
    )


def load_keypoints(path: str | Path) -> dict[str, np.ndarray]:
    """Load a keypoint cache written by :func:`save_keypoints`."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
