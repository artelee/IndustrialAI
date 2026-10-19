"""Gallery file-name parsing, per-camera sampling and keypoint JSON I/O.

File-name parsers are copied from walking-direction-analyzer (``walkdir.data``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

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


def list_camera_images(gallery: str | Path, parser: str, cam: int) -> list[str]:
    """Sorted gallery images of one camera, excluding junk/distractor ids (-1, 0)."""
    parse = PARSERS[parser]
    out = []
    for f in sorted(str(p) for p in Path(gallery).glob("*.jpg")):
        pid, c = parse(f)
        if c == cam and pid not in (-1, 0):
            out.append(f)
    return out


def sample_cameras(
    gallery: str | Path, parser: str, cams: Sequence[int], samples: int, seed: int = 0
) -> dict[int, list[str]]:
    """Up to ``samples`` random images per camera.

    One ``RandomState(seed)`` is shared across cameras in the given order, so the
    sample of a camera depends on the cameras before it (as in the original run).
    """
    rng = np.random.RandomState(seed)
    out: dict[int, list[str]] = {}
    for cam in cams:
        imgs = list_camera_images(gallery, parser, cam)
        if not imgs:
            out[cam] = []
            continue
        idx = rng.choice(len(imgs), min(samples, len(imgs)), replace=False)
        out[cam] = [imgs[i] for i in idx]
    return out


def save_keypoints_json(path: str | Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Save ``{"c<cam>": [{"src", "body", "score", "image_size"}, ...]}``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_keypoints_json(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Inverse of :func:`save_keypoints_json`."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detection_size(width: int, height: int, resolution: int = 512) -> tuple[int, int]:
    """(w, h) of an image after controlnet_aux ``resize_image(img, resolution)``.

    DWPose keypoints are divided by this size, so it is the size needed to turn
    them back into pixels with the correct aspect ratio when rendering.
    """
    k = float(resolution) / min(float(height), float(width))
    h = int(np.round(float(height) * k / 64.0)) * 64
    w = int(np.round(float(width) * k / 64.0)) * 64
    return w, h
