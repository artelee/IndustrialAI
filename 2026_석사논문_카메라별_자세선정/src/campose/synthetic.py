# make_skeleton is copied from walking-direction-analyzer (src/walkdir/synthetic.py).
"""Synthetic 18-joint skeletons for tests and the CPU demo (no pose model, no images)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .posecore.direction import DIRECTIONS

TORSO = 0.28  # neck -> hip height in image-relative y units
NECK_XY = (0.5, 0.15)
#: Typical shoulder width (in torso lengths) used per direction.
DEFAULT_WIDTH = {"front": 1.6, "back": 1.6, "oblique": 1.2, "side": 0.5}
def make_skeleton(
    direction: str, shoulder_width: float | None = None, stride: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (body 18x2 image-relative coords, scores 18) for a walking person.

    ``shoulder_width`` is in torso lengths; ``stride`` moves the legs apart in x
    (side views). Back views hide the nose; the eyes keep a low score, as DWPose
    often still reports them on people seen from behind.
    """
    if direction not in DIRECTIONS:
        raise ValueError(direction)
    sw = DEFAULT_WIDTH[direction] if shoulder_width is None else shoulder_width
    half = sw * TORSO / 2
    hip = max(0.25 * half, 0.012)
    leg = 0.6 * hip
    # x offsets for (right, left) joints of a person facing the camera:
    # the person's right side appears on the image left (negative x).
    j = np.zeros((18, 2))
    j[0] = (0.0, -0.10)                      # nose
    j[1] = (0.0, 0.0)                        # neck
    j[2], j[5] = (-half, 0.0), (half, 0.0)   # shoulders
    j[3], j[6] = (-half - 0.02, 0.15), (half + 0.02, 0.15)
    j[4], j[7] = (-half - 0.03, 0.28), (half + 0.03, 0.28)
    j[8], j[11] = (-hip, TORSO), (hip, TORSO)
    j[9], j[12] = (-leg - stride / 2, TORSO + 0.21), (leg + stride / 2, TORSO + 0.21)
    j[10], j[13] = (-leg - stride, TORSO + 0.42), (leg + stride, TORSO + 0.42)
    j[14], j[15] = (-0.025, -0.12), (0.025, -0.12)
    j[16], j[17] = (-0.05, -0.11), (0.05, -0.11)
    scores = np.full(18, 0.9)
    if direction == "back":
        j[:, 0] *= -1                         # mirrored: right side on image right
        scores[0] = 0.0
        scores[[14, 15]] = 0.5
        j[[16, 17], 1] -= 0.005
    if direction in ("oblique", "side"):
        scores[16] = 0.0                      # far ear hidden
        if direction == "side":
            scores[[14]] = 0.0
            j[0, 0] += 0.04
    j[:, 0] += NECK_XY[0]
    j[:, 1] += NECK_XY[1]
    return j, scores


def make_camera_samples(
    mix: Mapping[str, float],
    n: int = 300,
    seed: int = 0,
    noise: float = 0.01,
    cam: int = 1,
    image_size: tuple[int, int] = (64, 128),
) -> list[dict[str, Any]]:
    """Keypoint samples of one synthetic camera whose direction shares follow ``mix``.

    Each sample has the keys used by ``campose.select``: ``src``, ``body``,
    ``score`` and ``image_size``.
    """
    rng = np.random.RandomState(seed)
    dirs = [d for d in mix if mix[d] > 0]
    p = np.array([mix[d] for d in dirs], float)
    p = p / p.sum()
    out = []
    for i in range(n):
        d = dirs[rng.choice(len(dirs), p=p)]
        sw = DEFAULT_WIDTH[d] + rng.uniform(-0.15, 0.15)
        stride = rng.uniform(0.0, 0.12) if d in ("side", "oblique") else rng.uniform(0.0, 0.03)
        b, s = make_skeleton(d, sw, stride)
        b = b + rng.normal(0, noise, b.shape)
        out.append(dict(
            src=f"{i + 1:04d}_c{cam}_synthetic_{i:05d}.jpg",
            body=b.tolist(), score=s.tolist(), image_size=list(image_size),
        ))
    return out
