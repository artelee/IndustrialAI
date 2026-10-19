"""Walking-direction classification from a normalized skeleton.

Rule:

* shoulder width ``|x(r_shoulder) - x(l_shoulder)|`` (in torso lengths)
  ``< q33`` -> ``side``; ``q33 <= w < q66`` -> ``oblique``; ``>= q66`` -> front/back.
* front vs. back from the left/right shoulder order: facing the camera, the
  person's right shoulder appears on the image left, so ``dx = x_r - x_l < 0``.
  (Face-joint counts are not used: DWPose often detects eyes on people seen
  from behind.)

``q33``/``q66`` are the 1/3 and 2/3 quantiles of shoulder widths of a dataset's
gallery sample. Depends on numpy only.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

DIRECTIONS: tuple[str, ...] = ("front", "back", "oblique", "side")
UNKNOWN = "unknown"
R_SHOULDER, L_SHOULDER = 2, 5

#: dx * FRONT_SIGN > 0 means "front" (person's right shoulder on image left).
FRONT_SIGN = -1

#: Shoulder-width quantile thresholds (q33, q66) measured on each gallery.
PRESET_THRESHOLDS: dict[str, tuple[float, float]] = {
    "market": (1.076, 1.351),
    "duke": (0.865, 1.266),
    "msmt": (1.164, 1.458),
}


def shoulder_dx(norm_body: np.ndarray, visible: np.ndarray) -> float | None:
    """Signed shoulder offset ``x_r - x_l`` or ``None`` if a shoulder is missing."""
    if not (visible[R_SHOULDER] and visible[L_SHOULDER]):
        return None
    return float(norm_body[R_SHOULDER, 0] - norm_body[L_SHOULDER, 0])


def classify_direction(
    norm_body: np.ndarray,
    visible: np.ndarray,
    q33: float,
    q66: float,
    front_sign: int = FRONT_SIGN,
) -> str:
    """Classify a normalized skeleton as front / back / oblique / side / unknown."""
    dx = shoulder_dx(norm_body, visible)
    if dx is None:
        return UNKNOWN
    return direction_from_dx(dx, q33, q66, front_sign)


def direction_from_dx(dx: float, q33: float, q66: float, front_sign: int = FRONT_SIGN) -> str:
    """Direction from a signed shoulder offset (lower bounds inclusive)."""
    width = abs(dx)
    if width < q33:
        return "side"
    if width < q66:
        return "oblique"
    return "front" if dx * front_sign > 0 else "back"


def compute_thresholds(widths: Iterable[float]) -> tuple[float, float]:
    """1/3 and 2/3 quantiles of shoulder widths -> (q33, q66)."""
    w = np.asarray(list(widths), dtype=np.float64)
    if w.size == 0:
        raise ValueError("no shoulder widths to compute thresholds from")
    q33, q66 = np.percentile(w, [100 / 3, 200 / 3])
    return float(q33), float(q66)
