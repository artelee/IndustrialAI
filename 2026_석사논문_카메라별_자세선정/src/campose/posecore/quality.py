# Copied from walking-direction-analyzer (src/walkdir/quality.py); kept identical so this
# repository installs and runs on its own.
"""Pose quality filter with explicit rejection reasons.

Two modes:

* ``"strict"`` - full filter used for direction statistics: core joints,
  ankle/head presence, skeleton height and limb-length sanity.
* ``"core"`` - only the core-joint check, used for the intra/inter-camera
  pose-variation analysis.

Rejection reasons:

========== ==============================================================
nodet      pose estimator found no person (assigned by the caller)
core       fewer than ``min_visible`` joints, or a core joint missing
           (neck, both shoulders, both hips)
halfbody   no ankle visible, or no head joint (nose / eyes) visible
small      skeleton height (image fraction) below ``min_height``
limb       an arm/leg bone longer than ``limb_ratio`` x torso length
unknown    shoulders not both visible -> direction undecidable (caller)
========== ==============================================================

Depends on numpy and ``pose.normalize_pose`` only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import DEFAULT_CONF_TH, normalize_pose, visible_mask

REASONS: tuple[str, ...] = ("nodet", "core", "halfbody", "small", "limb", "unknown")

CORE_JOINTS: tuple[int, ...] = (1, 2, 5, 8, 11)  # neck, shoulders, hips
ANKLES: tuple[int, ...] = (10, 13)
HEAD: tuple[int, ...] = (0, 14, 15)  # nose, eyes
LIMBS: tuple[tuple[int, int], ...] = (
    (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13),
)


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds of the quality filter."""

    mode: str = "strict"  # "strict" or "core"
    conf_th: float = DEFAULT_CONF_TH
    min_visible: int = 12
    min_height: float = 0.55
    limb_ratio: float = 1.6


def filter_reason(
    body: np.ndarray, scores: np.ndarray, cfg: QualityConfig = QualityConfig()
) -> str | None:
    """Return the first failing reason for a raw skeleton, or ``None`` if it passes."""
    _, reason = filter_and_normalize(body, scores, cfg)
    return reason


def filter_and_normalize(
    body: np.ndarray | None, scores: np.ndarray | None, cfg: QualityConfig = QualityConfig()
) -> tuple[tuple[np.ndarray, np.ndarray] | None, str | None]:
    """Apply the quality filter and normalize.

    Returns:
        ``((norm_body, visible), None)`` on success, or ``(None, reason)``.
    """
    if body is None or scores is None:
        return None, "nodet"
    if cfg.mode not in ("strict", "core"):
        raise ValueError(f"unknown quality mode: {cfg.mode}")
    body = np.asarray(body, dtype=np.float64)
    v = visible_mask(scores, cfg.conf_th)
    if v.sum() < cfg.min_visible or not v[list(CORE_JOINTS)].all():
        return None, "core"
    if cfg.mode == "strict":
        if not v[list(ANKLES)].any() or not v[list(HEAD)].any():
            return None, "halfbody"
        ys = body[v, 1]
        if ys.max() - ys.min() < cfg.min_height:
            return None, "small"
    b, v = normalize_pose(body, scores, cfg.conf_th)
    if cfg.mode == "strict":
        for a, c in LIMBS:
            if v[a] and v[c] and np.linalg.norm(b[a] - b[c]) > cfg.limb_ratio:
                return None, "limb"
    return (b, v), None


def empty_reason_counts() -> dict[str, int]:
    """Zero-initialized counter over all rejection reasons."""
    return {r: 0 for r in REASONS}
