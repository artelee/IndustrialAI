"""Body keypoints: optional DWPose extraction and neck/torso normalization.

Keypoints follow the 18-joint OpenPose body layout used by DWPose / controlnet_aux.
Coordinates are image-relative (x / width, y / height).

This module depends only on numpy. ``DWPoseExtractor`` imports ``controlnet_aux``,
``cv2`` and ``torch`` lazily, so normalization works without any pose model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NUM_JOINTS = 18
JOINT_NAMES: tuple[str, ...] = (
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
)
NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 1, 2, 5, 8, 11

#: Skeleton edges (OpenPose body layout), used for drawing.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)

DEFAULT_CONF_TH = 0.3


def visible_mask(scores: np.ndarray, conf_th: float = DEFAULT_CONF_TH) -> np.ndarray:
    """Boolean mask of joints whose confidence exceeds ``conf_th``."""
    return np.asarray(scores, dtype=np.float64) > conf_th


def normalize_pose(
    body: np.ndarray, scores: np.ndarray, conf_th: float = DEFAULT_CONF_TH
) -> tuple[np.ndarray, np.ndarray]:
    """Remove position and scale from an 18x2 skeleton.

    The neck becomes the origin and coordinates are divided by the torso length
    (neck -> right hip, or neck -> left hip if the right hip is missing). If no
    usable hip exists, the mean distance of visible joints to the neck is used.
    Invisible joints are set to 0.

    Returns:
        (normalized 18x2 array, visibility mask of shape (18,))
    """
    b = np.asarray(body, dtype=np.float64).copy()
    v = visible_mask(scores, conf_th)
    b = b - b[NECK]
    ref = 1.0
    for hip in (R_HIP, L_HIP):
        if v[hip]:
            ref = float(np.linalg.norm(b[hip]))
            break
    if ref < 1e-3:
        ref = float(np.linalg.norm(b[v], axis=1).mean()) + 1e-6 if v.any() else 1.0
    b = b / ref
    b[~v] = 0.0
    return b, v


def pose_vector(norm_body: np.ndarray) -> np.ndarray:
    """Flatten a normalized 18x2 skeleton into the 36-dim pose vector."""
    return np.asarray(norm_body, dtype=np.float64).reshape(-1)


class DWPoseExtractor:
    """Thin wrapper around controlnet_aux's DWPose (optional dependency).

    Weights are downloaded by controlnet_aux/mmpose on first use unless local
    checkpoint paths are given.
    """

    def __init__(
        self,
        device: str = "cpu",
        det_ckpt: str | None = None,
        pose_ckpt: str | None = None,
        detect_resolution: int = 512,
    ) -> None:
        try:
            from controlnet_aux.dwpose import DWposeDetector  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Pose extraction needs controlnet_aux (+ mmpose/mmdet/mmcv). "
                "See README 'Install'. Use --keypoints to analyze cached keypoints instead."
            ) from exc
        kwargs: dict[str, Any] = {"device": device}
        if det_ckpt:
            kwargs["det_ckpt"] = det_ckpt
        if pose_ckpt:
            kwargs["pose_ckpt"] = pose_ckpt
        self.detector = DWposeDetector(**kwargs)
        self.detect_resolution = detect_resolution

    def extract(self, image: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (body 18x2 in [0,1] coords, scores 18) of the first person, or (None, None)."""
        import cv2
        import torch
        from controlnet_aux.util import HWC3, resize_image

        img = cv2.cvtColor(np.array(image.convert("RGB"), np.uint8), cv2.COLOR_RGB2BGR)
        img = resize_image(HWC3(img), self.detect_resolution)
        h, w, _ = img.shape
        with torch.no_grad():
            cand, sub = self.detector.pose_estimation(img)
        if cand is None or len(cand) == 0:
            return None, None
        cand = cand.copy()
        cand[..., 0] /= w
        cand[..., 1] /= h
        return cand[0, :NUM_JOINTS].copy(), sub[0, :NUM_JOINTS].copy()
