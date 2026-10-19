"""Pose normalization, quality filter and walking-direction rule.

Copied from walking-direction-analyzer (``src/walkdir``) so that this package is
self-contained.
"""
from .direction import (
    DIRECTIONS,
    PRESET_THRESHOLDS,
    UNKNOWN,
    classify_direction,
    compute_thresholds,
    shoulder_dx,
)
from .pose import (
    JOINT_NAMES,
    NUM_JOINTS,
    SKELETON_EDGES,
    DWPoseExtractor,
    normalize_pose,
    pose_vector,
    visible_mask,
)
from .quality import REASONS, QualityConfig, empty_reason_counts, filter_and_normalize

__all__ = [
    "DIRECTIONS", "PRESET_THRESHOLDS", "UNKNOWN", "classify_direction", "compute_thresholds",
    "shoulder_dx", "JOINT_NAMES", "NUM_JOINTS", "SKELETON_EDGES", "DWPoseExtractor",
    "normalize_pose", "pose_vector", "visible_mask", "REASONS", "QualityConfig",
    "empty_reason_counts", "filter_and_normalize",
]
