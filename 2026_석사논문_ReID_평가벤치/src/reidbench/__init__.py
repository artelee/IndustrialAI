"""reidbench: Re-ID evaluation bench (unified feature cache, metrics, bootstrap CIs, breakdowns)."""
from .bootstrap import format_diff, paired_bootstrap_ci, significance_star
from .breakdown import paired_by_tag, per_camera, weighted_mean
from .metrics import average_precision, evaluate, evaluate_per_query, l2_normalize

__all__ = [
    "average_precision", "evaluate", "evaluate_per_query", "l2_normalize",
    "paired_bootstrap_ci", "significance_star", "format_diff",
    "per_camera", "weighted_mean", "paired_by_tag",
]
__version__ = "0.1.0"
