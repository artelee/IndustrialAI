"""campose: camera-aware selection and reference-matched rendering of generation poses."""
from .allocate import allocate_poses, allocation_string, direction_counts, direction_fractions
from .render import REFERENCE_LAYOUT, Layout, place, render_rgb, save_like_reference
from .select import kmeans_representatives, select_camera_poses, select_representatives

__all__ = [
    "allocate_poses", "allocation_string", "direction_counts", "direction_fractions",
    "REFERENCE_LAYOUT", "Layout", "place", "render_rgb", "save_like_reference",
    "kmeans_representatives", "select_camera_poses", "select_representatives",
]
__version__ = "0.1.0"
