"""Cut IPG output panels into individual generated images.

IPG saves one horizontal panel per reference image with ``2 * P + 1`` equal
columns (``P`` = number of poses, 8 by default)::

    | ref | pose_0 | gen_0 | pose_1 | gen_1 | ... | pose_{P-1} | gen_{P-1} |

Column width ``cw = panel_width // (2P + 1)`` (128 px in the stock IPG code).
The k-th generated image is the box ``((2k+2)*cw, 0, (2k+3)*cw, H)``.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

Box = tuple[int, int, int, int]

DEFAULT_NUM_POSES = 8


def num_columns(num_poses: int = DEFAULT_NUM_POSES) -> int:
    """Number of equal-width columns in a panel (reference + pose/gen pairs)."""
    if num_poses < 1:
        raise ValueError("num_poses must be >= 1")
    return 2 * num_poses + 1


def column_width(panel_width: int, num_poses: int = DEFAULT_NUM_POSES) -> int:
    """Width of one panel column."""
    cw = panel_width // num_columns(num_poses)
    if cw < 1:
        raise ValueError(f"panel width {panel_width} too small for {num_poses} poses")
    return cw


def generated_boxes(panel_width: int, panel_height: int, num_poses: int = DEFAULT_NUM_POSES) -> list[Box]:
    """Crop boxes ``(left, top, right, bottom)`` of the generated images."""
    cw = column_width(panel_width, num_poses)
    return [((2 * k + 2) * cw, 0, (2 * k + 3) * cw, panel_height) for k in range(num_poses)]


def pose_boxes(panel_width: int, panel_height: int, num_poses: int = DEFAULT_NUM_POSES) -> list[Box]:
    """Crop boxes of the pose (skeleton) conditioning images."""
    cw = column_width(panel_width, num_poses)
    return [((2 * k + 1) * cw, 0, (2 * k + 2) * cw, panel_height) for k in range(num_poses)]


def reference_box(panel_width: int, panel_height: int, num_poses: int = DEFAULT_NUM_POSES) -> Box:
    """Crop box of the reference image (first column)."""
    return (0, 0, column_width(panel_width, num_poses), panel_height)


def is_panel_width(panel_width: int, num_poses: int = DEFAULT_NUM_POSES) -> bool:
    """True if the width splits exactly into ``2P + 1`` columns."""
    return panel_width > 0 and panel_width % num_columns(num_poses) == 0


def crop_generated(panel: Image.Image, num_poses: int = DEFAULT_NUM_POSES) -> list[Image.Image]:
    """Return the ``num_poses`` generated images contained in ``panel``."""
    w, h = panel.size
    return [panel.crop(b) for b in generated_boxes(w, h, num_poses)]


def crop_panel_file(
    panel_path: str | Path,
    out_dir: str | Path,
    num_poses: int = DEFAULT_NUM_POSES,
    suffix: str = "_g{k}",
    overwrite: bool = False,
) -> list[Path]:
    """Crop one panel file and save ``<stem><suffix><ext>`` per pose into ``out_dir``.

    Existing outputs are kept unless ``overwrite`` is set. Returns the paths of
    all (new or existing) crops.
    """
    panel_path = Path(panel_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"{panel_path.stem}{suffix.format(k=k)}{panel_path.suffix}" for k in range(num_poses)]
    if not overwrite and all(t.exists() for t in targets):
        return targets
    with Image.open(panel_path) as im:
        crops = crop_generated(im.convert("RGB"), num_poses)
    for crop, target in zip(crops, targets):
        if overwrite or not target.exists():
            if target.suffix.lower() in {".jpg", ".jpeg"}:
                crop.save(target, quality=95)
            else:
                crop.save(target)
    return targets
