"""Side-by-side check figure: common reference poses vs. camera-specific poses.

Only skeleton images are shown (no person images). Each row is one pose set
(e.g. the common 8 poses, then one row per camera) with the direction of every
pose written under it and the allocation (front/back/oblique/side) in the row title.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .allocate import allocation_string, direction_counts


@dataclass
class PoseRow:
    """One row of the comparison figure."""

    title: str
    images: list[np.ndarray]      # RGB arrays (display colors)
    directions: list[str]


def row_from_files(title: str, paths: Sequence[str | Path], directions: Sequence[str]) -> PoseRow:
    """Build a row from pose images stored like the reference poses (BGR-swapped files)."""
    from .render import load_display_rgb

    if len(paths) != len(directions):
        raise ValueError("paths and directions must have the same length")
    return PoseRow(title, [load_display_rgb(p) for p in paths], list(directions))


def make_comparison_figure(
    rows: Sequence[PoseRow],
    out_path: str | Path | None = None,
    cell_w: float = 0.62,
    dpi: int = 160,
):
    """Draw rows of pose images; saves to ``out_path`` if given and returns the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = max(len(r.images) for r in rows)
    gap, title_h, label_h, margin = 0.06, 0.26, 0.2, 0.08   # inches
    img_h = 2 * cell_w
    row_h = title_h + img_h + label_h
    fig_w = 2 * margin + ncol * cell_w + (ncol - 1) * gap
    fig_h = 2 * margin + len(rows) * row_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    for ri, row in enumerate(rows):
        y_top = fig_h - margin - ri * row_h
        alloc = allocation_string(direction_counts(row.directions))
        fig.text(margin / fig_w, (y_top - title_h * 0.45) / fig_h,
                 f"{row.title}  -  front/back/oblique/side = {alloc}",
                 fontsize=8, va="center", ha="left")
        for ci, img in enumerate(row.images):
            x0 = margin + ci * (cell_w + gap)
            y0 = y_top - title_h - img_h
            ax = fig.add_axes((x0 / fig_w, y0 / fig_h, cell_w / fig_w, img_h / fig_h))
            ax.imshow(img, interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.text((x0 + cell_w / 2) / fig_w, (y0 - label_h * 0.5) / fig_h, row.directions[ci],
                     fontsize=6.5, ha="center", va="center")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, facecolor="white")
    return fig
