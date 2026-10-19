"""Figures: per-camera direction bars, variation bars and skeleton-only drawings.

All figures are drawn from statistics or keypoint arrays; no image pixels are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .direction import DIRECTIONS
from .pose import SKELETON_EDGES

DIRECTION_COLORS = {"front": "#2a78d6", "back": "#eb6834", "oblique": "#1baf7a", "side": "#eda100"}
INTRA_COLOR, INTER_COLOR = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT,
    })
    return plt


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_direction_bars(
    datasets: Mapping[str, Mapping[int, Mapping[str, float]]],
    out: str | Path,
    title: str = "Walking-direction distribution per camera",
) -> Path:
    """Stacked 100% bars of direction fractions, one panel per dataset."""
    plt = _plt()
    names = list(datasets)
    widths = [max(len(datasets[n]), 4) for n in names]
    fig, axes = plt.subplots(
        1, len(names), figsize=(1.6 + 0.55 * sum(widths), 3.9), dpi=150,
        gridspec_kw={"width_ratios": widths}, squeeze=False,
    )
    for ax, name in zip(axes[0], names):
        cams = sorted(datasets[name])
        x = np.arange(len(cams))
        bottom = np.zeros(len(cams))
        for d in DIRECTIONS:
            v = np.array([datasets[name][c].get(d, 0.0) * 100 for c in cams])
            ax.bar(x, v, bottom=bottom, width=0.72, color=DIRECTION_COLORS[d],
                   edgecolor="white", linewidth=1.0, label=d)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in cams], fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(name, fontsize=11)
        _style(ax)
    axes[0][0].set_ylabel("share of gallery sample (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def plot_variation_bars(
    values: Mapping[str, tuple[float, float]],
    out: str | Path,
    title: str = "Intra-camera vs. inter-camera pose variation",
) -> Path:
    """Grouped bars (intra, inter) per dataset with the intra/inter ratio annotated."""
    plt = _plt()
    names = list(values)
    intra = [values[n][0] for n in names]
    inter = [values[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(1.8 + 2.4 * len(names), 3.9), dpi=150)
    b1 = ax.bar(x - w / 2, intra, w, color=INTRA_COLOR, edgecolor="white", linewidth=1.0, label="intra-camera")
    b2 = ax.bar(x + w / 2, inter, w, color=INTER_COLOR, edgecolor="white", linewidth=1.0, label="inter-camera")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=9, color=TEXT)
    top = max(max(intra), max(inter), 1e-3) * 1.12
    for i, (a, b) in enumerate(zip(intra, inter)):
        ax.text(i, top + 0.08, f"{a / b:.1f}x" if b > 0 else "n/a", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("mean distance in pose space")
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    _style(ax)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def draw_skeleton(ax, body: np.ndarray, visible: np.ndarray | None = None, color: str = TEXT) -> None:
    """Draw an 18-joint skeleton (y axis pointing down) on a matplotlib axis."""
    body = np.asarray(body)
    vis = np.ones(len(body), bool) if visible is None else np.asarray(visible, bool)
    for a, b in SKELETON_EDGES:
        if vis[a] and vis[b]:
            ax.plot([body[a, 0], body[b, 0]], [body[a, 1], body[b, 1]], color=color, lw=2.2,
                    solid_capstyle="round")
    ax.scatter(body[vis, 0], body[vis, 1], s=14, color=color, zorder=3)
    # mark the person's right shoulder to show the left/right order rule
    if vis[2]:
        ax.scatter([body[2, 0]], [body[2, 1]], s=46, color="white", edgecolor=color, lw=1.6, zorder=4)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def plot_skeleton_examples(
    examples: Sequence[tuple[str, np.ndarray, str]], out: str | Path, title: str = ""
) -> Path:
    """Row of skeleton drawings: ``(label, normalized 18x2 body, caption)``."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(examples), figsize=(2.3 * len(examples), 3.9), dpi=150)
    for ax, (label, body, caption) in zip(np.atleast_1d(axes), examples):
        draw_skeleton(ax, body, color=DIRECTION_COLORS.get(label, TEXT))
        ax.set_title(label, fontsize=12, color=TEXT)
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    if title:
        fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out
