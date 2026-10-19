"""Render selected poses exactly like the generator's reference pose images.

The reference poses of Pose2ID's IPG (``standard_poses/*.jpg``) are 64x128
(width x height) skeleton images drawn with the OpenPose body style, drawn on a
large canvas and then downscaled. To make camera-specific poses directly
interchangeable with them, a pose is rendered as follows:

1. ``place``: keep the pose's own aspect ratio but move/scale it so that its
   top end, bottom end and horizontal center match the reference layout
   (median over the reference images, see :func:`measure_layout`). If the
   scaled pose is wider than 95% of the canvas, it is scaled by width instead.
2. ``draw_bodypose`` on a ``canvas_h x canvas_h/2`` black canvas
   (default ``canvas_h = 320``), colors as in controlnet_aux (RGB).
3. LANCZOS downscale to 64x128 (W x H) with PIL.
4. ``save_like_reference``: the RGB array is written with ``cv2.imwrite``
   unchanged, so the file stores the channels in BGR-swapped order - the same
   as the reference poses. IPG reads poses with ``cv2.imread`` and then
   ``Image.fromarray`` (interpreting BGR as RGB), so the generator sees the
   controlnet_aux colors as intended.

``draw_bodypose`` uses controlnet_aux when it is installed (``backend="auto"``)
and otherwise a built-in copy of the same function (pixel-identical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

#: Output size of a rendered pose, (width, height) as in PIL.
OUT_SIZE: tuple[int, int] = (64, 128)
DEFAULT_CANVAS_H = 320
DEFAULT_CONF_TH = 0.3


@dataclass(frozen=True)
class Layout:
    """Where the body sits in a reference pose image (fractions of height / width)."""

    top: float
    bottom: float
    cx: float


#: Median layout of Pose2ID's eight reference poses (IPG ``standard_poses``),
#: measured with :func:`measure_layout` (threshold 40 on the grayscale image).
REFERENCE_LAYOUT = Layout(top=0.07421875, bottom=0.9375, cx=0.48828125)


def measure_layout(paths: Sequence[str | Path], threshold: float = 40) -> Layout:
    """Median top / bottom / horizontal-center of the drawn pixels of reference pose images."""
    from PIL import Image

    tops, bots, cxs = [], [], []
    for p in paths:
        g = np.asarray(Image.open(p).convert("L"), float)
        ys, xs = np.where(g > threshold)
        if len(ys) == 0:
            continue
        tops.append(ys.min() / g.shape[0])
        bots.append(ys.max() / g.shape[0])
        cxs.append((xs.min() + xs.max()) / 2 / g.shape[1])
    if not tops:
        raise ValueError("no drawn pixels found in the reference images")
    return Layout(float(np.median(tops)), float(np.median(bots)), float(np.median(cxs)))


# ---------------------------------------------------------------------------
# OpenPose body drawing.
# Adapted from controlnet_aux 0.0.10 ``controlnet_aux/dwpose/util.py::draw_bodypose``
# (Apache-2.0; https://github.com/huggingface/controlnet_aux), which in turn derives
# from ControlNet (lllyasviel) and pytorch-openpose (Hzzone), reproducing the
# CMU OpenPose body visualization. Logic, constants and drawing order are kept
# identical so that the output matches controlnet_aux pixel for pixel.
# ---------------------------------------------------------------------------
_LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
             [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
             [1, 16], [16, 18], [3, 17], [6, 18]]
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
           [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def draw_bodypose_builtin(canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """OpenPose-style body drawing (copy of controlnet_aux ``draw_bodypose``).

    Args:
        canvas: HxWx3 uint8 image (modified in place by the limb pass, like the original).
        candidate: Nx2 joint coordinates in [0, 1] (x / W, y / H).
        subset: P x 18 joint indices into ``candidate`` (-1 = missing).
    """
    import cv2

    H, W, _ = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(_LIMB_SEQ[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, _COLORS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, _COLORS[i], thickness=-1)
    return canvas


DrawFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def get_draw_bodypose(backend: str = "auto") -> DrawFn:
    """Return the body drawing function.

    ``"controlnet_aux"`` requires controlnet_aux, ``"builtin"`` uses the copy above,
    ``"auto"`` prefers controlnet_aux and falls back to the built-in copy.
    """
    if backend not in ("auto", "controlnet_aux", "builtin"):
        raise ValueError(f"unknown backend: {backend}")
    if backend in ("auto", "controlnet_aux"):
        try:
            from controlnet_aux.dwpose.util import draw_bodypose  # type: ignore

            return draw_bodypose
        except ImportError:
            if backend == "controlnet_aux":
                raise
    return draw_bodypose_builtin


# ---------------------------------------------------------------------------
# Placement and rendering
# ---------------------------------------------------------------------------
def place(
    body: np.ndarray,
    score: np.ndarray,
    w: float,
    h: float,
    canvas_h: int = DEFAULT_CANVAS_H,
    layout: Layout = REFERENCE_LAYOUT,
    th: float = DEFAULT_CONF_TH,
) -> np.ndarray:
    """Move a pose to the reference layout; returns canvas-normalized 18x2 coordinates.

    ``body`` is image-relative (x / w, y / h) for an image of size ``w x h``; it is
    converted to pixels first so the pose keeps its true aspect ratio.
    """
    cw, ch = canvas_h // 2, canvas_h
    P = np.asarray(body, float) * np.array([w, h], float)
    v = np.asarray(score) > th
    if not v.any():
        raise ValueError("no visible joint to place")
    top, bot = P[v, 1].min(), P[v, 1].max()
    xmin, xmax = P[v, 0].min(), P[v, 0].max()
    s = (layout.bottom - layout.top) * ch / max(bot - top, 1e-6)
    if (xmax - xmin) * s > 0.95 * cw:
        s = 0.95 * cw / (xmax - xmin)
    Q = (P - np.array([(xmin + xmax) / 2, top])) * s + np.array([layout.cx * cw, layout.top * ch])
    return Q / np.array([cw, ch], float)


def render_rgb(
    body: np.ndarray,
    score: np.ndarray,
    w: float,
    h: float,
    canvas_h: int = DEFAULT_CANVAS_H,
    layout: Layout = REFERENCE_LAYOUT,
    th: float = DEFAULT_CONF_TH,
    out_size: tuple[int, int] = OUT_SIZE,
    backend: str = "auto",
) -> np.ndarray:
    """Render one pose; returns an ``out_size[1] x out_size[0] x 3`` uint8 array in RGB order."""
    from PIL import Image

    pts = place(body, score, w, h, canvas_h, layout, th)
    subset = np.arange(18, dtype=float)[None, :]
    subset[0, np.asarray(score) < th] = -1
    canvas = np.zeros((canvas_h, canvas_h // 2, 3), dtype=np.uint8)
    canvas = get_draw_bodypose(backend)(canvas, pts.copy(), subset)
    return np.array(Image.fromarray(canvas).resize(out_size, Image.LANCZOS))


def save_like_reference(rgb: np.ndarray, path: str | Path, jpeg_quality: int = 95) -> None:
    """Write an RGB render the way the reference poses are stored (BGR-swapped file).

    The array is passed to ``cv2.imwrite`` unchanged; OpenCV assumes BGR, so a
    viewer that reads the file as RGB sees red and blue swapped.
    """
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"could not write {path}")


def load_display_rgb(path: str | Path) -> np.ndarray:
    """Load a pose image saved by :func:`save_like_reference` with its intended (controlnet_aux) colors."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read {path}")
    return img  # cv2 returns the stored channel order, i.e. the original RGB array
