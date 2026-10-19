"""Detect (and optionally delete) broken output images.

A generation job killed mid-write can leave zero-byte or truncated files. Since
resume skips any file that already exists, such files must be removed before
resuming, otherwise they are never regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .crop import is_panel_width


@dataclass(frozen=True)
class BadFile:
    """A file that failed the check, with the reason."""

    path: Path
    reason: str


def check_image(path: str | Path, full_decode: bool = True, num_poses: int | None = None) -> str | None:
    """Return ``None`` if ``path`` is a readable image, else a short reason.

    ``full_decode=False`` only runs ``PIL.Image.verify()`` (fast, but may miss
    truncated JPEG data); ``True`` additionally decodes all pixels. If
    ``num_poses`` is given, the width must split into ``2 * num_poses + 1``
    panel columns.
    """
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return "empty file"
        with Image.open(path) as im:
            im.verify()
        if full_decode or num_poses is not None:
            with Image.open(path) as im:
                im.load()
                width = im.size[0]
            if num_poses is not None and not is_panel_width(width, num_poses):
                return f"width {width} is not a {2 * num_poses + 1}-column panel"
    except Exception as exc:  # PIL raises many different exception types
        return f"{type(exc).__name__}: {exc}"
    return None


def find_bad_images(
    directory: str | Path,
    pattern: str = "*.jpg",
    full_decode: bool = True,
    num_poses: int | None = None,
) -> list[BadFile]:
    """Check every file matching ``pattern`` in ``directory`` (sorted order)."""
    bad: list[BadFile] = []
    for p in sorted(Path(directory).glob(pattern)):
        if p.is_file():
            reason = check_image(p, full_decode=full_decode, num_poses=num_poses)
            if reason is not None:
                bad.append(BadFile(p, reason))
    return bad


def remove_bad_images(bad: list[BadFile]) -> int:
    """Delete the given files; return how many were removed."""
    n = 0
    for b in bad:
        try:
            b.path.unlink()
            n += 1
        except FileNotFoundError:
            pass
    return n
