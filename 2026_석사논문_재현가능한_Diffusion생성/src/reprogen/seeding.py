"""Per-image seeding that mirrors ``patches/ipg_seed.patch``.

The patched IPG ``inference.py`` reseeds the diffusion generator right before
each reference image is generated::

    generator.manual_seed(12580 + (zlib.crc32(ref_name.encode()) & 0x7fffffff))

``ref_name`` is the basename of the reference image (e.g. ``0001_c1s1_001051_00.jpg``).
The seed therefore depends only on the file name, not on the processing order,
the other images in the folder, or which pose set is used. With the same number
of poses (batch size) the initial noise for a given image is identical across
pose sets and across re-runs.
"""
from __future__ import annotations

import zlib
from pathlib import Path

BASE_SEED: int = 12580
"""Default base seed (``seed`` in IPG ``configs/inference.yaml``)."""

SEED_MASK: int = 0x7FFFFFFF
"""Mask applied to the CRC32 value (keeps the offset in the positive int32 range)."""

SEED_PATCH_MARKER: str = "zlib.crc32"
"""String whose presence in ``inference.py`` indicates the seed patch is applied."""


def name_offset(name: str) -> int:
    """Return the masked CRC32 of a file name (UTF-8 encoded)."""
    return zlib.crc32(name.encode()) & SEED_MASK


def image_seed(name: str, base: int = BASE_SEED) -> int:
    """Return the generator seed used by the patched IPG for image ``name``.

    ``name`` should be a basename; a path is reduced to its basename first so
    that the result matches ``os.path.basename(path_ref)`` in IPG.
    """
    return base + name_offset(Path(name).name)


def is_seed_patched(inference_py: str | Path) -> bool:
    """Return True if ``inference_py`` contains the per-image seed patch."""
    path = Path(inference_py)
    if not path.is_file():
        return False
    return SEED_PATCH_MARKER in path.read_text(encoding="utf-8", errors="replace")
