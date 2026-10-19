"""Market-style file naming for MSMT17 images."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_MARKET_RE = re.compile(r"^(-?\d+)_c(\d+)_")


def camera_from_name(basename: str) -> int:
    """Camera index from an MSMT17 file name ``{pid}_{idx}_{cam}_{date}_{frame}_{n}.jpg``.

    MSMT17 cameras are already numbered 1..15 (``01``..``15``), so no offset is applied.
    """
    fields = basename.split("_")
    if len(fields) < 3 or not fields[2].isdigit():
        raise ValueError(f"cannot read camera id from {basename!r}")
    return int(fields[2])


def market_name(pid: int, rel_path: str) -> str:
    """Target name ``{pid:04d}_c{cam}_{original basename}`` (camera not zero-padded)."""
    basename = PurePosixPath(rel_path).name
    return f"{pid:04d}_c{camera_from_name(basename)}_{basename}"


def parse_market_name(name: str) -> tuple[int, int]:
    """Return ``(pid, cam)`` from a Market-style file name."""
    m = _MARKET_RE.match(name)
    if m is None:
        raise ValueError(f"not a Market-style name: {name!r}")
    return int(m.group(1)), int(m.group(2))
