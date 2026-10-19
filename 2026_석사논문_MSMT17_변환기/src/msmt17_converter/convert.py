"""Plan, execute and verify the MSMT17 V1 -> Market-1501 layout conversion."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .lists import SplitSpec, read_list, splits
from .naming import market_name, parse_market_name

LINK_MODES = ("copy", "hardlink", "symlink")


@dataclass(frozen=True)
class PlannedFile:
    """One image to place: source path and target ``<market_dir>/<name>``."""

    market_dir: str
    src: Path
    name: str
    pid: int
    cam: int


@dataclass
class SplitStats:
    """Image, identity and camera summary of one Market folder."""

    images: int = 0
    pids: set[int] = field(default_factory=set)
    cams: Counter = field(default_factory=Counter)

    def add(self, pid: int, cam: int) -> None:
        """Count one image."""
        self.images += 1
        self.pids.add(pid)
        self.cams[cam] += 1

    def line(self, name: str) -> str:
        """One printable summary line."""
        if not self.images:
            return f"{name:<20} {0:>7,} images"
        lo, hi = min(self.cams), max(self.cams)
        return (f"{name:<20} {self.images:>7,} images  {len(self.pids):>5,} ids  "
                f"cameras {lo}-{hi} ({len(self.cams)} used)")


def plan(src_root: str | Path, include_val: bool = False) -> list[PlannedFile]:
    """Build the list of files to create, in list-file order. Raises on name collisions."""
    src_root = Path(src_root)
    planned: list[PlannedFile] = []
    seen: set[tuple[str, str]] = set()
    for spec in splits(include_val):
        planned.extend(_plan_split(src_root, spec, seen))
    return planned


def _plan_split(src_root: Path, spec: SplitSpec, seen: set[tuple[str, str]]) -> list[PlannedFile]:
    list_path = src_root / spec.list_name
    if not list_path.is_file():
        raise FileNotFoundError(f"missing list file: {list_path}")
    out = []
    for entry in read_list(list_path):
        name = market_name(entry.pid, entry.rel_path)
        key = (spec.market_dir, name)
        if key in seen:
            raise ValueError(f"duplicate target name {spec.market_dir}/{name}")
        seen.add(key)
        _, cam = parse_market_name(name)
        out.append(PlannedFile(spec.market_dir, src_root / spec.image_dir / entry.rel_path,
                               name, entry.pid, cam))
    return out


def summarize(files: list[PlannedFile]) -> dict[str, SplitStats]:
    """Per-folder statistics of a plan."""
    stats: dict[str, SplitStats] = {}
    for f in files:
        stats.setdefault(f.market_dir, SplitStats()).add(f.pid, f.cam)
    return stats


def missing_sources(files: list[PlannedFile]) -> list[Path]:
    """Source images listed in the list files but not present on disk."""
    return [f.src for f in files if not f.src.is_file()]


def place(src: Path, dst: Path, link: str) -> None:
    """Create ``dst`` from ``src`` by copy, hard link or (absolute) symlink."""
    if link == "copy":
        shutil.copy2(src, dst)
    elif link == "hardlink":
        os.link(src, dst)
    elif link == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"unknown link mode {link!r}; choose from {LINK_MODES}")


def execute(files: list[PlannedFile], dst_root: str | Path, link: str = "copy",
            overwrite: bool = False, progress_every: int = 20000) -> tuple[int, int]:
    """Write the planned files. Returns ``(written, skipped_existing)``."""
    if link not in LINK_MODES:
        raise ValueError(f"unknown link mode {link!r}; choose from {LINK_MODES}")
    dst_root = Path(dst_root)
    written = skipped = 0
    for d in sorted({f.market_dir for f in files}):
        (dst_root / d).mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dst_root / f.market_dir / f.name
        if target.exists() or target.is_symlink():
            if not overwrite:
                skipped += 1
                continue
            target.unlink()
        place(f.src, target, link)
        written += 1
        if progress_every and written % progress_every == 0:
            print(f"  {written:,} files written", flush=True)
    return written, skipped


def scan_market(dst_root: str | Path, dirs: list[str] | None = None) -> dict[str, SplitStats]:
    """Count ``*.jpg`` files per Market folder of an existing output (the verify step)."""
    dst_root = Path(dst_root)
    dirs = dirs or ["query", "bounding_box_test", "bounding_box_train"]
    stats: dict[str, SplitStats] = {}
    for d in dirs:
        s = stats.setdefault(d, SplitStats())
        folder = dst_root / d
        if not folder.is_dir():
            continue
        for entry in os.scandir(folder):
            if entry.name.endswith(".jpg"):
                s.add(*parse_market_name(entry.name))
    return stats


def compare_names(files: list[PlannedFile], market_root: str | Path) -> dict[str, tuple[int, int]]:
    """Compare planned names with an existing Market folder tree.

    Returns ``{market_dir: (planned_not_found, existing_not_planned)}``.
    """
    market_root = Path(market_root)
    planned: dict[str, set[str]] = {}
    for f in files:
        planned.setdefault(f.market_dir, set()).add(f.name)
    out = {}
    for d, names in planned.items():
        folder = market_root / d
        existing = {e.name for e in os.scandir(folder) if e.name.endswith(".jpg")} if folder.is_dir() else set()
        out[d] = (len(names - existing), len(existing - names))
    return out
