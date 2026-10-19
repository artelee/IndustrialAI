"""Parsing of the MSMT17 V1 ``list_*.txt`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ListEntry:
    """One line of a list file: image path relative to its image folder and person id."""

    rel_path: str
    pid: int


@dataclass(frozen=True)
class SplitSpec:
    """Where a list file's images live and which Market folder they go to."""

    list_name: str
    image_dir: str
    market_dir: str


# MSMT17 V1: query/gallery images are under test/, train/val images under train/.
QUERY = SplitSpec("list_query.txt", "test", "query")
GALLERY = SplitSpec("list_gallery.txt", "test", "bounding_box_test")
TRAIN = SplitSpec("list_train.txt", "train", "bounding_box_train")
VAL = SplitSpec("list_val.txt", "train", "bounding_box_train")


def parse_list_line(line: str) -> ListEntry | None:
    """Parse ``"0000/0000_000_01_0303morning_0015_0.jpg 0"``; return None for blank lines."""
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 2:
        raise ValueError(f"expected '<path> <pid>', got: {line!r}")
    return ListEntry(rel_path=parts[0], pid=int(parts[1]))


def read_list(path: str | Path) -> list[ListEntry]:
    """Read every entry of one ``list_*.txt`` file, in file order."""
    entries: list[ListEntry] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                entry = parse_list_line(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from None
            if entry is not None:
                entries.append(entry)
    return entries


def splits(include_val: bool = False) -> list[SplitSpec]:
    """Splits to convert. ``list_val.txt`` is merged into the train folder only on request."""
    out = [QUERY, GALLERY, TRAIN]
    if include_val:
        out.append(VAL)
    return out
