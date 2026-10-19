"""Synthetic miniature MSMT17 V1 tree (tiny placeholder jpgs, no real images)."""

from __future__ import annotations

from pathlib import Path

import pytest

# Smallest byte string with JPEG start/end markers; content is never decoded.
TINY_JPG = b"\xff\xd8\xff\xd9"

# (list file, image folder, [(relative path, pid)])
MINI_LISTS = {
    "list_query.txt": ("test", [("0000/0000_000_01_0303morning_0015_0.jpg", 0),
                                ("0001/0001_003_15_0303noon_0100_1_ex.jpg", 1)]),
    "list_gallery.txt": ("test", [("0000/0000_008_01_0303morning_0019_2.jpg", 0),
                                  ("0000/0000_019_05_0303morning_0029_1.jpg", 0),
                                  ("0001/0001_010_10_0303noon_0200_0.jpg", 1)]),
    "list_train.txt": ("train", [("0000/0000_045_12_0303morning_0006_2.jpg", 0),
                                 ("0012/0012_046_02_0303morning_0006_2_ex.jpg", 12)]),
    "list_val.txt": ("train", [("0000/0000_013_04_0303morning_0020_2.jpg", 0)]),
}


@pytest.fixture
def mini_msmt(tmp_path: Path) -> Path:
    """Create a miniature MSMT17 V1 root and return it."""
    root = tmp_path / "msmt17_v1"
    for list_name, (image_dir, entries) in MINI_LISTS.items():
        lines = []
        for rel, pid in entries:
            img = root / image_dir / rel
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(TINY_JPG)
            lines.append(f"{rel} {pid}\n")
        (root / list_name).write_text("".join(lines))
    return root
