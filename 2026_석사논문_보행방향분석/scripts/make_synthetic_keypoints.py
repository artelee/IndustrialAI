#!/usr/bin/env python
"""Write a synthetic keypoint file (no images, no pose model) for a CPU demo.

Example:
    python scripts/make_synthetic_keypoints.py --out outputs/keypoints_synthetic.npz
    python scripts/analyze_cameras.py --keypoints outputs/keypoints_synthetic.npz \
        --parser market --name synthetic --out outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walkdir.data import save_keypoints  # noqa: E402
from walkdir.synthetic import make_camera_set  # noqa: E402

MIXES = {
    1: {"front": 0.15, "back": 0.15, "oblique": 0.4, "side": 0.3},
    2: {"front": 0.45, "back": 0.4, "oblique": 0.1, "side": 0.05},
    3: {"front": 0.05, "back": 0.1, "oblique": 0.3, "side": 0.55},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/keypoints_synthetic.npz")
    ap.add_argument("--per-camera", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    kp = make_camera_set(MIXES, a.per_camera, a.seed)
    save_keypoints(a.out, kp["files"], kp["pids"], kp["cams"], kp["bodies"], kp["scores"], kp["detected"])
    print(f"saved {len(kp['files'])} synthetic skeletons: {a.out}")


if __name__ == "__main__":
    main()
