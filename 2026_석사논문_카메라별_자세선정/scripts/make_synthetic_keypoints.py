#!/usr/bin/env python
"""Write a keypoint JSON for synthetic cameras (CPU demo; no images, no pose model).

Example:
    python scripts/make_synthetic_keypoints.py --out outputs/demo/keypoints.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campose.data import save_keypoints_json  # noqa: E402
from campose.synthetic import make_camera_samples  # noqa: E402

#: Direction mixes of the demo cameras (front, back, oblique, side shares).
DEMO_MIXES = {
    1: {"front": 0.13, "back": 0.17, "oblique": 0.37, "side": 0.33},   # close to the common 1/1/3/3
    2: {"front": 0.35, "back": 0.17, "oblique": 0.27, "side": 0.21},   # front-heavy
    3: {"front": 0.49, "back": 0.45, "oblique": 0.06, "side": 0.0},    # no side views
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="poses per camera")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = {f"c{c}": make_camera_samples(mix, n=args.n, seed=args.seed + c, cam=c)
            for c, mix in DEMO_MIXES.items()}
    save_keypoints_json(args.out, data)
    print(f"saved {sum(len(v) for v in data.values())} synthetic poses of {len(data)} cameras: {args.out}")


if __name__ == "__main__":
    main()
