#!/usr/bin/env python
"""Side-by-side check figure: common poses vs. camera-specific poses (skeletons only).

Example:
    python scripts/compare_poses.py \
        --common-dir path/to/Pose2ID/IPG/standard_poses \
        --camera "Market C2=outputs/poses/market/c2,outputs/selection/market/c2.json" \
        --camera "Market C6=outputs/poses/market/c6,outputs/selection/market/c6.json" \
        --out outputs/compare_market.png

``--camera LABEL=POSE_DIR,SELECTION_JSON`` may be repeated. Pose images are read
as stored by render_poses.py / the reference poses (BGR-swapped files).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campose.compare import make_comparison_figure, row_from_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common-dir", help="directory with the common reference poses 1.jpg..8.jpg")
    ap.add_argument("--common-config", default="configs/common_poses.yaml",
                    help="YAML with the directions of the common poses")
    ap.add_argument("--camera", action="append", default=[], help="LABEL=POSE_DIR,SELECTION_JSON")
    ap.add_argument("--out", required=True, help="output figure (png)")
    args = ap.parse_args()

    rows = []
    if args.common_dir:
        dirs = yaml.safe_load(open(args.common_config, encoding="utf-8"))["directions"]
        paths = [Path(args.common_dir) / f"{k}.jpg" for k in range(1, len(dirs) + 1)]
        rows.append(row_from_files("Common poses", paths, dirs))
    for spec in args.camera:
        label, rest = spec.split("=", 1)
        pose_dir, sel_json = rest.split(",", 1)
        sel = json.load(open(sel_json, encoding="utf-8"))
        poses = sel["poses"]
        paths = [Path(pose_dir) / f"{p['idx']}.jpg" for p in poses]
        rows.append(row_from_files(label, paths, [p["dir"] for p in poses]))
    if not rows:
        sys.exit("nothing to draw: give --common-dir and/or --camera")
    make_comparison_figure(rows, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
