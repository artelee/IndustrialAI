#!/usr/bin/env python
"""Render selected poses like the generator's reference poses (64x128, BGR-swapped JPEG).

Example:
    python scripts/render_poses.py --selection outputs/selection/market \
        --out outputs/poses/market --config configs/render.yaml
    # optionally re-measure the reference layout from Pose2ID's standard poses:
    #   --reference-dir path/to/Pose2ID/IPG/standard_poses

Output: ``<out>/c<cam>/<k>.jpg`` (k = 1..K in selection order), ready to be used as
a per-camera pose directory of the generator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campose.render import Layout, measure_layout, render_rgb, save_like_reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True, help="directory with c<cam>.json from select_poses.py")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--config", default="configs/render.yaml")
    ap.add_argument("--reference-dir", help="measure the layout from these reference pose images instead")
    ap.add_argument("--canvas-h", type=int, help="override canvas height")
    ap.add_argument("--backend", choices=["auto", "controlnet_aux", "builtin"], help="override drawing backend")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.reference_dir:
        refs = sorted(Path(args.reference_dir).glob("*.jpg"))
        layout = measure_layout(refs)
        print(f"reference layout from {len(refs)} images: {layout}")
    else:
        layout = Layout(**cfg["layout"])
    canvas_h = args.canvas_h or cfg["canvas_h"]
    backend = args.backend or cfg["backend"]

    files = sorted(Path(args.selection).glob("c*.json"), key=lambda p: int(p.stem[1:]))
    if not files:
        sys.exit(f"no c<cam>.json in {args.selection}")
    for f in files:
        sel = json.load(open(f, encoding="utf-8"))
        dst = Path(args.out) / f.stem
        dst.mkdir(parents=True, exist_ok=True)
        for p in sel["poses"]:
            w, h = p["image_size"]
            rgb = render_rgb(p["body"], p["score"], w, h, canvas_h=canvas_h, layout=layout,
                             th=cfg["conf_th"], out_size=tuple(cfg["out_size"]), backend=backend)
            save_like_reference(rgb, dst / f"{p['idx']}.jpg", cfg["jpeg_quality"])
        print(f"{f.stem}: {len(sel['poses'])} poses -> {dst}")


if __name__ == "__main__":
    main()
