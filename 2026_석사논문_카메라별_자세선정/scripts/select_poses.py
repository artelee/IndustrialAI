#!/usr/bin/env python
"""Select K camera-specific generation poses per camera.

Per camera: sample gallery images -> DWPose keypoints -> quality filter ->
walking direction -> proportional allocation of K poses -> per-direction
K-means -> real pose nearest to each cluster center.

Examples:
    # from gallery images (needs the optional DWPose dependencies)
    python scripts/select_poses.py --config configs/market.yaml \
        --gallery data/market1501/bounding_box_test --out outputs/selection/market \
        --save-keypoints outputs/keypoints/market.json --device cpu

    # from a keypoint JSON (no pose model; see scripts/make_synthetic_keypoints.py)
    python scripts/select_poses.py --config configs/market.yaml \
        --keypoints outputs/demo/keypoints.json --out outputs/demo/selection

Output: ``<out>/c<cam>.json`` with ``alloc``, ``frac``, ``n``, ``drop`` and
``poses`` (``idx``, ``dir``, ``src``, ``body``, ``score``, ``image_size``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from campose.allocate import allocation_string  # noqa: E402
from campose.data import (  # noqa: E402
    detection_size,
    load_keypoints_json,
    sample_cameras,
    save_keypoints_json,
)
from campose.posecore.quality import QualityConfig  # noqa: E402
from campose.select import select_camera_poses  # noqa: E402


def extract_keypoints(
    paths: list[str], device: str, resolution: int = 512
) -> list[dict[str, Any]]:
    """Run DWPose on gallery images; returns keypoint samples."""
    from PIL import Image

    from campose.posecore.pose import DWPoseExtractor

    ext = DWPoseExtractor(device=device, detect_resolution=resolution)
    out = []
    for i, p in enumerate(paths, 1):
        img = Image.open(p).convert("RGB")
        body, score = ext.extract(img)
        out.append(dict(
            src=os.path.basename(p),
            body=None if body is None else body.tolist(),
            score=None if score is None else score.tolist(),
            image_size=list(detection_size(*img.size, resolution=resolution)),
        ))
        if i % 100 == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="dataset config, e.g. configs/market.yaml")
    ap.add_argument("--gallery", help="override gallery directory from the config")
    ap.add_argument("--keypoints", help="keypoint JSON {'c<cam>': [samples]} instead of running DWPose")
    ap.add_argument("--save-keypoints", help="write extracted keypoints to this JSON")
    ap.add_argument("--out", required=True, help="output directory for c<cam>.json")
    ap.add_argument("--cams", type=int, nargs="*", help="subset of cameras")
    ap.add_argument("--k", type=int, help="override number of poses per camera")
    ap.add_argument("--q33", type=float, help="override side/oblique threshold")
    ap.add_argument("--q66", type=float, help="override oblique/front-back threshold")
    ap.add_argument("--device", default="cpu", help="DWPose device (cpu or cuda)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    q33, q66 = cfg["thresholds"]
    q33 = args.q33 if args.q33 is not None else q33
    q66 = args.q66 if args.q66 is not None else q66
    k = args.k or cfg["k"]
    cams = args.cams or cfg["cams"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.keypoints:
        all_kp = load_keypoints_json(args.keypoints)
        cams = [c for c in cams if f"c{c}" in all_kp]
    else:
        gallery = args.gallery or cfg["gallery"]
        # sample over all configured cameras so the shared RNG matches a full run
        sampled = sample_cameras(gallery, cfg["parser"], cfg["cams"], cfg["samples_per_camera"], cfg["seed"])
        all_kp = {}
        for c in cams:
            print(f"C{c}: extracting keypoints of {len(sampled[c])} images", flush=True)
            all_kp[f"c{c}"] = extract_keypoints(sampled[c], args.device)
        if args.save_keypoints:
            save_keypoints_json(args.save_keypoints, all_kp)

    for c in cams:
        res = select_camera_poses(
            all_kp[f"c{c}"], q33, q66, k=k, quality=QualityConfig(mode="strict"),
            min_per_dir=cfg["min_per_dir"], min_samples=cfg["min_samples"],
            seed=cfg["seed"], n_init=cfg["kmeans_n_init"],
        )
        res.update(camera=c, dataset=cfg["dataset"], thresholds=[q33, q66])
        with open(out_dir / f"c{c}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        drops = " ".join(f"{r}:{n}" for r, n in res["drop"].items() if n)
        fr = " ".join(f"{d}:{v * 100:.0f}%" for d, v in res["frac"].items())
        print(f"C{c}: n={res['n']} (dropped {drops or 'none'})  {fr}  "
              f"-> front/back/oblique/side = {allocation_string(res['alloc'])}")
    print(f"saved: {out_dir}/c*.json")


if __name__ == "__main__":
    main()
