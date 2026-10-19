#!/usr/bin/env python
"""Per-camera walking-direction analysis of a Re-ID gallery.

Either extract keypoints with DWPose (``--gallery``) or re-analyze a cached
keypoint file (``--keypoints``) without loading any pose model.

Examples:
    python scripts/analyze_cameras.py --gallery data/market1501/bounding_box_test \
        --parser market --samples 500 --out outputs/
    python scripts/analyze_cameras.py --keypoints outputs/keypoints_market.npz \
        --parser market --out outputs/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walkdir.analysis import analyze_keypoints  # noqa: E402
from walkdir.data import PARSERS, list_gallery, load_keypoints, sample_per_camera, save_keypoints  # noqa: E402
from walkdir.direction import DIRECTIONS  # noqa: E402
from walkdir.quality import REASONS, QualityConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gallery", help="gallery image folder (runs DWPose)")
    src.add_argument("--keypoints", help="cached keypoint .npz (no pose model needed)")
    ap.add_argument("--parser", choices=sorted(PARSERS), required=True, help="file-name convention")
    ap.add_argument("--name", default=None, help="dataset label for outputs (default: parser)")
    ap.add_argument("--samples", type=int, default=None, help="images per camera (default: config)")
    ap.add_argument("--out", default="outputs/", help="output folder")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--thresholds", default=None,
                    help='"preset", "auto" or "q33,q66" (default: config)')
    ap.add_argument("--device", default=None, help="pose model device, e.g. cpu or cuda")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def resolve_thresholds(spec: str, parser: str, cfg: dict) -> tuple[float, float] | None:
    """Turn the ``--thresholds`` spec into (q33, q66) or None for auto."""
    if spec == "auto":
        return None
    if spec == "preset":
        q = cfg["preset_thresholds"][parser]
        return float(q[0]), float(q[1])
    a, b = (float(x) for x in spec.split(","))
    return a, b


def extract_keypoints(args: argparse.Namespace, cfg: dict, samples: int) -> dict[str, np.ndarray]:
    """Sample the gallery per camera and run DWPose on every sampled image."""
    from PIL import Image

    from walkdir.pose import NUM_JOINTS, DWPoseExtractor

    items = sample_per_camera(list_gallery(args.gallery, args.parser), samples, cfg.get("seed", 0))
    if not items:
        raise SystemExit(f"no gallery images found in {args.gallery}")
    pc = cfg["pose"]
    ext = DWPoseExtractor(device=args.device or pc["device"], det_ckpt=pc.get("det_ckpt"),
                          pose_ckpt=pc.get("pose_ckpt"), detect_resolution=pc["detect_resolution"])
    n = len(items)
    bodies = np.zeros((n, NUM_JOINTS, 2), np.float32)
    scores = np.zeros((n, NUM_JOINTS), np.float32)
    detected = np.zeros(n, bool)
    for i, (f, _, _) in enumerate(items):
        body, sc = ext.extract(Image.open(f))
        if body is not None:
            bodies[i], scores[i], detected[i] = body, sc, True
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  pose {i + 1}/{n}", flush=True)
    kp = {"files": np.array([Path(f).name for f, _, _ in items]),
          "pids": np.array([p for _, p, _ in items]), "cams": np.array([c for _, _, c in items]),
          "bodies": bodies, "scores": scores, "detected": detected}
    return kp


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = args.name or args.parser
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = args.samples or cfg["samples"]

    if args.keypoints:
        kp = load_keypoints(args.keypoints)
    else:
        kp = extract_keypoints(args, cfg, samples)
        cache = out / f"keypoints_{name}.npz"
        save_keypoints(cache, kp["files"], kp["pids"], kp["cams"], kp["bodies"], kp["scores"], kp["detected"])
        print(f"keypoints cached: {cache}")

    q = cfg["quality"]
    qcfg = QualityConfig(mode="strict", conf_th=q["conf_th"], min_visible=q["min_visible"],
                         min_height=q["min_height"], limb_ratio=q["limb_ratio"])
    thr = resolve_thresholds(args.thresholds or cfg["thresholds"], args.parser, cfg)
    res = analyze_keypoints(kp, thr, qcfg, cfg.get("variation_quality", "core"))
    res["dataset"] = name

    t = res["thresholds"]
    print(f"[{name}] thresholds: side < {t['q33']:.3f} <= oblique < {t['q66']:.3f} <= front/back")
    print(f"{'cam':<5}{'n':>5}" + "".join(f"{d:>9}" for d in DIRECTIONS) + "   dropped")
    for c, r in res["cameras"].items():
        drop = " ".join(f"{k}:{v}" for k, v in r["dropped"].items() if v)
        print(f"C{c:<4}{r['n_classified']:>5}" + "".join(f"{r['direction'][d] * 100:>8.1f}%" for d in DIRECTIONS)
              + f"   {drop}")
    if "variation" in res:
        v = res["variation"]
        print(f"pose variation ({v['n_poses']} poses): intra {v['intra']:.3f}  inter {v['inter']:.3f}"
              f"  ratio {v['ratio']:.2f}x")

    js = out / f"directions_{name}.json"
    json.dump(res, open(js, "w"), indent=1)
    with open(out / f"directions_{name}.csv", "w") as f:
        f.write("camera,n_classified," + ",".join(DIRECTIONS) + "," + ",".join(f"drop_{r}" for r in REASONS) + "\n")
        for c, r in res["cameras"].items():
            f.write(f"{c},{r['n_classified']}," + ",".join(f"{r['direction'][d]:.4f}" for d in DIRECTIONS)
                    + "," + ",".join(str(r["dropped"][k]) for k in REASONS) + "\n")
    print(f"saved: {js}")

    if not args.no_figures:
        from walkdir.plot import plot_direction_bars, plot_variation_bars

        dist = {int(c): r["direction"] for c, r in res["cameras"].items()}
        p1 = plot_direction_bars({name: dist}, out / f"fig_directions_{name}.png",
                                 title=f"Walking-direction distribution per camera ({name})")
        print(f"saved: {p1}")
        if "variation" in res:
            v = res["variation"]
            p2 = plot_variation_bars({name: (v["intra"], v["inter"])}, out / f"fig_variation_{name}.png")
            print(f"saved: {p2}")


if __name__ == "__main__":
    main()
