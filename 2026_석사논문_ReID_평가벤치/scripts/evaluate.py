#!/usr/bin/env python
"""Evaluate one query/gallery pair from the unified cache: Rank-1, mAP and the
per-camera breakdown (optionally with camera calibration). Saves per-query
scores so that two runs can be compared with scripts/compare.py.

Example:
    python scripts/evaluate.py --query data/features/clip_duke_market_query.npz \
        --gallery data/features/clip_duke_market_gal.npz --out outputs/eval/base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.breakdown import per_camera, weighted_mean  # noqa: E402
from reidbench.fusion import camera_mean_subtract  # noqa: E402
from reidbench.io import load_split  # noqa: E402
from reidbench.metrics import evaluate_per_query, l2_normalize  # noqa: E402
from reidbench.report_md import markdown_table, write_json, write_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, help="query .npz (feat, pid, cam, name)")
    ap.add_argument("--gallery", required=True, help="gallery .npz (feat, pid, cam, name)")
    ap.add_argument("--calibrate", action="store_true", help="subtract per-camera mean features")
    ap.add_argument("--out", default="outputs/eval/run", help="output prefix (.json, .md, _per_query.npz)")
    args = ap.parse_args()

    q, g = load_split(args.query), load_split(args.gallery)
    qf, gf = l2_normalize(q.feat), l2_normalize(g.feat)
    if args.calibrate:
        qf, gf = camera_mean_subtract(qf, q.cam), camera_mean_subtract(gf, g.cam)
    res = evaluate_per_query(qf, q.pid, q.cam, gf, g.pid, g.cam)
    cams_map = per_camera(res.ap[res.valid], q.cam[res.valid])
    cams_r1 = per_camera(res.r1[res.valid], q.cam[res.valid])
    summary = dict(query=Path(args.query).name, gallery=Path(args.gallery).name, calibrate=args.calibrate,
                   n_query=len(q), n_gallery=len(g), n_valid_query=int(res.valid.sum()),
                   rank1=res.rank1, mAP=res.mAP,
                   per_camera={c: dict(mAP=cams_map[c]["mean"], rank1=cams_r1[c]["mean"], n=cams_map[c]["n"])
                               for c in cams_map},
                   per_camera_weighted_mAP=weighted_mean(cams_map))
    out = Path(args.out)
    write_json(out.with_suffix(".json"), summary)
    rows = [[c, v["n"], f"{v['mAP']:.1f}", f"{v['rank1']:.1f}"] for c, v in summary["per_camera"].items()]
    rows.append(["all", summary["n_valid_query"], f"{res.mAP:.1f}", f"{res.rank1:.1f}"])
    write_text(out.with_suffix(".md"), markdown_table(["Query camera", "Queries", "mAP %", "Rank-1 %"], rows) + "\n")
    np.savez(f"{out}_per_query.npz", name=q.name, cam=q.cam, r1=res.r1, ap=res.ap, valid=res.valid)
    print(f"Rank-1 {res.rank1:.2f}  mAP {res.mAP:.2f}  (per-camera weighted mAP {summary['per_camera_weighted_mAP']:.2f})")
    print(f"wrote {out.with_suffix('.json')}, {out.with_suffix('.md')}, {out}_per_query.npz")


if __name__ == "__main__":
    main()
