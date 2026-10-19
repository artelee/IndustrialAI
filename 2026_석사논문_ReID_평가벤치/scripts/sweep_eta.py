#!/usr/bin/env python
"""Fusion-weight (eta) sensitivity: mAP of baseline / common fusion / per-camera
fusion for each eta, with paired bootstrap CI of (per-camera - common).

Example:
    python scripts/sweep_eta.py --config configs/default.yaml --cache-dir data/features --backbone clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.conditions import load_experiment  # noqa: E402
from reidbench.config import experiment_columns, load_config  # noqa: E402
from reidbench.io import cache_path  # noqa: E402
from reidbench.report_md import eta_table_md, write_json, write_text  # noqa: E402
from reidbench.sweep import eta_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--backbone")
    ap.add_argument("--etas", type=float, nargs="*", help="override sweep.etas")
    ap.add_argument("--only", nargs="*", help="run only experiments whose label contains one of these strings")
    args = ap.parse_args()
    cfg = load_config(args.config, cache_dir=args.cache_dir, out_dir=args.out_dir, backbone=args.backbone)
    bb = cfg["backbone"]
    etas = [float(x) for x in (args.etas or cfg["sweep"]["etas"])]
    rng = np.random.RandomState(cfg["seed"])
    flat, cells, cols = {}, {}, []
    for e in experiment_columns(cfg):
        if args.only and not any(s in e["label"] for s in args.only):
            continue
        if not cache_path(cfg["cache_dir"], bb, e["train"], e["eval"], "gal").exists():
            continue
        ds = cfg["datasets"][e["eval"]]
        exp = load_experiment(cfg["cache_dir"], bb, e["train"], e["eval"], ds["cameras"],
                              ds.get("common_set"), ds.get("camera_set"))
        if not exp.has_sets:
            continue
        print(f"[{e['label']}]", flush=True)
        res = eta_sweep(exp, etas, cfg["assign_modes"], (False, True), rng=rng, n_boot=cfg["n_boot"])
        cols.append(e["label"])
        for (mode, cal, eta), r in res.items():
            flat[f"{e['label']}|{mode}|{'on' if cal else 'off'}|{eta}"] = r
            cells[(e["label"], mode, cal, eta)] = r
            print(f"  {mode:<6} calib {'on ' if cal else 'off'} eta={eta:<5g} base {r['base']:.1f}  "
                  f"common {r['common']:.1f}  per-camera {r['per_camera']:.1f}  "
                  f"diff {r['d']['mean']:+.2f}{'*' if r['d']['sig'] else ''}", flush=True)
    out_dir = Path(cfg["out_dir"])
    write_json(out_dir / f"eta_sweep_{bb}.json", flat)
    md = [f"# Fusion weight sensitivity ({bb}, mAP %: common / per-camera / diff)\n"]
    for mode in cfg["assign_modes"]:
        for cal in (False, True):
            sub = {(c, eta): cells[(c, mode, cal, eta)] for c in cols for eta in etas}
            md.append(eta_table_md(sub, etas, cols, title=f"assignment {mode}, calibration {'on' if cal else 'off'}"))
    write_text(out_dir / f"eta_sweep_{bb}.md", "\n".join(md))
    print(f"\nwrote {out_dir / f'eta_sweep_{bb}.json'} and .md")


if __name__ == "__main__":
    main()
