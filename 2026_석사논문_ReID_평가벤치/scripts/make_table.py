#!/usr/bin/env python
"""Main results table: mean mAP / Rank-1 over all queries, per-camera breakdown,
and paired bootstrap 95% CIs of key differences. Writes JSON + Markdown.

Example:
    python scripts/make_table.py --config configs/default.yaml --cache-dir data/features --backbone clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.bootstrap import paired_bootstrap_ci  # noqa: E402
from reidbench.breakdown import per_camera, weighted_mean  # noqa: E402
from reidbench.conditions import ROWS, load_experiment, run_all_conditions  # noqa: E402
from reidbench.config import experiment_columns, load_config  # noqa: E402
from reidbench.io import cache_path  # noqa: E402
from reidbench.report_md import final_table_md, write_json, write_text  # noqa: E402

#: (label, condition a, condition b): CI of b - a, in this order (matters for the shared RNG)
DIFFS = [
    ("Δ per-camera − common (calib off)", ("S", False), ("T", False)),
    ("Δ per-camera − common (calib on)", ("S", True), ("T", True)),
    ("Calib effect (Base)", ("B", False), ("B", True)),
    ("Calib effect (per-camera fusion)", ("T", False), ("T", True)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-dir", help="feature cache directory (read only)")
    ap.add_argument("--out-dir", help="output directory")
    ap.add_argument("--backbone")
    ap.add_argument("--eta", type=float)
    ap.add_argument("--only", nargs="*", help="run only experiments whose label contains one of these strings")
    args = ap.parse_args()
    cfg = load_config(args.config, cache_dir=args.cache_dir, out_dir=args.out_dir, backbone=args.backbone, eta=args.eta)
    bb, eta = cfg["backbone"], float(cfg["eta"])
    rng = np.random.RandomState(cfg["seed"])
    results: dict = {}
    for e in experiment_columns(cfg):
        if args.only and not any(s in e["label"] for s in args.only):
            continue
        if not cache_path(cfg["cache_dir"], bb, e["train"], e["eval"], "gal").exists():
            continue
        ds = cfg["datasets"][e["eval"]]
        exp = load_experiment(cfg["cache_dir"], bb, e["train"], e["eval"], ds["cameras"],
                              ds.get("common_set"), ds.get("camera_set"))
        qc = exp.query.cam
        print(f"[{e['label']}] gallery {len(exp.gallery)}  queries {len(exp.query)}  "
              f"assignment accuracy {exp.assign_accuracy:.1f}%", flush=True)
        keys = ("B", "S", "T") if exp.has_sets else ("B",)
        rows = [r for r in ROWS if r[1] in keys]
        diffs = [d for d in DIFFS if d[1][0] in keys and d[2][0] in keys]
        for mode in cfg["assign_modes"] if exp.has_sets else ["sim"]:
            res = run_all_conditions(exp, eta, mode, keys)
            out: dict = {lab: [float(res[(k, c)][0].mean() * 100), float(res[(k, c)][1].mean() * 100)]
                         for lab, k, c in rows}
            for lab, a, b in diffs:
                out[lab] = paired_bootstrap_ci(res[a][1], res[b][1], n_boot=cfg["n_boot"], rng=rng)
            cams = {lab: per_camera(res[(k, c)][1], qc) for lab, k, c in rows}
            out["per_camera_mAP"] = {cam: {lab: cams[lab][cam]["mean"] for lab, _, _ in rows} for cam in cams["Base"]}
            out["per_camera_n"] = {cam: v["n"] for cam, v in cams["Base"].items()}
            # consistency check: query-count weighted mean of per-camera mAP == overall mAP
            out["per_camera_weighted_mAP_check"] = {lab: weighted_mean(cams[lab]) for lab, _, _ in rows}
            for lab, _, _ in rows:
                assert abs(out["per_camera_weighted_mAP_check"][lab] - out[lab][1]) < 1e-6, lab
            out["assignment_accuracy"] = exp.assign_accuracy
            results[f"{e['label']}|{mode}"] = out
            print(f"  {mode:<6} " + "  ".join(f"{lab} {out[lab][1]:.1f}" for lab, _, _ in rows), flush=True)
            if exp.has_sets:
                d0, d1 = out[DIFFS[0][0]], out[DIFFS[1][0]]
                print(f"         per-camera − common mAP: off {d0['mean']:+.2f}{'*' if d0['sig'] else ''}  "
                      f"on {d1['mean']:+.2f}{'*' if d1['sig'] else ''}", flush=True)

    out_dir = Path(cfg["out_dir"])
    write_json(out_dir / f"final_table_{bb}.json", dict(backbone=bb, eta=eta, n_boot=cfg["n_boot"],
                                                        seed=cfg["seed"], results=results))
    md = [f"# Results ({bb}, eta = {eta:g})\n"]
    for mode in cfg["assign_modes"]:
        cols = [k.rsplit("|", 1)[0] for k in results if k.endswith(f"|{mode}")]
        if cols:
            sub = {c: results[f"{c}|{mode}"] for c in cols}
            md.append(final_table_md(sub, [r[0] for r in ROWS], [d[0] for d in DIFFS[:2]], cols,
                                     title=f"Reference assignment: {mode}"))
    write_text(out_dir / f"final_table_{bb}.md", "\n".join(md))
    print(f"\nwrote {out_dir / f'final_table_{bb}.json'} and .md")


if __name__ == "__main__":
    main()
