#!/usr/bin/env python
"""Per-query-tag breakdown (e.g. walking direction of the query): mAP of common
vs per-camera fusion and paired bootstrap CI of the difference for each tag.

Tag files are JSON ``{query_image_name: tag}``; queries without a tag get ``unknown``
and only appear in the ``all`` row.

Example:
    python scripts/query_tags.py --config configs/default.yaml --cache-dir data/features --backbone clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.breakdown import paired_by_tag  # noqa: E402
from reidbench.conditions import load_experiment, run_condition  # noqa: E402
from reidbench.config import experiment_columns, load_config  # noqa: E402
from reidbench.io import cache_path, load_tags  # noqa: E402
from reidbench.report_md import tag_table_md, write_json, write_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--backbone")
    ap.add_argument("--tag-dir", help="directory with query_direction_<eval>.json (overrides tags.files)")
    ap.add_argument("--assign", default="sim")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, cache_dir=args.cache_dir, out_dir=args.out_dir, backbone=args.backbone)
    bb, eta, order = cfg["backbone"], float(cfg["eta"]), list(cfg["tags"]["order"])
    rng = np.random.RandomState(cfg["seed"])
    flat, cells = {}, {}
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
        tag_file = (Path(args.tag_dir) / f"query_direction_{e['eval']}.json") if args.tag_dir \
            else cfg["tags"]["files"][e["eval"]]
        tags = load_tags(tag_file, exp.query.name)
        print(f"[{e['label']}] " + " ".join(f"{t} {(tags == t).sum()}" for t in order)
              + f"  other {(~np.isin(tags, order)).sum()}", flush=True)
        for cal in (False, True):
            _, ap_s = run_condition(exp, "S", cal, eta, args.assign)
            _, ap_t = run_condition(exp, "T", cal, eta, args.assign)
            res = paired_by_tag(ap_s, ap_t, tags, order, rng=rng, n_boot=cfg["n_boot"])
            cells[(e["label"], cal)] = res
            for t, r in res.items():
                flat[f"{e['label']}|{'on' if cal else 'off'}|{t}"] = dict(n=r["n"], common=r["a"], per_camera=r["b"], d=r["d"])
                print(f"  calib {'on ' if cal else 'off'} {t:<8} n={r['n']:5d}  common {r['a']:.1f}  "
                      f"per-camera {r['b']:.1f}  diff {r['d']['mean']:+.2f}{'*' if r['d']['sig'] else ''}", flush=True)
    out_dir = Path(cfg["out_dir"])
    write_json(out_dir / f"query_tags_{bb}.json", flat)
    cols = sorted({c for c, _ in cells}, key=[e["label"] for e in experiment_columns(cfg)].index)
    md = [f"# Per-query-tag difference, per-camera − common fusion ({bb}, eta = {eta:g}, mAP %p)\n"]
    for cal in (False, True):
        md.append(tag_table_md({c: cells[(c, cal)] for c in cols}, order + ["all"], cols,
                               title=f"calibration {'on' if cal else 'off'}"))
    write_text(out_dir / f"query_tags_{bb}.md", "\n".join(md))
    print(f"\nwrote {out_dir / f'query_tags_{bb}.json'} and .md")


if __name__ == "__main__":
    main()
