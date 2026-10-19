#!/usr/bin/env python
"""Paired query-level bootstrap 95% CI of the difference between two runs of
scripts/evaluate.py (B minus A), overall and per query camera.

Example:
    python scripts/compare.py --a outputs/eval/base_per_query.npz --b outputs/eval/calib_per_query.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.bootstrap import format_diff, paired_bootstrap_ci  # noqa: E402
from reidbench.report_md import markdown_table, write_json, write_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="per-query npz of condition A")
    ap.add_argument("--b", required=True, help="per-query npz of condition B")
    ap.add_argument("--metric", default="ap", choices=["ap", "r1"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval/compare")
    args = ap.parse_args()
    A, B = np.load(args.a), np.load(args.b)
    if not np.array_equal(A["name"], B["name"]):
        raise SystemExit("the two runs must use the same queries in the same order")
    a, b, cam = A[args.metric], B[args.metric], A["cam"]
    rng = np.random.RandomState(args.seed)
    res = {"all": paired_bootstrap_ci(a, b, n_boot=args.n_boot, rng=rng)}
    for c in np.unique(cam):
        sel = cam == c
        res[int(c)] = paired_bootstrap_ci(a[sel], b[sel], n_boot=args.n_boot, rng=rng)
    out = Path(args.out)
    write_json(out.with_suffix(".json"), dict(a=args.a, b=args.b, metric=args.metric, n_boot=args.n_boot,
                                              seed=args.seed, diff=res))
    rows = [[k, v["n"], format_diff(v, with_ci=True)] for k, v in res.items()]
    write_text(out.with_suffix(".md"), markdown_table(["Query camera", "Queries", f"Δ {args.metric} (B − A, %p)"], rows)
               + "\n\n\\* = 95% CI excludes 0.\n")
    for k, v in res.items():
        print(f"{str(k):>4}  n={v['n']:5d}  {format_diff(v, with_ci=True)}")
    print(f"wrote {out.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
