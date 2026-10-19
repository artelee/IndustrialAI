#!/usr/bin/env python
"""Write a small synthetic feature cache in the unified format (no real data needed).

Identities are random unit vectors, each camera adds its own bias, and the
extra-feature sets are noisy copies of the identity vector. Used for smoke
runs of all scripts (configs/smoke.yaml).

Example:
    python scripts/make_synthetic_cache.py --out outputs/synthetic_cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reidbench.io import FeatureSet, FeatureSplit, cache_path, save_set, save_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/synthetic_cache")
    ap.add_argument("--ids", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cams, k, d = [1, 2, 3], 8, args.dim
    ident = rng.normal(size=(args.ids, d))
    bias = {c: rng.normal(scale=0.8, size=d) for c in cams}

    def sample(pid: int, cam: int) -> np.ndarray:
        return ident[pid - 1] + bias[cam] + rng.normal(scale=0.9, size=d)

    q, g = [], []
    for pid in range(1, args.ids + 1):
        for cam in cams:
            q.append((pid, cam, sample(pid, cam)))
            for _ in range(2):
                g.append((pid, cam, sample(pid, cam)))
    for j in range(40):                        # distractors (pid 0)
        cam = cams[j % 3]
        g.append((0, cam, rng.normal(size=d) + bias[cam]))

    def split(rows: list, prefix: str) -> FeatureSplit:
        names = np.array([f"{p:04d}_c{c}_{prefix}{i:05d}.jpg" for i, (p, c, _) in enumerate(rows)])
        return FeatureSplit(feat=np.array([f for _, _, f in rows], np.float32), pid=np.array([p for p, _, _ in rows]),
                            cam=np.array([c for _, c, _ in rows]), name=names)

    qs, gs = split(q, "q"), split(g, "g")
    out = Path(args.out)
    save_split(cache_path(out, "toy", "toy", "toy", "query"), qs)
    save_split(cache_path(out, "toy", "toy", "toy", "gal"), gs)
    ref = np.array([np.where(gs.pid == p)[0][0] for p in range(1, args.ids + 1)])   # one reference image per id
    ref_name = gs.name[ref]
    common = ident[:, None, :] + rng.normal(scale=1.0, size=(args.ids, k, d))
    save_set(cache_path(out, "toy", "toy", "toy", "toy_std"), FeatureSet(common.astype(np.float32), ref_name))
    for c in cams:
        per_cam = ident[:, None, :] + 0.5 * bias[c] + rng.normal(scale=1.0, size=(args.ids, k, d))
        save_set(cache_path(out, "toy", "toy", "toy", f"toy_dm_c{c}"), FeatureSet(per_cam.astype(np.float32), ref_name))
    tags = {str(n): ["front", "back", "side"][i % 3] for i, n in enumerate(qs.name)}
    (out / "query_tags_toy.json").write_text(json.dumps(tags), encoding="utf-8")
    print(f"wrote synthetic cache to {out} ({len(qs)} queries, {len(gs)} gallery, {args.ids} reference sets)")


if __name__ == "__main__":
    main()
