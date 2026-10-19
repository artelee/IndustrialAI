#!/usr/bin/env python
"""Base vs. + camera calibration for one or more (backbone, train, eval) settings.

For every setting the raw query / gallery features are loaded from the feature
cache (``--features``, default ``outputs/features``); missing files are
extracted first with the backbone (GPU recommended) and written there.
Reported: Rank-1 / mAP of Base (``N(f)``) and, with ``--calib``, of
+ Calib (``N(N(f) - mu_c)``, camera means per set), and the paired
query-level bootstrap 95 % CI of the difference.

Examples:
    python scripts/run_eval.py --backbone clipreid --train duke --eval market --calib
    python scripts/run_eval.py --backbone clipreid personvit --train market duke \
        --eval market duke --calib --features data/features      # all 8 settings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reid_calib.evaluate import evaluate_base_and_calibrated  # noqa: E402
from reid_calib.features import FeatureSplit, cache_path, load_split, save_split  # noqa: E402
from reid_calib.stats import format_diff, paired_bootstrap_ci  # noqa: E402

LABEL = {"market": "Market", "duke": "Duke"}
ORDER = [("market", "market"), ("duke", "duke"), ("duke", "market"), ("market", "duke")]  # table column order


def load_yaml(path: Path) -> dict:
    """Read a YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setting_name(train: str, eval_: str) -> str:
    """``Same Market`` / ``Cross Duke->Market``."""
    if train == eval_:
        return f"Same {LABEL[eval_]}"
    return f"Cross {LABEL[train]}->{LABEL[eval_]}"


def get_splits(args: argparse.Namespace, bb: dict, train: str, eval_: str,
               datasets: dict[str, dict]) -> tuple[FeatureSplit, FeatureSplit]:
    """Load cached query / gallery features, extracting the missing ones."""
    paths = {part: cache_path(args.features, bb["cache_prefix"], train, eval_, part) for part in ("query", "gal")}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        from reid_calib.extract import extract_split, load_backbone  # lazy: needs torch + upstream code

        print(f"  extracting {', '.join(p.name for p in missing)} on {args.device}", flush=True)
        model, forward = load_backbone(bb, train, datasets[train]["num_train_ids"], args.device)
        for part, p in paths.items():
            if p.exists():
                continue
            split = extract_split(model, forward, datasets[eval_], part == "query", bb["input"],
                                  args.device, int(bb.get("batch_size", 64)))
            save_split(p, split)
            print(f"  saved {p} ({len(split)} images)", flush=True)
    return load_split(paths["query"]), load_split(paths["gal"])


def run_setting(args: argparse.Namespace, bb: dict, train: str, eval_: str, datasets: dict[str, dict]) -> dict:
    """Evaluate one setting, write its JSON / per-query file and return the summary."""
    tag = f"{bb['name']}_{train}_{eval_}"
    print(f"\n=== {bb['name']}: {setting_name(train, eval_)} ===", flush=True)
    t0 = time.time()
    q, g = get_splits(args, bb, train, eval_, datasets)
    res = evaluate_base_and_calibrated(q.feat, q.pid, q.cam, g.feat, g.pid, g.cam, calibrate=args.calib)
    base = res["base"]
    out: dict = dict(backbone=bb["name"], train=train, eval=eval_, setting=setting_name(train, eval_),
                     n_query=len(q), n_gallery=len(g), n_valid_query=int(base.valid.sum()),
                     base=dict(rank1=base.rank1, mAP=base.mAP))
    per_query = dict(name=q.name, cam=q.cam, valid=base.valid, base_ap=base.ap, base_r1=base.r1)
    line = f"  Base   Rank-1 {base.rank1:6.2f}  mAP {base.mAP:6.2f}"
    if args.calib:
        cal = res["calib"]
        v = base.valid  # same queries are valid in both conditions (validity depends on labels only)
        rng = np.random.RandomState(args.seed)  # one stream per setting: mAP CI first, then Rank-1 CI
        out["calib"] = dict(rank1=cal.rank1, mAP=cal.mAP)
        out["diff_mAP"] = paired_bootstrap_ci(base.ap[v], cal.ap[v], n_boot=args.n_boot, rng=rng)
        out["diff_rank1"] = paired_bootstrap_ci(base.r1[v], cal.r1[v], n_boot=args.n_boot, rng=rng)
        out["bootstrap"] = dict(n_boot=args.n_boot, seed=args.seed, level=0.95, unit="percentage points")
        per_query.update(calib_ap=cal.ap, calib_r1=cal.r1)
        line += (f"\n  +Calib Rank-1 {cal.rank1:6.2f}  mAP {cal.mAP:6.2f}"
                 f"\n  diff   mAP {format_diff(out['diff_mAP'], with_ci=True)}"
                 f"   Rank-1 {format_diff(out['diff_rank1'], with_ci=True)}")
    print(line + f"\n  ({len(q)} queries, {len(g)} gallery, {time.time() - t0:.0f} s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    np.savez(args.out / f"{tag}_per_query.npz", **per_query)
    return out


def markdown(rows: list[dict], calib: bool) -> str:
    """Markdown table of all evaluated settings."""
    if calib:
        hdr = "| Backbone | Setting | Rank-1 Base / +Calib | mAP Base / +Calib | mAP diff [95% CI] | Rank-1 diff [95% CI] |"
        sep = "|---|---|---|---|---|---|"
        body = [f"| {r['backbone']} | {r['setting']} | {r['base']['rank1']:.1f} / {r['calib']['rank1']:.1f} "
                f"| {r['base']['mAP']:.1f} / {r['calib']['mAP']:.1f} | {format_diff(r['diff_mAP'], with_ci=True)} "
                f"| {format_diff(r['diff_rank1'], with_ci=True)} |" for r in rows]
    else:
        hdr, sep = "| Backbone | Setting | Rank-1 | mAP |", "|---|---|---|---|"
        body = [f"| {r['backbone']} | {r['setting']} | {r['base']['rank1']:.1f} | {r['base']['mAP']:.1f} |" for r in rows]
    note = "\n\n`*` = paired query-level bootstrap 95% CI excludes 0." if calib else ""
    return "\n".join([hdr, sep, *body]) + note + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", nargs="+", default=["clipreid"], choices=["clipreid", "personvit"])
    ap.add_argument("--train", nargs="+", default=["duke"], choices=["market", "duke"], help="training dataset(s) of the model")
    ap.add_argument("--eval", nargs="+", default=["market"], choices=["market", "duke"], help="evaluation dataset(s)")
    ap.add_argument("--calib", action="store_true", help="also evaluate + camera calibration and the bootstrap CI")
    ap.add_argument("--features", type=Path, default=Path("outputs/features"), help="feature cache directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/results"), help="result directory")
    ap.add_argument("--configs", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    ap.add_argument("--data-root", nargs=2, action="append", metavar=("DATASET", "PATH"), default=[],
                    help="override a dataset root, e.g. --data-root market /data/Market-1501-v15.09.15")
    ap.add_argument("--repo", type=Path, help="override the backbone repository checkout (extraction only)")
    ap.add_argument("--device", default="cuda", help="extraction device (cuda / cpu)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    datasets = {d: load_yaml(args.configs / f"{d}.yaml") for d in ("market", "duke")}
    for d, p in args.data_root:
        datasets[d]["root"] = p
    rows = []
    for name in args.backbone:
        bb = load_yaml(args.configs / "backbones" / f"{name}.yaml")
        if args.repo:
            bb["repo"] = str(args.repo)
        cells = [c for c in ORDER if c[0] in args.train and c[1] in args.eval]
        for train, eval_ in cells:
            rows.append(run_setting(args, bb, train, eval_, datasets))
    table = markdown(rows, args.calib)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("\n" + table + f"\nwrote {args.out}/summary.md, summary.json and one JSON per setting")


if __name__ == "__main__":
    main()
