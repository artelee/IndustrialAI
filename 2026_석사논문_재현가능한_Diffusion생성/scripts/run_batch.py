#!/usr/bin/env python
"""Run Pose2ID IPG over every set in a sets YAML, reproducibly and resumably.

Steps: pre-flight checks (seed patch, inputs, exactly N poses) -> per set:
optionally delete broken outputs -> compute the remaining (not yet generated)
inputs in sorted order -> stage them in a temp folder -> call IPG
``inference.py`` in a subprocess. Re-running the same command resumes.

Example:
    python scripts/run_batch.py --config configs/sets.yaml --dry-run
    python scripts/run_batch.py --config configs/sets.yaml --sets mkt_std duke_std --repair
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reprogen.batch import build_command, list_inputs, preflight, remaining_inputs, run_ipg, stage_inputs  # noqa: E402
from reprogen.sets import load_config  # noqa: E402
from reprogen.verify import find_bad_images, remove_bad_images  # noqa: E402


def log(msg: str) -> None:
    """Print a timestamped line."""
    print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/sets.yaml", help="sets YAML")
    ap.add_argument("--sets", nargs="*", help="run only these set names (default: all, in file order)")
    ap.add_argument("--python", help="override ipg.python (interpreter of the Pose2ID env)")
    ap.add_argument("--ipg-dir", help="override ipg.ipg_dir (Pose2ID/IPG checkout)")
    ap.add_argument("--ckpt-dir", help="override ipg.ckpt_dir (IPG pretrained weights)")
    ap.add_argument("--output-root", help="override output_root")
    ap.add_argument("--repair", action="store_true",
                    help="delete unreadable/truncated outputs before computing what remains")
    ap.add_argument("--skip-preflight", action="store_true", help="do not abort on pre-flight problems")
    ap.add_argument("--dry-run", action="store_true", help="run checks and print the plan; do not call IPG")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ipg = cfg.ipg
    if args.python:
        ipg = replace(ipg, python=args.python)
    if args.ipg_dir:
        ipg = replace(ipg, ipg_dir=Path(args.ipg_dir))
    if args.ckpt_dir:
        ipg = replace(ipg, ckpt_dir=Path(args.ckpt_dir))
    cfg = replace(cfg, ipg=ipg)
    if args.output_root:
        cfg = replace(cfg, output_root=Path(args.output_root), staging_root=Path(args.output_root) / "_staging")

    sets = cfg.sets
    if args.sets:
        unknown = sorted(set(args.sets) - {s.name for s in sets})
        if unknown:
            log(f"unknown set names: {unknown}")
            return 2
        sets = [s for s in sets if s.name in set(args.sets)]

    log(f"pre-flight: {len(sets)} sets, IPG at {cfg.ipg.ipg_dir}")
    problems = preflight(cfg, sets)
    for p in problems:
        log(f"  PROBLEM {p}")
    if problems and not args.skip_preflight:
        log("pre-flight failed; aborting")
        return 1
    if not problems:
        log(f"pre-flight passed: {len(sets)} sets")

    failed = []
    for s in sets:
        out_dir = cfg.output_root / s.name
        if args.repair and out_dir.is_dir():
            bad = find_bad_images(out_dir, cfg.input_glob, full_decode=True)
            for b in bad:
                log(f"  broken output: {b.path.name} ({b.reason})")
            if bad and not args.dry_run:
                log(f"  removed {remove_bad_images(bad)} broken files")
        inputs = list_inputs(s.ref_dir, cfg.input_glob)
        todo = remaining_inputs(inputs, out_dir)
        eta_h = len(todo) * cfg.seconds_per_image / 3600
        log(f"{s.name}: done {len(inputs) - len(todo)} / {len(inputs)}, remaining {len(todo)} (~{eta_h:.1f} h)")
        if not todo:
            continue
        staging = cfg.staging_root / s.name
        cmd = build_command(cfg, s, staging, out_dir)
        if args.dry_run:
            log(f"  [dry-run] first: {todo[0].name}  last: {todo[-1].name}")
            log(f"  [dry-run] cwd={cfg.ipg.ipg_dir}  cmd: {' '.join(cmd)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        stage_inputs(todo, staging)
        rc = run_ipg(cmd, cwd=cfg.ipg.ipg_dir, log=lambda line: print("    " + line, flush=True))
        left = len(remaining_inputs(inputs, out_dir))
        log(f"{s.name} finished (exit {rc}): {len(inputs) - left} / {len(inputs)}")
        if rc != 0 or left:
            failed.append(s.name)
        else:
            shutil.rmtree(staging, ignore_errors=True)
    if args.dry_run:
        log("dry-run complete (nothing generated)")
        return 0
    log("all sets done" if not failed else f"incomplete sets: {failed} (re-run to resume)")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
