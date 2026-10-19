"""Command line entry: ``python -m msmt17_converter --src <msmt17_v1> --dst <out>``."""

from __future__ import annotations

import argparse
import sys

from .convert import LINK_MODES, compare_names, execute, missing_sources, plan, scan_market, summarize


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the converter."""
    p = argparse.ArgumentParser(
        prog="msmt17_converter",
        description="Convert MSMT17 V1 (list_*.txt + train/ + test/) to the Market-1501 layout.")
    p.add_argument("--src", required=True, help="MSMT17 V1 root containing list_*.txt, train/, test/")
    p.add_argument("--dst", required=True, help="output root (query/, bounding_box_test/, bounding_box_train/)")
    p.add_argument("--include-val", action="store_true",
                   help="also put list_val.txt images into bounding_box_train (default: train list only)")
    p.add_argument("--link", choices=LINK_MODES, default="copy", help="how to place files (default: copy)")
    p.add_argument("--dry-run", action="store_true", help="plan and report only; write nothing")
    p.add_argument("--overwrite", action="store_true", help="replace files that already exist in --dst")
    p.add_argument("--compare", metavar="MARKET_ROOT",
                   help="compare planned file names with an existing Market-style tree")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the conversion; returns a process exit code."""
    args = build_parser().parse_args(argv)
    files = plan(args.src, include_val=args.include_val)

    print("planned:")
    for d, s in summarize(files).items():
        print("  " + s.line(d))

    missing = missing_sources(files)
    if missing:
        print(f"ERROR: {len(missing):,} source images not found, e.g. {missing[0]}", file=sys.stderr)
        return 1

    status = 0
    if args.compare:
        print(f"name comparison with {args.compare}:")
        for d, (not_found, extra) in compare_names(files, args.compare).items():
            print(f"  {d:<20} planned-not-found {not_found:>6,}  existing-not-planned {extra:>6,}")
            status |= int(bool(not_found or extra))

    if args.dry_run:
        print("dry run: nothing written")
        return status

    written, skipped = execute(files, args.dst, link=args.link, overwrite=args.overwrite)
    print(f"{args.link}: {written:,} written, {skipped:,} already present")
    print("verify (scan of --dst):")
    planned_counts = {d: s.images for d, s in summarize(files).items()}
    for d, s in scan_market(args.dst, list(planned_counts)).items():
        ok = "ok" if s.images == planned_counts[d] else "MISMATCH"
        print(f"  {s.line(d)}  [{ok}]")
        status |= int(ok != "ok")
    return status


if __name__ == "__main__":
    sys.exit(main())
