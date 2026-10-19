#!/usr/bin/env python
"""Find (and optionally delete) broken images in one or more output folders.

Example:
    python scripts/verify_outputs.py outputs/gen_seed/mkt_std --num-poses 8
    python scripts/verify_outputs.py outputs/gen_seed/* --delete
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reprogen.verify import find_bad_images, remove_bad_images  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="folders to check")
    ap.add_argument("--pattern", default="*.jpg", help="file glob (default *.jpg)")
    ap.add_argument("--quick", action="store_true", help="only PIL verify(), no full decode")
    ap.add_argument("--num-poses", type=int, help="also require a (2N+1)-column panel width")
    ap.add_argument("--delete", action="store_true", help="delete the broken files")
    args = ap.parse_args(argv)

    total_bad = 0
    for d in args.dirs:
        n_files = len([p for p in Path(d).glob(args.pattern) if p.is_file()])
        bad = find_bad_images(d, args.pattern, full_decode=not args.quick, num_poses=args.num_poses)
        print(f"{d}: {n_files} files, {len(bad)} broken")
        for b in bad:
            print(f"  {b.path.name}: {b.reason}")
        if bad and args.delete:
            print(f"  deleted {remove_bad_images(bad)}")
        total_bad += len(bad)
    return 1 if total_bad and not args.delete else 0


if __name__ == "__main__":
    sys.exit(main())
