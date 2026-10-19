#!/usr/bin/env python
"""Cut IPG panels into individual generated images.

Each panel ``<name>.jpg`` (ref | pose_0 | gen_0 | ... | pose_7 | gen_7) yields
``<name>_g0.jpg`` ... ``<name>_g7.jpg`` in the output folder; the k-th image is
the box ``((2k+2)*cw, 0, (2k+3)*cw, H)`` with ``cw = W // (2N+1)``.

Example:
    python scripts/crop_panels.py outputs/gen_seed/mkt_std outputs/crops/mkt_std
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reprogen.crop import crop_panel_file  # noqa: E402
from reprogen.verify import check_image  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel_dir", help="folder of IPG panels")
    ap.add_argument("out_dir", help="folder for the cropped images")
    ap.add_argument("--num-poses", type=int, default=8)
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--suffix", default="_g{k}", help="name suffix, {k} = pose index")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    panels = sorted(p for p in Path(args.panel_dir).glob(args.pattern) if p.is_file())
    n_ok, skipped = 0, []
    for p in panels:
        reason = check_image(p, full_decode=True, num_poses=args.num_poses)
        if reason:
            skipped.append((p.name, reason))
            continue
        crop_panel_file(p, args.out_dir, args.num_poses, args.suffix, args.overwrite)
        n_ok += 1
    print(f"{n_ok} panels -> {n_ok * args.num_poses} images in {args.out_dir}; skipped {len(skipped)}")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
