#!/usr/bin/env python
"""Draw the README figures from summary statistics only (no images).

Example:
    python scripts/make_readme_figures.py --stats docs/figures/figure_data.json --out docs/figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from walkdir.direction import PRESET_THRESHOLDS, classify_direction  # noqa: E402
from walkdir.plot import plot_direction_bars, plot_skeleton_examples, plot_variation_bars  # noqa: E402
from walkdir.pose import normalize_pose  # noqa: E402
from walkdir.synthetic import example_skeletons  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", default=str(ROOT / "docs" / "figures" / "figure_data.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    s = json.load(open(a.stats))
    out = Path(a.out)

    dist = {name: {int(c): fr for c, fr in cams.items()} for name, cams in s["direction_share"].items()}
    print(plot_direction_bars(dist, out / "direction_distribution.png"))

    var = {name: (v["intra"], v["inter"]) for name, v in s["variation"].items()}
    print(plot_variation_bars(var, out / "intra_vs_inter_variation.png"))

    q33, q66 = PRESET_THRESHOLDS["market"]
    examples = []
    for d, body, sc in example_skeletons():
        nb, vis = normalize_pose(body, sc)
        dx = nb[2, 0] - nb[5, 0]
        assert classify_direction(nb, vis, q33, q66) == d
        examples.append((d, nb, f"shoulder dx = {dx:+.2f}"))
    print(plot_skeleton_examples(
        examples, out / "skeleton_directions.png",
        title="Normalized skeletons (neck = origin, torso length = 1); ring = person's right shoulder"))


if __name__ == "__main__":
    main()
