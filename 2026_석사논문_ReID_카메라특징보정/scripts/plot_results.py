#!/usr/bin/env python
"""Bar chart of Base vs. + camera calibration mAP (README figure).

Example:
    python scripts/plot_results.py --table docs/results_table.json --out docs/figures/calibration_map.png
    python scripts/plot_results.py --table outputs/results/summary.json --out outputs/calibration_map.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASE, CALIB = "#2a78d6", "#eb6834"
NAMES = {"clipreid": "CLIP-ReID", "personvit": "PersonViT"}


def load_table(path: Path) -> dict:
    """Read ``docs/results_table.json`` or a ``run_eval.py`` ``summary.json`` into one layout."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    settings: list[str] = []
    res: dict = {}
    for r in data:
        s = r["setting"].replace("->", "→")
        if s not in settings:
            settings.append(s)
        d = res.setdefault(NAMES.get(r["backbone"], r["backbone"]), {"base": [], "calib": []})
        d["base"].append(r["base"]["mAP"])
        d["calib"].append(r["calib"]["mAP"])
    return {"metric": "mAP (%)", "settings": settings, "results": res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=Path("docs/results_table.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/calibration_map.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = load_table(args.table)
    labels = [s.replace(" ", "\n", 1) for s in t["settings"]]
    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, len(t["results"]), figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (name, r) in zip(np.atleast_1d(axes), t["results"].items()):
        ax.set_facecolor(SURFACE)
        for off, key, color, lab in ((-w / 2, "base", BASE, "Base"), (w / 2, "calib", CALIB, "+ camera calibration")):
            bars = ax.bar(x + off, r[key], w - 0.03, color=color, label=lab, zorder=2)
            for b, v in zip(bars, r[key]):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8, color=MUTED)
        ax.set_title(name, color=INK, fontsize=12, loc="left")
        ax.set_xticks(x, labels, fontsize=9, color=INK)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=MUTED, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    np.atleast_1d(axes)[0].set_ylabel(t["metric"], color=MUTED)
    handles, labs = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Camera calibration helps on unseen cameras (cross-domain), not on trained ones",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
