"""Markdown and JSON report writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .bootstrap import format_diff

__all__ = ["markdown_table", "write_json", "write_text", "final_table_md", "eta_table_md", "tag_table_md"]

STAR_NOTE = "\\* = paired query-level bootstrap 95% CI excludes 0."


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines)


def write_json(path: str | Path, obj: object) -> None:
    """Write UTF-8 JSON (creates parent dirs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text (creates parent dirs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def final_table_md(results: dict, row_labels: Sequence[str], diff_labels: Sequence[str],
                   columns: Sequence[str], title: str = "") -> str:
    """Rows = conditions (``mAP / R1``), columns = experiments; diff rows with stars.

    ``results[col][label]`` is ``[r1, mAP]`` for condition rows and a CI dict
    for diff rows.
    """
    rows = [[lab] + [f"{results[c][lab][1]:.1f} / {results[c][lab][0]:.1f}" for c in columns]
            for lab in row_labels if all(lab in results[c] for c in columns)]
    rows += [[lab] + [format_diff(results[c][lab]) for c in columns]
             for lab in diff_labels if all(lab in results[c] for c in columns)]
    head = f"### {title}\n\n" if title else ""
    return head + markdown_table(["Condition (mAP / R1 %)"] + list(columns), rows) + "\n\n" + STAR_NOTE + "\n"


def eta_table_md(cells: dict, etas: Sequence[float], columns: Sequence[str], title: str = "") -> str:
    """Rows = eta, columns = experiments, cell = ``common / per-camera / diff``.

    ``cells[(col, eta)]`` is a dict with keys common, per_camera, d.
    """
    rows = []
    for eta in etas:
        rows.append([f"{eta:g}"] + [
            f"{cells[(c, eta)]['common']:.1f} / {cells[(c, eta)]['per_camera']:.1f} / {format_diff(cells[(c, eta)]['d'])}"
            for c in columns])
    head = f"### {title}\n\n" if title else ""
    return head + markdown_table(["eta"] + list(columns), rows) + "\n\n" + STAR_NOTE + "\n"


def tag_table_md(cells: dict, tags: Sequence[str], columns: Sequence[str], title: str = "") -> str:
    """Rows = query tags, columns = experiments, cell = ``diff (n=..)``.

    ``cells[col][tag]`` is a :func:`reidbench.breakdown.paired_by_tag` entry.
    """
    rows = [[t] + [f"{format_diff(cells[c][t]['d'])} (n={cells[c][t]['n']})" for c in columns] for t in tags]
    head = f"### {title}\n\n" if title else ""
    return head + markdown_table(["Query tag"] + list(columns), rows) + "\n\n" + STAR_NOTE + "\n"
