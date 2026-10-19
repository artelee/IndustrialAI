import numpy as np
import pytest

from campose.compare import PoseRow, make_comparison_figure, row_from_files
from campose.data import detection_size, parse_duke, parse_market
from campose.render import render_rgb, save_like_reference
from campose.synthetic import make_skeleton


def test_comparison_figure(tmp_path):
    paths, dirs = [], []
    for i, d in enumerate(["front", "back", "oblique", "side"], 1):
        b, s = make_skeleton(d)
        p = tmp_path / f"{i}.jpg"
        save_like_reference(render_rgb(b, s, 64, 128, backend="builtin"), p)
        paths.append(p)
        dirs.append(d)
    rows = [row_from_files("common", paths, dirs),
            PoseRow("camera", [np.zeros((128, 64, 3), np.uint8)] * 2, ["side", "side"])]
    out = tmp_path / "fig.png"
    make_comparison_figure(rows, out)
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError):
        row_from_files("x", paths, dirs[:2])


def test_parsers_and_detection_size():
    assert parse_market("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_duke("0002_c7_f0044158.jpg") == (2, 7)
    assert detection_size(64, 128) == (512, 1024)
    assert detection_size(100, 250) == (512, 1280)
