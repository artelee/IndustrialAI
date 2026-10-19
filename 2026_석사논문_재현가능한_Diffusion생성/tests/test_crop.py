import pytest
from PIL import Image

from reprogen.crop import (
    column_width,
    crop_generated,
    crop_panel_file,
    generated_boxes,
    is_panel_width,
    pose_boxes,
    reference_box,
)


def test_boxes_stock_ipg_panel():
    # stock IPG: 128x256 columns, 8 poses -> 17 columns
    boxes = generated_boxes(17 * 128, 256)
    assert len(boxes) == 8
    assert boxes[0] == (256, 0, 384, 256)
    assert boxes[7] == (16 * 128, 0, 17 * 128, 256)
    assert all(b[2] - b[0] == 128 for b in boxes)
    assert boxes == [((2 * k + 2) * 128, 0, (2 * k + 3) * 128, 256) for k in range(8)]
    assert pose_boxes(17 * 128, 256)[0] == (128, 0, 256, 256)
    assert reference_box(17 * 128, 256) == (0, 0, 128, 256)


def test_boxes_do_not_overlap_and_cover():
    w, h, n = 17 * 10, 20, 8
    cols = [reference_box(w, h, n)] + [b for pair in zip(pose_boxes(w, h, n), generated_boxes(w, h, n)) for b in pair]
    assert [c[0] for c in cols] == [i * 10 for i in range(17)]
    assert cols[-1][2] == w


def test_other_pose_counts():
    assert column_width(5 * 64, 2) == 64
    assert generated_boxes(5 * 64, 128, 2) == [(128, 0, 192, 128), (256, 0, 320, 128)]
    assert is_panel_width(17 * 128) and not is_panel_width(17 * 128 + 1)
    with pytest.raises(ValueError):
        column_width(10, 8)


def test_crop_pixels(tmp_path):
    cw, h = 4, 6
    panel = Image.new("RGB", (17 * cw, h))
    for c in range(17):
        panel.paste((c * 10, 0, 0), (c * cw, 0, (c + 1) * cw, h))
    crops = crop_generated(panel)
    assert [im.getpixel((0, 0))[0] for im in crops] == [(2 * k + 2) * 10 for k in range(8)]
    p = tmp_path / "x.png"
    panel.save(p)
    outs = crop_panel_file(p, tmp_path / "out")
    assert [o.name for o in outs] == [f"x_g{k}.png" for k in range(8)]
    assert Image.open(outs[3]).getpixel((1, 1)) == (80, 0, 0)
