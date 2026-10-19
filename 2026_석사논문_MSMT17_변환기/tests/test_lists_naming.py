"""List parsing and file naming."""

import pytest

from msmt17_converter.lists import parse_list_line, read_list, splits
from msmt17_converter.naming import camera_from_name, market_name, parse_market_name


def test_parse_list_line():
    e = parse_list_line("0000/0000_000_01_0303morning_0015_0.jpg 0\n")
    assert e.rel_path == "0000/0000_000_01_0303morning_0015_0.jpg" and e.pid == 0
    assert parse_list_line("   \n") is None
    with pytest.raises(ValueError):
        parse_list_line("only_one_field.jpg")


def test_read_list_keeps_order(mini_msmt):
    entries = read_list(mini_msmt / "list_gallery.txt")
    assert [e.pid for e in entries] == [0, 0, 1]


def test_splits_val_optional():
    assert [s.list_name for s in splits()] == ["list_query.txt", "list_gallery.txt", "list_train.txt"]
    assert splits(include_val=True)[-1].market_dir == "bounding_box_train"


def test_camera_is_one_based_and_unpadded():
    assert camera_from_name("0000_000_01_0303morning_0015_0.jpg") == 1
    assert camera_from_name("0001_003_15_0303noon_0100_1_ex.jpg") == 15
    with pytest.raises(ValueError):
        camera_from_name("bad.jpg")


def test_market_name_matches_reference_format():
    assert (market_name(0, "0000/0000_030_14_0303morning_0245_0.jpg")
            == "0000_c14_0000_030_14_0303morning_0245_0.jpg")
    assert (market_name(1040, "1040/1040_221_09_0302morning_0191_0.jpg")
            == "1040_c9_1040_221_09_0302morning_0191_0.jpg")


def test_parse_market_name_roundtrip():
    assert parse_market_name("0012_c2_0012_046_02_x.jpg") == (12, 2)
    with pytest.raises(ValueError):
        parse_market_name("0012_046_02_x.jpg")
