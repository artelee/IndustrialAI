"""Planning, conversion modes, dry run and verification on the miniature tree."""

import os

import pytest

from msmt17_converter.__main__ import main
from msmt17_converter.convert import compare_names, execute, missing_sources, plan, scan_market, summarize


def test_plan_counts_and_cameras(mini_msmt):
    stats = summarize(plan(mini_msmt))
    assert {d: s.images for d, s in stats.items()} == {
        "query": 2, "bounding_box_test": 3, "bounding_box_train": 2}
    assert sorted(stats["query"].cams) == [1, 15]
    assert summarize(plan(mini_msmt, include_val=True))["bounding_box_train"].images == 3


def test_plan_source_paths(mini_msmt):
    files = plan(mini_msmt)
    assert not missing_sources(files)
    assert all(f.src.parent.parent.name == ("train" if f.market_dir == "bounding_box_train" else "test")
               for f in files)


def test_duplicate_target_raises(mini_msmt):
    with open(mini_msmt / "list_query.txt", "a") as f:
        f.write("0000/0000_000_01_0303morning_0015_0.jpg 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        plan(mini_msmt)


@pytest.mark.parametrize("link", ["copy", "hardlink", "symlink"])
def test_execute_modes(mini_msmt, tmp_path, link):
    dst = tmp_path / "out"
    files = plan(mini_msmt)
    assert execute(files, dst, link=link) == (7, 0)
    target = dst / "query" / "0001_c15_0001_003_15_0303noon_0100_1_ex.jpg"
    assert target.read_bytes() == b"\xff\xd8\xff\xd9"
    assert target.is_symlink() == (link == "symlink")
    if link == "hardlink":
        assert os.stat(target).st_nlink == 2
    assert execute(files, dst, link=link) == (0, 7)
    assert execute(files, dst, link=link, overwrite=True) == (7, 0)
    scanned = scan_market(dst)
    assert {d: s.images for d, s in scanned.items()} == {
        "query": 2, "bounding_box_test": 3, "bounding_box_train": 2}
    assert all(v == (0, 0) for v in compare_names(files, dst).values())


def test_cli_dry_run_writes_nothing(mini_msmt, tmp_path, capsys):
    dst = tmp_path / "out"
    assert main(["--src", str(mini_msmt), "--dst", str(dst), "--dry-run"]) == 0
    assert not dst.exists()
    assert "cameras 1-15" in capsys.readouterr().out


def test_cli_copy_and_verify(mini_msmt, tmp_path, capsys):
    dst = tmp_path / "out"
    assert main(["--src", str(mini_msmt), "--dst", str(dst), "--include-val"]) == 0
    out = capsys.readouterr().out
    assert "MISMATCH" not in out and "bounding_box_train" in out
    assert len(os.listdir(dst / "bounding_box_train")) == 3


def test_cli_compare_reports_mismatch(mini_msmt, tmp_path, capsys):
    ref = tmp_path / "ref"
    (ref / "query").mkdir(parents=True)
    (ref / "query" / "0000_c01_wrong.jpg").write_bytes(b"")
    assert main(["--src", str(mini_msmt), "--dst", str(tmp_path / "o"), "--dry-run", "--compare", str(ref)]) == 1


def test_cli_missing_source(mini_msmt, tmp_path):
    (mini_msmt / "test" / "0000" / "0000_008_01_0303morning_0019_2.jpg").unlink()
    assert main(["--src", str(mini_msmt), "--dst", str(tmp_path / "o"), "--dry-run"]) == 1
