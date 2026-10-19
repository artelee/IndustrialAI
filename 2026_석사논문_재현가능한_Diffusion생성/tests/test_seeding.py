import re
import zlib

from reprogen.seeding import BASE_SEED, SEED_MASK, image_seed, is_seed_patched


def _patch_seed_expr(repo_root):
    """Expression passed to generator.manual_seed in the shipped patch."""
    text = (repo_root / "patches" / "ipg_seed.patch").read_text()
    added = [l[1:] for l in text.splitlines() if l.startswith("+") and "manual_seed" in l]
    assert len(added) == 1
    m = re.search(r"manual_seed\((.*)\)\s*$", added[0])
    assert m
    return m.group(1)


def test_same_name_same_seed():
    name = "0001_c1s1_001051_00.jpg"
    assert image_seed(name) == image_seed(name)
    assert image_seed(name) == image_seed("/some/other/dir/" + name)


def test_different_names_differ():
    names = [f"{i:04d}_c1s1_000000_00.jpg" for i in range(200)]
    assert len({image_seed(n) for n in names}) == len(names)


def test_formula_values():
    name = "0002_c2s1_000301_01.jpg"
    assert image_seed(name) == 12580 + (zlib.crc32(name.encode()) & 0x7FFFFFFF)
    assert image_seed(name, base=0) == zlib.crc32(name.encode()) & SEED_MASK
    assert BASE_SEED <= image_seed(name) < BASE_SEED + 2**31


def test_matches_patch_expression(repo_root):
    expr = _patch_seed_expr(repo_root)
    for name in ["0001_c1s1_001051_00.jpg", "a.jpg", "ünïcode.png", "0499_c3_f0161281.jpg"]:
        assert eval(expr, {"zlib": zlib}, {"ref_name": name}) == image_seed(name)


def test_is_seed_patched(tmp_path):
    f = tmp_path / "inference.py"
    f.write_text("generator.manual_seed(cfg.seed)\n")
    assert not is_seed_patched(f)
    f.write_text("generator.manual_seed(12580 + (zlib.crc32(ref_name.encode()) & 0x7fffffff))\n")
    assert is_seed_patched(f)
    assert not is_seed_patched(tmp_path / "missing.py")
