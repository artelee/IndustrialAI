import random

from PIL import Image

from reprogen.verify import check_image, find_bad_images, remove_bad_images


def _noise(path, size=(17 * 4, 8)):
    rng = random.Random(0)
    Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3))).save(path, quality=95)


def test_detects_broken_files(tmp_path):
    good = tmp_path / "good.jpg"
    _noise(good)
    Image.new("RGB", (17 * 4, 8), (10, 200, 30)).save(tmp_path / "solid.jpg")
    (tmp_path / "empty.jpg").write_bytes(b"")
    (tmp_path / "text.jpg").write_text("not an image")
    data = good.read_bytes()
    (tmp_path / "truncated.jpg").write_bytes(data[: len(data) // 2])

    assert check_image(good) is None
    bad = find_bad_images(tmp_path)
    assert sorted(b.path.name for b in bad) == ["empty.jpg", "text.jpg", "truncated.jpg"]
    assert remove_bad_images(bad) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["good.jpg", "solid.jpg"]
    assert find_bad_images(tmp_path) == []


def test_panel_width_check(tmp_path):
    ok = tmp_path / "ok.png"
    Image.new("RGB", (17 * 4, 8)).save(ok)
    odd = tmp_path / "odd.png"
    Image.new("RGB", (17 * 4 + 3, 8)).save(odd)
    assert check_image(ok, num_poses=8) is None
    assert "panel" in check_image(odd, num_poses=8)
    assert check_image(odd) is None
