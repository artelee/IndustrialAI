import numpy as np
import pytest
from PIL import Image

from campose.render import (
    OUT_SIZE,
    REFERENCE_LAYOUT,
    Layout,
    draw_bodypose_builtin,
    get_draw_bodypose,
    load_display_rgb,
    measure_layout,
    place,
    render_rgb,
    save_like_reference,
)
from campose.synthetic import make_skeleton


def _pose(direction="front"):
    b, s = make_skeleton(direction)
    return b, s


def test_output_size_is_64_wide_128_high():
    assert OUT_SIZE == (64, 128)
    b, s = _pose()
    rgb = render_rgb(b, s, 64, 128, backend="builtin")
    assert rgb.shape == (128, 64, 3)  # numpy: H x W x C
    assert rgb.dtype == np.uint8


def test_saved_file_size(tmp_path):
    b, s = _pose("side")
    f = tmp_path / "1.jpg"
    save_like_reference(render_rgb(b, s, 64, 128, backend="builtin"), f)
    assert Image.open(f).size == (64, 128)  # PIL: (W, H)


def test_file_channel_order_is_bgr(tmp_path):
    # a pure-red patch in the RGB render must appear blue to an RGB reader (PIL)
    rgb = np.zeros((128, 64, 3), np.uint8)
    rgb[..., 0] = 200
    f = tmp_path / "c.png"
    save_like_reference(rgb, f)
    as_rgb = np.asarray(Image.open(f).convert("RGB"))
    assert as_rgb[..., 2].min() == 200 and as_rgb[..., 0].max() == 0
    # cv2-based loader returns the original array (what the generator sees)
    assert np.array_equal(load_display_rgb(f), rgb)


def test_rendered_colors_follow_openpose_palette_in_file(tmp_path):
    # only the right shoulder-neck limb (color [255, 0, 0] * 0.6 in RGB) and its joints
    b, s = _pose()
    s = np.zeros(18)
    s[[1, 2]] = 0.9
    rgb = render_rgb(b, s, 64, 128, backend="builtin")
    assert rgb[..., 0].sum() > rgb[..., 2].sum()          # red-dominant render
    f = tmp_path / "limb.png"
    save_like_reference(rgb, f)
    as_rgb = np.asarray(Image.open(f).convert("RGB")).astype(int)
    assert as_rgb[..., 2].sum() > as_rgb[..., 0].sum()    # stored BGR -> looks blue as RGB


def test_place_matches_reference_layout():
    b, s = _pose("front")
    q = place(b, s, 64, 128, canvas_h=320)
    v = s > 0.3
    assert q[v, 1].min() == pytest.approx(REFERENCE_LAYOUT.top)
    assert q[v, 1].max() == pytest.approx(REFERENCE_LAYOUT.bottom)
    assert (q[v, 0].min() + q[v, 0].max()) / 2 == pytest.approx(REFERENCE_LAYOUT.cx)


def test_place_wide_pose_is_limited_by_width():
    b, s = _pose("front")
    b = b.copy()
    b[:, 0] = (b[:, 0] - 0.5) * 6 + 0.5  # very wide pose
    q = place(b, s, 64, 128, canvas_h=320)
    v = s > 0.3
    assert (q[v, 0].max() - q[v, 0].min()) == pytest.approx(0.95)
    assert q[v, 1].min() == pytest.approx(REFERENCE_LAYOUT.top)


def test_builtin_draw_matches_controlnet_aux_if_available():
    ref = pytest.importorskip("controlnet_aux.dwpose.util")
    rng = np.random.RandomState(0)
    for _ in range(20):
        cand = rng.uniform(0.05, 0.95, (18, 2))
        sub = np.arange(18, dtype=float)[None]
        sub[0, rng.uniform(size=18) < 0.2] = -1
        a = ref.draw_bodypose(np.zeros((320, 160, 3), np.uint8), cand.copy(), sub.copy())
        c = draw_bodypose_builtin(np.zeros((320, 160, 3), np.uint8), cand.copy(), sub.copy())
        assert np.array_equal(a, c)


def test_backend_selection():
    assert get_draw_bodypose("builtin") is draw_bodypose_builtin
    assert callable(get_draw_bodypose("auto"))
    with pytest.raises(ValueError):
        get_draw_bodypose("nope")


def test_missing_joints_are_not_drawn():
    b, s = _pose("front")
    full = render_rgb(b, s, 64, 128, backend="builtin")
    s2 = s.copy()
    s2[[3, 4, 6, 7]] = 0.0  # drop both arms
    part = render_rgb(b, s2, 64, 128, backend="builtin")
    assert (part > 0).sum() < (full > 0).sum()


def test_measure_layout(tmp_path):
    for k in range(3):
        g = np.zeros((128, 64), np.uint8)
        g[10:120, 20 + k:40 + k] = 255
        Image.fromarray(g).convert("RGB").save(tmp_path / f"{k}.png")
    lay = measure_layout(sorted(tmp_path.glob("*.png")))
    assert isinstance(lay, Layout)
    assert lay.top == pytest.approx(10 / 128)
    assert lay.bottom == pytest.approx(119 / 128)
    assert lay.cx == pytest.approx((21 + 40) / 2 / 64)
