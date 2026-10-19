"""Camera calibration: per-camera mean is zero before the final normalisation."""
import numpy as np

from reid_calib.calibrate import (calibrate_query_gallery, camera_mean_subtract, camera_means,
                                  center_by_camera)
from reid_calib.evaluate import l2_normalize


def _toy(seed: int = 0, n: int = 60, d: int = 16, n_cam: int = 3):
    r = np.random.RandomState(seed)
    cam = r.randint(1, n_cam + 1, n)
    bias = r.randn(n_cam + 1, d) * 3.0
    return r.randn(n, d) + bias[cam], cam


def test_per_camera_mean_zero_before_normalisation():
    x, cam = _toy()
    y = center_by_camera(l2_normalize(x), cam)
    for c in np.unique(cam):
        np.testing.assert_allclose(y[cam == c].mean(0), 0.0, atol=1e-12)


def test_output_is_unit_norm_and_matches_formula():
    x, cam = _toy(1)
    f = l2_normalize(x)
    mu = camera_means(f, cam)
    expected = l2_normalize(np.stack([f[i] - mu[int(cam[i])] for i in range(len(f))]))
    out = camera_mean_subtract(f, cam)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_reference_implementation():
    """Same result as the reference loop ``y[cam == c] -= x[cam == c].mean(0); N(y)``."""
    x, cam = _toy(2)
    f = l2_normalize(x)
    y = f.copy()
    for c in np.unique(cam):
        y[cam == c] -= f[cam == c].mean(0)
    np.testing.assert_array_equal(camera_mean_subtract(f, cam), l2_normalize(y))


def test_query_and_gallery_use_their_own_means():
    xq, cq = _toy(3, n=30)
    xg, cg = _toy(4, n=80)
    q, g = calibrate_query_gallery(xq, cq, xg, cg)
    np.testing.assert_allclose(q, camera_mean_subtract(l2_normalize(xq), cq))
    np.testing.assert_allclose(g, camera_mean_subtract(l2_normalize(xg), cg))
    # changing the gallery must not change the calibrated queries
    q2, _ = calibrate_query_gallery(xq, cq, xg * 2 + 1, cg)
    np.testing.assert_array_equal(q, q2)


def test_removes_camera_bias_for_retrieval():
    """With a strong camera offset, calibration restores cross-camera matches."""
    r = np.random.RandomState(5)
    ident = r.randn(10, 32)
    off = {1: r.randn(32) * 5, 2: r.randn(32) * 5}
    q = np.stack([ident[i] + off[1] + 0.1 * r.randn(32) for i in range(10)])
    g = np.stack([ident[i] + off[2] + 0.1 * r.randn(32) for i in range(10)])
    qc, gc = np.ones(10, int), np.full(10, 2)
    qn, gn = calibrate_query_gallery(q, qc, g, gc)
    assert ((qn @ gn.T).argmax(1) == np.arange(10)).all()
