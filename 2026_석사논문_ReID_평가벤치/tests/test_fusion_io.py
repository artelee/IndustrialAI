import numpy as np
import pytest

from reidbench.conditions import load_experiment, run_condition
from reidbench.fusion import assign_by_identity, camera_mean_subtract, fuse_extra, set_sums
from reidbench.io import FeatureSet, FeatureSplit, cache_path, load_split, save_set, save_split
from reidbench.metrics import l2_normalize


def test_camera_mean_subtract_zero_mean_per_camera_before_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4)) + 5
    cam = np.repeat([1, 2], 10)
    y = camera_mean_subtract(x, cam)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1, rtol=1e-6)
    z = x.copy()
    for c in (1, 2):
        z[cam == c] -= x[cam == c].mean(0)
    np.testing.assert_allclose(y, l2_normalize(z))


def test_fuse_extra_formula():
    g = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    extra = np.array([[[0.0, 2.0]] * 8])          # one reference, 8 identical extra features
    sums = set_sums(extra)                          # = [[0, 8]]
    out = fuse_extra(g, sums, np.array([0, -1]), eta=0.5, k=8)
    np.testing.assert_allclose(out[0], l2_normalize(np.array([1.0, 0.5])))
    np.testing.assert_allclose(out[1], g[1])        # unassigned item unchanged


def test_assign_by_identity():
    assert assign_by_identity(np.array([3, 5, 7]), np.array([5, 3])).tolist() == [1, 0, -1]


def test_io_roundtrip_and_conditions(tmp_path):
    rng = np.random.default_rng(0)
    ids, d = 6, 8
    ident = rng.normal(size=(ids, d))
    q = FeatureSplit(feat=(ident + 0.1 * rng.normal(size=(ids, d))).astype(np.float32), pid=np.arange(1, ids + 1),
                     cam=np.ones(ids, int), name=np.array([f"q{i}" for i in range(ids)]))
    g = FeatureSplit(feat=(ident + 0.1 * rng.normal(size=(ids, d))).astype(np.float32), pid=np.arange(1, ids + 1),
                     cam=np.full(ids, 2), name=np.array([f"g{i}" for i in range(ids)]))
    save_split(cache_path(tmp_path, "bb", "a", "b", "query"), q)
    save_split(cache_path(tmp_path, "bb", "a", "b", "gal"), g)
    s = FeatureSet(feat=np.repeat(ident[:, None], 8, 1).astype(np.float32), name=g.name)
    save_set(cache_path(tmp_path, "bb", "a", "b", "common"), s)
    save_set(cache_path(tmp_path, "bb", "a", "b", "percam_c1"), s)
    loaded = load_split(cache_path(tmp_path, "bb", "a", "b", "query"))
    np.testing.assert_array_equal(loaded.pid, q.pid)
    exp = load_experiment(tmp_path, "bb", "a", "b", [1], "common", "percam_c{cam}")
    assert exp.k == 8 and exp.assign_accuracy == pytest.approx(100.0)
    for key in ("B", "S", "T"):
        r1, ap = run_condition(exp, key, calibrate=False)
        assert ap.mean() == pytest.approx(1.0)
