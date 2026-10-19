import numpy as np

from walkdir.pose import NECK, R_HIP, normalize_pose, pose_vector
from walkdir.synthetic import make_skeleton


def test_neck_at_origin_and_unit_torso():
    body, sc = make_skeleton("front")
    body = body * 0.7 + np.array([0.13, -0.05])  # arbitrary shift and scale
    b, v = normalize_pose(body, sc)
    assert np.allclose(b[NECK], 0.0)
    assert np.isclose(np.linalg.norm(b[R_HIP]), 1.0)
    assert v.all()


def test_invisible_joints_zeroed_and_left_hip_fallback():
    body, sc = make_skeleton("front")
    sc = sc.copy()
    sc[[R_HIP, 3]] = 0.0
    b, v = normalize_pose(body, sc)
    assert not v[R_HIP] and np.allclose(b[[R_HIP, 3]], 0.0)
    assert np.isclose(np.linalg.norm(b[11]), 1.0)  # left hip used as torso reference
    assert np.allclose(b[NECK], 0.0)


def test_normalization_is_translation_and_scale_invariant():
    body, sc = make_skeleton("oblique")
    b1, _ = normalize_pose(body, sc)
    b2, _ = normalize_pose(body * 3.0 + 5.0, sc)
    assert np.allclose(b1, b2)
    assert pose_vector(b1).shape == (36,)
