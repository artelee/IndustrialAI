import numpy as np
import pytest

from walkdir.direction import (
    PRESET_THRESHOLDS, classify_direction, compute_thresholds, direction_from_dx,
)
from walkdir.pose import normalize_pose
from walkdir.synthetic import make_skeleton

Q33, Q66 = 1.0, 1.5


@pytest.mark.parametrize("dx,expected", [
    (0.0, "side"),
    (0.999, "side"),
    (1.0, "oblique"),      # lower bound inclusive
    (-1.0, "oblique"),
    (1.4999, "oblique"),
    (-1.5, "front"),       # right shoulder on image left -> facing camera
    (1.5, "back"),
    (-2.0, "front"),
    (2.0, "back"),
])
def test_boundaries(dx, expected):
    assert direction_from_dx(dx, Q33, Q66) == expected


def test_front_sign_flip():
    assert direction_from_dx(-2.0, Q33, Q66, front_sign=+1) == "back"


def test_unknown_without_both_shoulders():
    b = np.zeros((18, 2))
    v = np.ones(18, bool)
    v[5] = False
    assert classify_direction(b, v, Q33, Q66) == "unknown"


@pytest.mark.parametrize("d", ["front", "back", "oblique", "side"])
def test_synthetic_templates(d):
    body, sc = make_skeleton(d)
    b, v = normalize_pose(body, sc)
    assert classify_direction(b, v, *PRESET_THRESHOLDS["market"]) == d


def test_thresholds_are_tertiles():
    q33, q66 = compute_thresholds(np.arange(1, 301, dtype=float))
    assert q33 == pytest.approx(np.percentile(np.arange(1, 301), 100 / 3))
    assert q33 < q66
    with pytest.raises(ValueError):
        compute_thresholds([])
