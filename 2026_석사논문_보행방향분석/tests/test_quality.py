import numpy as np
import pytest

from walkdir.quality import REASONS, QualityConfig, filter_and_normalize, filter_reason
from walkdir.synthetic import make_skeleton


def good():
    body, sc = make_skeleton("front")
    return body.copy(), sc.copy()


def test_pass():
    body, sc = good()
    res, reason = filter_and_normalize(body, sc)
    assert reason is None and res is not None
    assert np.allclose(res[0][1], 0.0)


def test_nodet():
    assert filter_and_normalize(None, None) == (None, "nodet")


def test_core_missing_shoulder():
    body, sc = good()
    sc[5] = 0.1
    assert filter_reason(body, sc) == "core"


def test_core_too_few_joints():
    body, sc = good()
    sc[[0, 3, 4, 6, 7, 16, 17]] = 0.0  # 11 visible joints left
    assert filter_reason(body, sc) == "core"


def test_halfbody_no_ankle():
    body, sc = good()
    sc[[10, 13]] = 0.0
    assert filter_reason(body, sc) == "halfbody"


def test_halfbody_no_head():
    body, sc = good()
    sc[[0, 14, 15]] = 0.0
    assert filter_reason(body, sc) == "halfbody"


def test_small():
    body, sc = good()
    body[:, 1] = 0.5 + (body[:, 1] - 0.5) * 0.5  # height 0.8 -> 0.4
    assert filter_reason(body, sc) == "small"


def test_limb():
    body, sc = good()
    body[10, 0] += 0.6  # right ankle flies far sideways
    assert filter_reason(body, sc) == "limb"


def test_core_mode_skips_strict_checks():
    body, sc = good()
    sc[[10, 13]] = 0.0
    body[4, 0] += 0.6
    assert filter_reason(body, sc, QualityConfig(mode="core")) is None


def test_reason_order_and_bad_mode():
    assert REASONS == ("nodet", "core", "halfbody", "small", "limb", "unknown")
    body, sc = good()
    with pytest.raises(ValueError):
        filter_and_normalize(body, sc, QualityConfig(mode="nope"))
