import numpy as np
import pytest

from reidbench.breakdown import paired_by_tag, per_camera, weighted_mean
from reidbench.metrics import evaluate_per_query, l2_normalize


def test_per_camera_weighted_mean_equals_overall():
    rng = np.random.default_rng(0)
    q, g = l2_normalize(rng.normal(size=(120, 16))), l2_normalize(rng.normal(size=(300, 16)))
    qp, qc = rng.integers(1, 20, 120), rng.integers(1, 7, 120)
    gp, gc = rng.integers(1, 20, 300), rng.integers(1, 7, 300)
    res = evaluate_per_query(q, qp, qc, g, gp, gc)
    cams = per_camera(res.ap[res.valid], qc[res.valid])
    assert sum(v["n"] for v in cams.values()) == res.valid.sum()
    assert weighted_mean(cams) == pytest.approx(res.mAP)
    # unweighted mean of camera means is generally different
    assert np.mean([v["mean"] for v in cams.values()]) != pytest.approx(res.mAP, abs=1e-9)


def test_paired_by_tag():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=90)
    b = a.copy()
    tags = np.array(["front", "back", "side"] * 30)
    b[tags == "front"] += 0.1
    out = paired_by_tag(a, b, tags, ["front", "back", "side"], rng=0)
    assert list(out) == ["front", "back", "side", "all"]
    assert out["front"]["d"]["sig"] and out["front"]["d"]["mean"] == pytest.approx(10.0)
    assert not out["back"]["d"]["sig"]
    assert out["all"]["n"] == 90
    assert paired_by_tag(a, b, tags, ["front", "back", "side"], rng=0) == out
