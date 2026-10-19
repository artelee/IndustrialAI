"""AP / Rank-1 against hand-computed examples."""
import numpy as np
import pytest

from reid_calib.evaluate import (average_precision, evaluate, evaluate_base_and_calibrated,
                                 evaluate_per_query, ranked_matches)


def test_average_precision_hand_example():
    # matches at ranks 1 and 3: (1/1 + 2/3) / 2
    assert average_precision(np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)
    # matches at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5
    assert average_precision(np.array([0, 1, 0, 1])) == pytest.approx(0.5)
    assert average_precision(np.array([0, 0])) == 0.0


def test_same_pid_same_camera_is_removed():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    gp = np.array([1, 2, 1, 1])
    gc = np.array([1, 2, 2, 3])  # first item: same pid and same camera as the query -> removed
    m = ranked_matches(scores, query_pid=1, query_cam=1, gallery_pid=gp, gallery_cam=gc)
    np.testing.assert_array_equal(m, [0, 1, 1])


def test_evaluate_hand_computed():
    # 1-D features: similarity = product. Query pid 1 (cam 1) ranks gallery by value.
    g = np.array([[0.9], [0.5], [0.3], [0.1]])
    gp = np.array([2, 1, 3, 1])
    gc = np.array([2, 2, 2, 3])
    q = np.array([[1.0], [-1.0]])
    qp = np.array([1, 2])
    qc = np.array([1, 1])
    res = evaluate_per_query(q, qp, qc, g, gp, gc)
    # query 0 order: [2, 1, 3, 1] -> matches [0,1,0,1] -> AP 0.5, R1 0
    # query 1 order (descending of -g): [1, 3, 1, 2] -> matches [0,0,0,1] -> AP 0.25, R1 0
    np.testing.assert_allclose(res.ap, [0.5, 0.25])
    np.testing.assert_array_equal(res.r1, [0, 0])
    r1, mAP = evaluate(q, qp, qc, g, gp, gc)
    assert (r1, mAP) == pytest.approx((0.0, 37.5))


def test_query_without_valid_match_is_skipped():
    g = np.array([[1.0], [0.5]])
    res = evaluate_per_query(np.array([[1.0], [1.0]]), np.array([1, 5]), np.array([1, 1]),
                             g, np.array([1, 1]), np.array([2, 1]))
    np.testing.assert_array_equal(res.valid, [True, False])
    assert res.mAP == pytest.approx(100.0)


def test_base_and_calibrated_same_queries():
    r = np.random.RandomState(0)
    qf, gf = r.randn(20, 8), r.randn(50, 8)
    qp, gp = r.randint(1, 5, 20), r.randint(1, 5, 50)
    qc, gc = r.randint(1, 3, 20), r.randint(1, 3, 50)
    out = evaluate_base_and_calibrated(qf, qp, qc, gf, gp, gc)
    assert set(out) == {"base", "calib"}
    np.testing.assert_array_equal(out["base"].valid, out["calib"].valid)
    assert set(evaluate_base_and_calibrated(qf, qp, qc, gf, gp, gc, calibrate=False)) == {"base"}
