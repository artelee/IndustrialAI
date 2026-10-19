import numpy as np
import pytest

from reidbench.metrics import average_precision, evaluate, evaluate_per_query, l2_normalize, ranked_matches


def test_ap_hand_computed():
    # hits at ranks 1, 3, 6: precisions 1/1, 2/3, 3/6 -> mean = (1 + 0.6667 + 0.5) / 3
    assert average_precision([1, 0, 1, 0, 0, 1]) == pytest.approx((1 + 2 / 3 + 0.5) / 3)
    assert average_precision([0, 1]) == pytest.approx(0.5)
    assert average_precision([1, 1, 1]) == pytest.approx(1.0)
    assert average_precision([0, 0, 0]) == 0.0


def test_toy_ranking_ap_and_rank1():
    # one query (pid 1, cam 1); scores define ranking g0 > g1 > g2 > g3
    gp = np.array([2, 1, 3, 1])
    gc = np.array([2, 2, 2, 3])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    m = ranked_matches(scores, 1, 1, gp, gc)
    assert m.tolist() == [0, 1, 0, 1]
    assert average_precision(m) == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_same_pid_same_cam_excluded():
    # gallery item 0 is the same person from the same camera -> must be ignored
    q = l2_normalize(np.array([[1.0, 0.0]]))
    g = l2_normalize(np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]]))
    gp, gc = np.array([1, 2, 1]), np.array([1, 2, 2])
    res = evaluate_per_query(q, np.array([1]), np.array([1]), g, gp, gc)
    # without exclusion Rank-1 would be 1; with exclusion ranking is [pid2, pid1] -> AP 0.5
    assert res.r1[0] == 0 and res.ap[0] == pytest.approx(0.5)
    # same pid in a different camera is kept
    r1, m = evaluate(q, np.array([1]), np.array([3]), g, gp, gc)
    assert r1 == 100.0 and m == pytest.approx(100 * (1 + 2 / 3) / 2)


def test_query_without_valid_match_is_invalid():
    q = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    g = l2_normalize(np.array([[1.0, 0.1], [0.1, 1.0]]))
    res = evaluate_per_query(q, np.array([1, 2]), np.array([1, 1]), g, np.array([1, 2]), np.array([1, 2]))
    assert res.valid.tolist() == [False, True]
    assert res.mAP == pytest.approx(100.0)


def test_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    q, g = l2_normalize(rng.normal(size=(30, 8))), l2_normalize(rng.normal(size=(50, 8)))
    qp, qc = rng.integers(1, 6, 30), rng.integers(1, 4, 30)
    gp, gc = rng.integers(1, 6, 50), rng.integers(1, 4, 50)
    a = evaluate_per_query(q, qp, qc, g, gp, gc, chunk=7)
    b = evaluate_per_query(q, qp, qc, g, gp, gc, chunk=1000)
    np.testing.assert_allclose(a.ap, b.ap)
    np.testing.assert_allclose(a.r1, b.r1)
