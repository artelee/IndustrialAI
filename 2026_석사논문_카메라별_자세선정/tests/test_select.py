import numpy as np

from campose.select import (
    build_records,
    kmeans_representatives,
    select_camera_poses,
    select_representatives,
)
from campose.synthetic import make_camera_samples, make_skeleton

Q33, Q66 = 1.076, 1.351  # Market thresholds


def test_kmeans_returns_actual_members():
    rng = np.random.RandomState(0)
    X = np.concatenate([rng.normal(m, 0.1, (40, 36)) for m in (0.0, 1.0, 2.0)])
    picks = kmeans_representatives(X, 3)
    assert len(picks) == 3
    # indices of real rows (not synthetic cluster centers)
    assert all(isinstance(p, int) and 0 <= p < len(X) for p in picks)
    # one representative per well-separated cluster
    assert sorted(p // 40 for p in picks) == [0, 1, 2]


def test_kmeans_small_set_returns_all():
    X = np.arange(12, dtype=float).reshape(3, 4)
    assert kmeans_representatives(X, 5) == [0, 1, 2]
    assert kmeans_representatives(X, 0) == []


def test_select_representatives_respects_allocation_and_labels():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(60, 36))
    labels = ["front"] * 10 + ["back"] * 10 + ["oblique"] * 20 + ["side"] * 20
    alloc = {"front": 1, "back": 1, "oblique": 3, "side": 3}
    chosen = select_representatives(X, labels, alloc)
    assert len(chosen) == 8
    for idx, d in chosen:
        assert labels[idx] == d
    assert [d for _, d in chosen] == ["front", "back"] + ["oblique"] * 3 + ["side"] * 3


def test_build_records_filters_and_tags():
    good = make_camera_samples({"front": 1.0}, n=5, seed=0)
    b, s = make_skeleton("side")
    s = s.copy()
    s[[10, 13]] = 0.0  # no ankle -> halfbody
    bad = [dict(src="x.jpg", body=b.tolist(), score=s.tolist(), image_size=[64, 128]),
           dict(src="y.jpg", body=None, score=None, image_size=[64, 128])]
    recs, drop = build_records(good + bad, Q33, Q66)
    assert len(recs) == 5
    assert all(r.direction == "front" for r in recs)
    assert drop["halfbody"] == 1 and drop["nodet"] == 1


def test_select_camera_poses_end_to_end():
    samples = make_camera_samples({"front": 0.45, "back": 0.45, "oblique": 0.1}, n=200, seed=3)
    res = select_camera_poses(samples, Q33, Q66)
    assert sum(res["alloc"].values()) == 8
    assert res["alloc"]["side"] == 0
    assert len(res["poses"]) == 8
    srcs = {s["src"]: s for s in samples}
    for p in res["poses"]:
        # every selected pose is a real sample, with its raw keypoints kept
        assert p["src"] in srcs
        assert np.allclose(p["body"], srcs[p["src"]]["body"])
        assert p["image_size"] == [64, 128]
    assert [p["idx"] for p in res["poses"]] == list(range(1, 9))
