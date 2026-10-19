"""Per-camera selection of K representative generation poses.

Pipeline for one camera (input: raw keypoints of sampled gallery images):

1. quality filter + neck/torso normalization (``posecore.quality``, strict mode)
2. walking direction per pose (``posecore.direction``)
3. allocation of K poses to directions (``allocate.allocate_poses``)
4. inside each direction, K-means with ``k = alloc[direction]`` clusters on the
   36-dim normalized pose vectors; for every cluster center, the **real** pose
   nearest to the center is selected (never the center itself). If a direction
   has no more poses than its allocation, all of them are taken.

Selected poses are ordered by direction (``front, back, oblique, side``) and,
inside a direction, by K-means cluster index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .allocate import DEFAULT_K, MIN_PER_DIR, MIN_SAMPLES, allocate_poses, direction_counts, direction_fractions
from .posecore.direction import DIRECTIONS, UNKNOWN, classify_direction
from .posecore.pose import pose_vector
from .posecore.quality import QualityConfig, empty_reason_counts, filter_and_normalize


@dataclass
class PoseRecord:
    """One gallery pose that passed the quality filter."""

    vector: np.ndarray            # 36-dim normalized pose vector
    direction: str
    body: np.ndarray              # raw 18x2 keypoints (image-relative)
    scores: np.ndarray            # 18 joint confidences
    source: str                   # gallery file name
    image_size: tuple[int, int]   # (width, height) of the image the keypoints refer to
    extra: dict[str, Any] = field(default_factory=dict)


def build_records(
    samples: Sequence[Mapping[str, Any]],
    q33: float,
    q66: float,
    quality: QualityConfig = QualityConfig(),
) -> tuple[list[PoseRecord], dict[str, int]]:
    """Filter, normalize and direction-tag raw keypoint samples.

    Each sample is a mapping with ``src``, ``body`` (18x2 or ``None`` when no
    person was detected), ``score`` and ``image_size`` (w, h).

    Returns:
        (records that passed, rejection counts per reason)
    """
    drop = empty_reason_counts()
    recs: list[PoseRecord] = []
    for s in samples:
        body = None if s.get("body") is None else np.asarray(s["body"], np.float64)
        score = None if s.get("score") is None else np.asarray(s["score"], np.float64)
        res, reason = filter_and_normalize(body, score, quality)
        if res is None:
            drop[str(reason)] += 1
            continue
        nb, vis = res
        d = classify_direction(nb, vis, q33, q66)
        if d == UNKNOWN:
            drop[UNKNOWN] += 1
            continue
        size = tuple(int(x) for x in s.get("image_size", (1, 1)))
        recs.append(PoseRecord(pose_vector(nb), d, body, score, str(s.get("src", "")), size))  # type: ignore[arg-type]
    return recs, drop


def kmeans_representatives(
    X: np.ndarray, k: int, seed: int = 0, n_init: int = 10
) -> list[int]:
    """Row indices of the members of ``X`` nearest to each of ``k`` K-means centers.

    If ``len(X) <= k`` all rows are returned. Uses scikit-learn's ``KMeans``
    (``n_init`` restarts, ``random_state=seed``). The same member can in
    principle be nearest to two centers; it is then returned twice, as in the
    original experiment.
    """
    X = np.asarray(X, np.float64)
    if k <= 0:
        return []
    if len(X) <= k:
        return list(range(len(X)))
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return [int(np.argmin(np.linalg.norm(X - c, axis=1))) for c in km.cluster_centers_]


def select_representatives(
    vectors: np.ndarray,
    labels: Sequence[str],
    alloc: Mapping[str, int],
    seed: int = 0,
    n_init: int = 10,
    directions: Sequence[str] = DIRECTIONS,
) -> list[tuple[int, str]]:
    """Pick ``alloc[d]`` representative rows per direction ``d``.

    Returns:
        ``[(row index into vectors, direction), ...]`` in direction order.
    """
    vectors = np.asarray(vectors, np.float64)
    labels_arr = np.asarray(list(labels))
    chosen: list[tuple[int, str]] = []
    for d in directions:
        k = int(alloc.get(d, 0))
        if k <= 0:
            continue
        idx = np.where(labels_arr == d)[0]
        if len(idx) == 0:
            continue
        pick = kmeans_representatives(vectors[idx], k, seed=seed, n_init=n_init)
        chosen.extend((int(idx[p]), d) for p in pick)
    return chosen


def select_camera_poses(
    samples: Sequence[Mapping[str, Any]],
    q33: float,
    q66: float,
    k: int = DEFAULT_K,
    quality: QualityConfig = QualityConfig(),
    min_per_dir: int = MIN_PER_DIR,
    min_samples: int = MIN_SAMPLES,
    seed: int = 0,
    n_init: int = 10,
) -> dict[str, Any]:
    """Full selection for one camera; returns a JSON-serializable result.

    Output keys: ``alloc``, ``frac``, ``n`` (poses passing the filter), ``drop``
    (rejection counts) and ``poses`` (list of ``idx``, ``dir``, ``src``, ``body``,
    ``score``, ``image_size``; keypoints are kept so the poses can be rendered
    without re-running the pose estimator).
    """
    recs, drop = build_records(samples, q33, q66, quality)
    labels = [r.direction for r in recs]
    counts = direction_counts(labels)
    alloc = allocate_poses(counts, k=k, min_per_dir=min_per_dir, min_samples=min_samples)
    vectors = np.array([r.vector for r in recs]) if recs else np.zeros((0, 36))
    chosen = select_representatives(vectors, labels, alloc, seed=seed, n_init=n_init)
    poses = []
    for i, (ri, d) in enumerate(chosen, 1):
        r = recs[ri]
        poses.append(dict(
            idx=i, dir=d, src=r.source,
            body=r.body.tolist(), score=r.scores.tolist(),
            image_size=list(r.image_size),
        ))
    return dict(
        alloc={d: alloc[d] for d in DIRECTIONS},
        frac=direction_fractions(counts),
        n=len(recs),
        drop=drop,
        poses=poses,
    )
