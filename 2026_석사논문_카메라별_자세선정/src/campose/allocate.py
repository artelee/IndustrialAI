"""Split a fixed budget of K generation poses across walking directions.

Rule (proportional allocation with a minimum per appearing direction):

1. A direction is *available* if at least ``min_samples`` sampled gallery
   poses of the camera have it. Unavailable directions get 0 poses.
2. Each available direction starts at ``max(min_per_dir, round(frac * K))``
   where ``frac`` is its share among all classified poses of the camera.
3. While the total exceeds K, remove one pose from the direction (with more
   than ``min_per_dir``) that is most over-allocated (``alloc - frac*K``).
4. While the total is below K, add one pose to the available direction that
   is most under-allocated (``frac*K - alloc``).

Ties are broken by direction order ``front, back, oblique, side`` (first wins),
and ``round`` is Python's round-half-to-even, as in the original experiment.
Allocations are reported in the same order, e.g. ``1/1/3/3`` means
front 1, back 1, oblique 3, side 3.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .posecore.direction import DIRECTIONS

DEFAULT_K = 8
MIN_PER_DIR = 1
MIN_SAMPLES = 5


def direction_counts(
    labels: Iterable[str], directions: Sequence[str] = DIRECTIONS
) -> dict[str, int]:
    """Count direction labels (labels outside ``directions`` are ignored)."""
    labels = list(labels)
    return {d: sum(1 for x in labels if x == d) for d in directions}


def direction_fractions(
    counts: Mapping[str, int], directions: Sequence[str] = DIRECTIONS
) -> dict[str, float]:
    """Share of each direction among all counted poses (0 if there are none)."""
    total = sum(int(counts.get(d, 0)) for d in directions)
    if total == 0:
        return {d: 0.0 for d in directions}
    return {d: float(np.float64(counts.get(d, 0)) / total) for d in directions}


def allocate_poses(
    counts: Mapping[str, int],
    k: int = DEFAULT_K,
    min_per_dir: int = MIN_PER_DIR,
    min_samples: int = MIN_SAMPLES,
    directions: Sequence[str] = DIRECTIONS,
) -> dict[str, int]:
    """Allocate ``k`` poses to directions in proportion to their counts.

    Args:
        counts: number of classified gallery poses per direction for one camera.
        k: total number of poses to select.
        min_per_dir: minimum poses for every available direction.
        min_samples: a direction needs at least this many poses to be available.
        directions: direction order (also the tie-break order).

    Returns:
        ``{direction: n}`` for **all** directions (0 for unavailable ones);
        the values sum to ``k`` whenever at least one direction is available.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    frac = direction_fractions(counts, directions)
    avail = [d for d in directions if int(counts.get(d, 0)) >= min_samples]
    alloc = {d: 0 for d in directions}
    if not avail:
        return alloc
    raw = {d: frac[d] * k for d in avail}
    for d in avail:
        alloc[d] = max(min_per_dir, int(round(raw[d])))
    while sum(alloc.values()) > k:
        cand = [d for d in avail if alloc[d] > min_per_dir]
        if not cand:
            break
        d = max(cand, key=lambda x: alloc[x] - raw[x])
        alloc[d] -= 1
    while sum(alloc.values()) < k:
        d = max(avail, key=lambda x: raw[x] - alloc[x])
        alloc[d] += 1
    return alloc


def allocation_string(alloc: Mapping[str, int], directions: Sequence[str] = DIRECTIONS) -> str:
    """Format an allocation as ``front/back/oblique/side`` counts, e.g. ``"3/1/2/2"``."""
    return "/".join(str(int(alloc.get(d, 0))) for d in directions)
