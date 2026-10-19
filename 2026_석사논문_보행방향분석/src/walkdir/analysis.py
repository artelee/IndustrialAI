"""End-to-end analysis of a keypoint set (no pose model needed)."""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from .direction import DIRECTIONS, UNKNOWN, classify_direction, compute_thresholds, shoulder_dx
from .quality import QualityConfig, empty_reason_counts, filter_and_normalize
from .stats import intra_inter_variation, per_camera_distribution


def analyze_keypoints(
    kp: Mapping[str, np.ndarray],
    thresholds: tuple[float, float] | None,
    quality: QualityConfig = QualityConfig(),
    variation_mode: str = "core",
) -> dict:
    """Direction distribution, rejection reasons and pose variation per camera.

    Args:
        kp: arrays ``cams``, ``bodies`` (N,18,2), ``scores`` (N,18), ``detected`` (N,).
        thresholds: ``(q33, q66)``; ``None`` -> computed from the strict-filtered sample.
        quality: strict-filter settings for direction statistics.
        variation_mode: quality mode for the variation analysis ("core" or "strict").
    """
    cams = np.asarray(kp["cams"]).astype(int)
    bodies, scores, detected = kp["bodies"], kp["scores"], np.asarray(kp["detected"], bool)
    strict = replace(quality, mode="strict")
    var_cfg = replace(quality, mode=variation_mode)

    passed: list[tuple[int, np.ndarray, np.ndarray]] = []
    drops: dict[int, dict[str, int]] = {int(c): empty_reason_counts() for c in sorted(set(cams.tolist()))}
    var_X, var_cams = [], []
    for i in range(len(cams)):
        c = int(cams[i])
        body = bodies[i] if detected[i] else None
        sc = scores[i] if detected[i] else None
        res, reason = filter_and_normalize(body, sc, strict)
        if res is None:
            drops[c][reason] += 1
        else:
            passed.append((c, *res))
        vres, _ = filter_and_normalize(body, sc, var_cfg)
        if vres is not None:
            var_X.append(vres[0].reshape(-1))
            var_cams.append(c)

    if thresholds is None:
        widths = [abs(dx) for _, b, v in passed if (dx := shoulder_dx(b, v)) is not None]
        thresholds = compute_thresholds(widths)
    q33, q66 = thresholds

    lab_cams, labels = [], []
    for c, b, v in passed:
        d = classify_direction(b, v, q33, q66)
        if d == UNKNOWN:
            drops[c]["unknown"] += 1
            continue
        lab_cams.append(c)
        labels.append(d)

    dist = per_camera_distribution(lab_cams, labels)
    counts = {c: sum(1 for x in lab_cams if x == c) for c in drops}
    out = {
        "thresholds": {"q33": float(q33), "q66": float(q66)},
        "n_total": int(len(cams)),
        "n_classified": int(len(labels)),
        "cameras": {
            str(c): {
                "n_sampled": int((cams == c).sum()),
                "n_classified": counts[c],
                "direction": dist.get(c, {d: 0.0 for d in DIRECTIONS}),
                "dropped": drops[c],
            }
            for c in drops
        },
    }
    if len(set(var_cams)) >= 2:
        var = intra_inter_variation(np.array(var_X), var_cams)
        var["n_poses"] = len(var_X)
        var["filter"] = variation_mode
        out["variation"] = var
    return out
