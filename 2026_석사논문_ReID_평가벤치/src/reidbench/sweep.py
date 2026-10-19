"""Parameter sweeps (e.g. the fusion weight eta)."""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .bootstrap import paired_bootstrap_ci
from .conditions import Experiment, run_condition

__all__ = ["sweep", "eta_sweep"]


def sweep(values: Iterable, fn: Callable[[object], dict]) -> dict:
    """Call ``fn(v)`` for each value and collect ``{v: result}``."""
    return {v: fn(v) for v in values}


def eta_sweep(
    exp: Experiment,
    etas: Iterable[float] = (0.25, 0.5, 1.0, 2.0),
    assigns: Iterable[str] = ("sim", "oracle"),
    calibrations: Iterable[bool] = (False, True),
    rng: np.random.RandomState | int | None = 0,
    n_boot: int = 1000,
) -> dict:
    """mAP of baseline / common fusion / per-camera fusion for each eta.

    Returns ``{(assign, calibrate, eta): dict(base, common, per_camera, d)}``
    where ``d`` is the paired bootstrap CI of per-camera minus common (AP, %p).
    Loop order (assign, calibrate, eta) is fixed so a shared ``rng`` gives
    reproducible CIs.
    """
    r = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(0 if rng is None else rng)
    out = {}
    for a in assigns:
        for cal in calibrations:
            _, ap_b = run_condition(exp, "B", cal)          # independent of eta
            for eta in etas:
                _, ap_s = run_condition(exp, "S", cal, eta, a)
                _, ap_t = run_condition(exp, "T", cal, eta, a)
                out[(a, cal, float(eta))] = dict(
                    base=float(ap_b.mean() * 100), common=float(ap_s.mean() * 100),
                    per_camera=float(ap_t.mean() * 100),
                    d=paired_bootstrap_ci(ap_s, ap_t, n_boot=n_boot, rng=r))
    return out
