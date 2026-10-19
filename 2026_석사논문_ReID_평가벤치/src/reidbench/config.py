"""YAML config loading with command-line overrides."""
from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["load_config", "experiment_columns"]


def load_config(path: str | Path, **overrides) -> dict:
    """Load a YAML config; non-None keyword overrides replace top-level keys."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def experiment_columns(cfg: dict) -> list[dict]:
    """Experiments from the config, each with a display label ``<domain> <train>→<eval>``."""
    out = []
    for e in cfg["experiments"]:
        e = dict(e)
        e.setdefault("label", f"{e['domain']} {e['train']}→{e['eval']}")
        out.append(e)
    return out
