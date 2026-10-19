"""Load the generation-set list (``configs/sets.yaml``).

A *set* is one IPG run: every reference image in ``ref_dir`` is generated with
the poses in ``pose_dir`` and the panels are written to ``<output_root>/<name>``.
An entry with ``cams: [...]`` is expanded into one set per camera, formatting
``{cam}`` in ``name``, ``ref_dir`` and ``pose_dir``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenSet:
    """One reference-folder x pose-folder generation job."""

    name: str
    ref_dir: Path
    pose_dir: Path
    expected_inputs: int | None = None


@dataclass(frozen=True)
class IPGConfig:
    """How to invoke Pose2ID IPG ``inference.py``."""

    python: str = "python"
    ipg_dir: Path = Path("third_party/Pose2ID/IPG")
    config: str = "configs/inference.yaml"
    ckpt_dir: Path = Path("third_party/Pose2ID/IPG/pretrained")
    extra_args: tuple[str, ...] = ()

    @property
    def inference_py(self) -> Path:
        """Path of the IPG inference script."""
        return self.ipg_dir / "inference.py"


@dataclass(frozen=True)
class BatchConfig:
    """Full batch configuration."""

    ipg: IPGConfig
    output_root: Path
    staging_root: Path
    sets: list[GenSet] = field(default_factory=list)
    input_glob: str = "*.jpg"
    num_poses: int = 8
    seconds_per_image: float = 23.0


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    cams = entry.get("cams")
    if not cams:
        return [entry]
    out = []
    for cam in cams:
        e = {k: v for k, v in entry.items() if k != "cams"}
        for key in ("name", "ref_dir", "pose_dir"):
            e[key] = str(e[key]).format(cam=cam)
        out.append(e)
    return out


def parse_config(raw: dict[str, Any]) -> BatchConfig:
    """Build a :class:`BatchConfig` from an already-parsed YAML mapping."""
    ipg_raw = raw.get("ipg", {}) or {}
    ipg = IPGConfig(
        python=str(ipg_raw.get("python", "python")),
        ipg_dir=Path(ipg_raw.get("ipg_dir", IPGConfig.ipg_dir)),
        config=str(ipg_raw.get("config", IPGConfig.config)),
        ckpt_dir=Path(ipg_raw.get("ckpt_dir", IPGConfig.ckpt_dir)),
        extra_args=tuple(str(a) for a in ipg_raw.get("extra_args", []) or []),
    )
    sets: list[GenSet] = []
    for entry in raw.get("sets", []) or []:
        for e in _expand(dict(entry)):
            sets.append(
                GenSet(
                    name=str(e["name"]),
                    ref_dir=Path(e["ref_dir"]),
                    pose_dir=Path(e["pose_dir"]),
                    expected_inputs=e.get("expected_inputs"),
                )
            )
    names = [s.name for s in sets]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate set names: {dupes}")
    output_root = Path(raw.get("output_root", "outputs/gen_seed"))
    return BatchConfig(
        ipg=ipg,
        output_root=output_root,
        staging_root=Path(raw.get("staging_root", output_root / "_staging")),
        sets=sets,
        input_glob=str(raw.get("input_glob", "*.jpg")),
        num_poses=int(raw.get("num_poses", 8)),
        seconds_per_image=float(raw.get("seconds_per_image", 23.0)),
    )


def load_config(path: str | Path) -> BatchConfig:
    """Read and parse a sets YAML file."""
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f) or {})
