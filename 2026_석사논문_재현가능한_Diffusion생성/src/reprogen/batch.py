"""Pre-flight checks, resume bookkeeping and IPG invocation for one batch."""
from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from .seeding import SEED_PATCH_MARKER, is_seed_patched
from .sets import BatchConfig, GenSet


def list_inputs(ref_dir: str | Path, pattern: str = "*.jpg") -> list[Path]:
    """Reference images in ``ref_dir`` matching ``pattern``, sorted by name."""
    return sorted(p for p in Path(ref_dir).glob(pattern) if p.is_file())


def remaining_inputs(inputs: Iterable[Path], out_dir: str | Path) -> list[Path]:
    """Inputs whose output panel ``out_dir/<basename>`` does not exist yet.

    IPG writes each panel under the reference file name, so an existing file
    means that image is done. Order of ``inputs`` is preserved.
    """
    out_dir = Path(out_dir)
    done = {p.name for p in out_dir.iterdir()} if out_dir.is_dir() else set()
    return [p for p in inputs if p.name not in done]


def pose_sort_key(filename: str) -> int:
    """Pose ordering used by IPG: integer prefix before ``_`` or ``.``."""
    return int(filename.split("_")[0].split(".")[0])


def preflight(cfg: BatchConfig, sets: list[GenSet] | None = None) -> list[str]:
    """Check everything that would make a long run fail or be non-reproducible.

    Returns a list of human-readable problems (empty list = OK):
    seed patch present in ``inference.py``, reference folder present (and with
    the expected image count if given), pose folder holding exactly
    ``num_poses`` files with integer-prefixed names.
    """
    problems: list[str] = []
    sets = cfg.sets if sets is None else sets
    if not sets:
        problems.append("no sets selected")
    if not cfg.ipg.inference_py.is_file():
        problems.append(f"IPG inference.py not found: {cfg.ipg.inference_py}")
    elif not is_seed_patched(cfg.ipg.inference_py):
        problems.append(
            f"seed patch not applied ({SEED_PATCH_MARKER!r} missing in {cfg.ipg.inference_py}); "
            "apply patches/ipg_seed.patch"
        )
    for s in sets:
        if not s.ref_dir.is_dir():
            problems.append(f"[{s.name}] reference folder missing: {s.ref_dir}")
        else:
            n_in = len(list_inputs(s.ref_dir, cfg.input_glob))
            if n_in == 0:
                problems.append(f"[{s.name}] no '{cfg.input_glob}' images in {s.ref_dir}")
            elif s.expected_inputs is not None and n_in != s.expected_inputs:
                problems.append(f"[{s.name}] expected {s.expected_inputs} inputs, found {n_in} in {s.ref_dir}")
        if not s.pose_dir.is_dir():
            problems.append(f"[{s.name}] pose folder missing: {s.pose_dir}")
            continue
        poses = os.listdir(s.pose_dir)  # IPG reads every entry of the folder
        if len(poses) != cfg.num_poses:
            problems.append(f"[{s.name}] pose folder has {len(poses)} files, expected {cfg.num_poses}: {s.pose_dir}")
        bad_names = []
        for p in poses:
            try:
                pose_sort_key(p)
            except ValueError:
                bad_names.append(p)
        if bad_names:
            problems.append(f"[{s.name}] pose files without integer prefix (IPG sort fails): {sorted(bad_names)[:3]}")
    return problems


def stage_inputs(files: Iterable[Path], staging_dir: str | Path) -> Path:
    """Recreate ``staging_dir`` holding links (or copies) to ``files`` only.

    IPG processes a whole folder, so the not-yet-generated images are exposed
    through a fresh folder. Symlinks are used when possible.
    """
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for f in files:
        target = staging_dir / f.name
        src = f.resolve()
        try:
            target.symlink_to(src)
        except OSError:
            shutil.copy2(src, target)
    return staging_dir


def build_command(cfg: BatchConfig, gen_set: GenSet, ref_dir: Path, out_dir: Path) -> list[str]:
    """Argument list for IPG ``inference.py`` (run with ``cwd=ipg_dir``).

    Paths are made absolute because the process runs inside the IPG folder.
    """
    return [
        cfg.ipg.python,
        "inference.py",
        "--config", cfg.ipg.config,
        "--ckpt_dir", str(cfg.ipg.ckpt_dir.resolve()),
        "--pose_dir", str(gen_set.pose_dir.resolve()),
        "--ref_dir", str(Path(ref_dir).resolve()),
        "--out_dir", str(Path(out_dir).resolve()),
        *cfg.ipg.extra_args,
    ]


def run_ipg(cmd: list[str], cwd: Path, log: Callable[[str], None] = print, quiet_prefix: str = "Saved to") -> int:
    """Run IPG, streaming its output (per-image ``Saved to`` lines suppressed)."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if not line.startswith(quiet_prefix):
            log(line.rstrip("\n"))
    return proc.wait()
