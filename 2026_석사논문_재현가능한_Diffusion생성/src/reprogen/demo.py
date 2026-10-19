"""Synthetic demo workspace (no real data, no GPU).

Creates noise "reference" images, 8 solid-colour "pose" images and a stand-in
``IPG/inference.py`` that accepts the same CLI as Pose2ID IPG, contains the
same per-image seed line as ``patches/ipg_seed.patch`` and writes 17-column
noise panels. Useful to exercise ``run_batch.py`` (resume, staging,
verification, cropping) end to end on a CPU.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml
from PIL import Image

STUB_INFERENCE = '''\
"""Stand-in for Pose2ID IPG inference.py (synthetic demo only)."""
import argparse
import os
import random
import zlib

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--ckpt_dir")
ap.add_argument("--pose_dir")
ap.add_argument("--ref_dir")
ap.add_argument("--out_dir")
ap.add_argument("--fail_after", type=int, default=-1)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
img_list = sorted(os.listdir(args.ref_dir))
pose_paths = sorted(os.listdir(args.pose_dir), key=lambda f: int(f.split('_')[0].split('.')[0]))
w, h = 8, 16
for n, ref_name in enumerate(img_list):
    if n == args.fail_after:
        raise SystemExit(1)
    seed = 12580 + (zlib.crc32(ref_name.encode()) & 0x7fffffff)
    rng = random.Random(seed)
    canvas = Image.new("RGB", (w * (1 + 2 * len(pose_paths)), h), "white")
    canvas.paste(Image.open(os.path.join(args.ref_dir, ref_name)).convert("RGB").resize((w, h)), (0, 0))
    for i, p in enumerate(pose_paths):
        canvas.paste(Image.open(os.path.join(args.pose_dir, p)).convert("RGB").resize((w, h)), (i * w * 2 + w, 0))
        gen = Image.frombytes("RGB", (w, h), bytes(rng.randrange(256) for _ in range(w * h * 3)))
        canvas.paste(gen, (i * w * 2 + 2 * w, 0))
    out = os.path.join(args.out_dir, ref_name)
    canvas.save(out, quality=95)
    print(f"Saved to {out}")
'''


def noise_image(size: tuple[int, int], seed: int) -> Image.Image:
    """Uniform RGB noise image."""
    rng = random.Random(seed)
    w, h = size
    return Image.frombytes("RGB", size, bytes(rng.randrange(256) for _ in range(w * h * 3)))


def make_demo(root: str | Path, n_refs: int = 5, num_poses: int = 8, n_cams: int = 2) -> Path:
    """Build the demo workspace under ``root`` and return the sets YAML path."""
    root = Path(root)
    refs = root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    for i in range(n_refs):
        noise_image((8, 16), i).save(refs / f"{i + 1:04d}_c1s1_{i:06d}_00.jpg", quality=95)
    pose_dirs = [root / "poses" / "std"] + [root / "poses" / f"c{c}" for c in range(1, n_cams + 1)]
    for d_i, d in enumerate(pose_dirs):
        d.mkdir(parents=True, exist_ok=True)
        for k in range(num_poses):
            Image.new("RGB", (8, 16), ((40 * k) % 256, 30 * d_i % 256, 200)).save(d / f"{k + 1}.png")
    ipg = root / "Pose2ID" / "IPG"
    (ipg / "configs").mkdir(parents=True, exist_ok=True)
    (ipg / "configs" / "inference.yaml").write_text("seed: 12580\n")
    (ipg / "inference.py").write_text(STUB_INFERENCE)
    (ipg / "pretrained").mkdir(exist_ok=True)
    cfg = {
        "ipg": {
            "python": sys.executable,
            "ipg_dir": str(ipg),
            "config": "configs/inference.yaml",
            "ckpt_dir": str(ipg / "pretrained"),
        },
        "output_root": str(root / "gen_seed"),
        "input_glob": "*.jpg",
        "num_poses": num_poses,
        "seconds_per_image": 23,
        "sets": [
            {"name": "demo_std", "ref_dir": str(refs), "pose_dir": str(pose_dirs[0]), "expected_inputs": n_refs},
            {"name": "demo_c{cam}", "ref_dir": str(refs), "pose_dir": str(root / "poses" / "c{cam}"),
             "cams": list(range(1, n_cams + 1))},
        ],
    }
    path = root / "sets_demo.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
