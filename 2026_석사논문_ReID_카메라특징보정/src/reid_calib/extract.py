"""Backbone feature extraction (CLIP-ReID ViT-B/16, PersonViT ViT-B/16).

The model code is **not** part of this repository: each backbone is imported
from a user-provided checkout of its official repository (``repo`` in
``configs/backbones/*.yaml`` or ``--repo``). ``torch``, ``torchvision``, PIL and
the upstream code are imported lazily inside the functions, so calibration,
evaluation and statistics work without them.

Both backbones output the global feature *before* the BNNeck (``NECK_FEAT
before``), stored raw (normalisation happens at evaluation). Input: RGB resized
to 256x128, normalised with mean = std = 0.5.

The upstream CLIP-ReID code calls ``.to("cuda")`` internally, so CLIP-ReID
extraction needs a GPU; PersonViT also runs on CPU.

Note: both upstream repositories define top-level packages named ``config`` and
``model``. Extract one backbone per process (``scripts/run_eval.py`` does), or
let :func:`_import_upstream` purge the cached modules when switching.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .features import FeatureSplit

__all__ = [
    "parse_market",
    "parse_duke",
    "PARSERS",
    "list_split",
    "load_backbone",
    "extract_features",
    "extract_split",
]


# ---------------------------------------------------------------- file names
def parse_market(path: str | Path) -> tuple[int, int]:
    """Market-1501 name ``0002_c1s1_000451_03.jpg`` -> ``(pid, cam)`` = ``(2, 1)``."""
    p = os.path.basename(str(path)).split("_")
    return int(p[0]), int(p[1][1:2])


def parse_duke(path: str | Path) -> tuple[int, int]:
    """DukeMTMC-reID name ``0005_c2_f0046985.jpg`` -> ``(pid, cam)`` = ``(5, 2)``."""
    p = os.path.basename(str(path)).split("_")
    return int(p[0]), int(p[1][1:])


PARSERS: dict[str, Callable[[str | Path], tuple[int, int]]] = {"market": parse_market, "duke": parse_duke}


def list_split(image_dir: str | Path, parser: Callable[[str | Path], tuple[int, int]],
               query: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sorted ``*.jpg`` files of one split with pid / cam.

    Queries drop junk (pid -1) and distractors (pid 0); the gallery drops junk only.
    """
    files = sorted(str(f) for f in Path(image_dir).glob("*.jpg"))
    drop = (-1, 0) if query else (-1,)
    files = [f for f in files if parser(f)[0] not in drop]
    pid = np.array([parser(f)[0] for f in files], dtype=np.int64)
    cam = np.array([parser(f)[1] for f in files], dtype=np.int64)
    return files, pid, cam


# ---------------------------------------------------------------- backbones
def _import_upstream(repo: str | Path) -> None:
    """Put an upstream checkout first on ``sys.path`` and drop cached ``config`` / ``model`` modules."""
    repo = str(Path(repo).resolve())
    if not Path(repo).is_dir():
        raise FileNotFoundError(f"backbone repository not found: {repo} (set 'repo' in the backbone config or --repo)")
    for name in list(sys.modules):
        if name.split(".")[0] in ("config", "model"):
            del sys.modules[name]
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)


def _build_clipreid(cfg: dict, train: str, num_classes: int) -> tuple[Any, Callable]:
    """CLIP-ReID ViT-B/16 (Li et al., AAAI 2023) from its official repository."""
    _import_upstream(cfg["repo"])
    from config import cfg as default_cfg  # type: ignore  # upstream CLIP-ReID
    from model.make_model_clipreid import make_model  # type: ignore

    weight = str(cfg["models"][train]["weights"])
    rc = default_cfg.clone()
    rc.MODEL.NAME = "ViT-B-16"
    rc.MODEL.STRIDE_SIZE = [16, 16]
    rc.MODEL.SIE_CAMERA = False
    rc.MODEL.SIE_COE = 0.0
    rc.MODEL.ID_LOSS_TYPE = "softmax"
    rc.INPUT.SIZE_TRAIN = list(cfg["input"]["size"])
    rc.INPUT.SIZE_TEST = list(cfg["input"]["size"])
    rc.INPUT.PIXEL_MEAN = list(cfg["input"]["mean"])
    rc.INPUT.PIXEL_STD = list(cfg["input"]["std"])
    rc.DATASETS.NAMES = "market1501"
    rc.TEST.WEIGHT = weight
    rc.TEST.NECK_FEAT = "before"
    try:
        model = make_model(rc, num_class=num_classes, camera_num=0, view_num=1)
    except Exception:  # some upstream versions require camera_num > 0 (SIE is off either way)
        model = make_model(rc, num_class=num_classes, camera_num=6, view_num=1)
    model.load_param(weight)

    def forward(m: Any, x: Any) -> Any:
        f = m(x, cam_label=None)
        if isinstance(f, (list, tuple)):
            f = f[0] if hasattr(f[0], "dim") else f[-1]
        return f.reshape(f.shape[0], -1)

    return model, forward


def _build_personvit(cfg: dict, train: str, num_classes: int) -> tuple[Any, Callable]:
    """PersonViT ViT-B/16 (TransReID code base) from its official repository."""
    _import_upstream(cfg["repo"])
    from config import cfg as default_cfg  # type: ignore  # upstream PersonViT / TransReID
    from model import make_model  # type: ignore

    m_cfg = cfg["models"][train]
    c = default_cfg.clone()
    c.merge_from_file(str(Path(cfg["repo"]) / m_cfg["config_file"]))
    c.merge_from_list(["MODEL.PRETRAIN_PATH", str(cfg["pretrained"]), "MODEL.PRETRAIN_HW_RATIO", 2,
                       "MODEL.PRETRAIN_CHOICE", "imagenet", "TEST.NECK_FEAT", "before", "TEST.FEAT_NORM", "no"])
    c.freeze()
    model = make_model(c, num_class=num_classes, camera_num=int(cfg.get("camera_num", 8)), view_num=1)
    if m_cfg.get("weights"):
        model.load_param(str(m_cfg["weights"]))

    def forward(m: Any, x: Any) -> Any:
        f = m(x)
        return f[0] if isinstance(f, (list, tuple)) else f

    return model, forward


_BUILDERS = {"clipreid": _build_clipreid, "personvit": _build_personvit}


def load_backbone(cfg: dict, train: str, num_classes: int, device: str = "cuda") -> tuple[Any, Callable]:
    """Build a backbone from its config, load the weights for ``train`` and return ``(model, forward)``."""
    name = cfg["name"]
    if name not in _BUILDERS:
        raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(_BUILDERS)}")
    model, forward = _BUILDERS[name](cfg, train, num_classes)
    return model.to(device).eval(), forward


# ---------------------------------------------------------------- extraction
def extract_features(model: Any, forward: Callable, files: list[str], input_cfg: dict,
                     device: str = "cuda", batch_size: int = 64, log_every: int = 5000) -> np.ndarray:
    """Raw features ``(N, D)`` float32 of a list of image files."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([T.Resize(list(input_cfg["size"])), T.ToTensor(),
                    T.Normalize(list(input_cfg["mean"]), list(input_cfg["std"]))])
    out = []
    with torch.no_grad():
        for s in range(0, len(files), batch_size):
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[s:s + batch_size]]).to(device)
            out.append(forward(model, x).float().cpu().numpy())
            done = min(s + batch_size, len(files))
            if log_every and (done // log_every) > (s // log_every):
                print(f"    {done}/{len(files)}", flush=True)
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def extract_split(model: Any, forward: Callable, dataset_cfg: dict, query: bool, input_cfg: dict,
                  device: str = "cuda", batch_size: int = 64) -> FeatureSplit:
    """Extract the query (``query=True``) or gallery split of a dataset."""
    sub = dataset_cfg["query_dir"] if query else dataset_cfg["gallery_dir"]
    files, pid, cam = list_split(Path(dataset_cfg["root"]) / sub, PARSERS[dataset_cfg["parser"]], query)
    if not files:
        raise FileNotFoundError(f"no images in {Path(dataset_cfg['root']) / sub}")
    feat = extract_features(model, forward, files, input_cfg, device, batch_size)
    return FeatureSplit(feat=feat, pid=pid, cam=cam, name=np.array([os.path.basename(f) for f in files]))
