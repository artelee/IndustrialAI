"""File-name parsing, split filtering, lazy imports, and the CLI on a synthetic cache."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from reid_calib.extract import list_split, parse_duke, parse_market
from reid_calib.features import FeatureSplit, cache_path, load_split, save_split

ROOT = Path(__file__).resolve().parents[1]


def test_parse_names():
    assert parse_market("data/market1501/query/0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market("-1_c3s2_000000_00.jpg") == (-1, 3)
    assert parse_duke("0005_c2_f0046985.jpg") == (5, 2)
    assert parse_duke("0012_c8_f0001.jpg") == (12, 8)


def test_list_split_filters_junk_and_distractors(tmp_path):
    for n in ["0001_c1s1_0_00.jpg", "0000_c2s1_0_00.jpg", "-1_c3s1_0_00.jpg", "0003_c4s1_0_00.jpg", "x.txt"]:
        (tmp_path / n).touch()
    files, pid, cam = list_split(tmp_path, parse_market, query=True)
    assert list(pid) == [1, 3] and list(cam) == [1, 4]
    files, pid, cam = list_split(tmp_path, parse_market, query=False)
    assert sorted(pid) == [0, 1, 3]


def test_numpy_modules_do_not_import_torch():
    code = ("import sys; sys.path.insert(0, 'src'); import reid_calib.extract, reid_calib.calibrate, "
            "reid_calib.evaluate, reid_calib.stats; print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _synthetic_cache(d: Path) -> None:
    """Identities + a strong per-camera offset (camera bias) that calibration should remove."""
    r = np.random.RandomState(0)
    n_id, dim = 30, 32
    ident = r.randn(n_id + 1, dim)
    off = {c: r.randn(dim) * 2.0 for c in range(1, 4)}

    def make(n: int, query: bool) -> FeatureSplit:
        pid = r.randint(1, n_id + 1, n)
        cam = r.randint(1, 4, n)
        if query:
            cam[:] = 1
        feat = ident[pid] + np.stack([off[c] for c in cam]) + 0.6 * r.randn(n, dim)
        return FeatureSplit(feat=feat, pid=pid, cam=cam, name=np.array([f"{i}.jpg" for i in range(n)]))

    save_split(cache_path(d, "clip", "duke", "market", "query"), make(60, True))
    save_split(cache_path(d, "clip", "duke", "market", "gal"), make(300, False))


def test_save_load_roundtrip(tmp_path):
    _synthetic_cache(tmp_path)
    s = load_split(cache_path(tmp_path, "clip", "duke", "market", "query"))
    assert s.feat.dtype == np.float32 and len(s) == 60


def test_cli_on_synthetic_cache(tmp_path):
    feats, out = tmp_path / "features", tmp_path / "results"
    _synthetic_cache(feats)
    cmd = [sys.executable, "-B", "scripts/run_eval.py", "--backbone", "clipreid", "--train", "duke",
           "--eval", "market", "--calib", "--features", str(feats), "--out", str(out), "--n-boot", "200"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    res = json.loads((out / "clipreid_duke_market.json").read_text())
    assert res["calib"]["mAP"] > res["base"]["mAP"]
    assert res["diff_mAP"]["lo"] <= res["diff_mAP"]["mean"] <= res["diff_mAP"]["hi"]
    assert (out / "summary.md").exists()
    # same seed -> identical CI
    subprocess.run(cmd[:-2] + ["--n-boot", "200", "--out", str(out / "again")], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    again = json.loads((out / "again" / "clipreid_duke_market.json").read_text())
    assert again["diff_mAP"] == res["diff_mAP"]
