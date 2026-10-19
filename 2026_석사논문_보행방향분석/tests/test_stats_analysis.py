import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from walkdir.analysis import analyze_keypoints
from walkdir.data import load_keypoints, parse_duke, parse_market, save_keypoints
from walkdir.stats import (
    coverage, direction_distribution, intra_inter_variation, total_variation_distance,
)
from walkdir.synthetic import make_camera_set

ROOT = Path(__file__).resolve().parents[1]


def test_direction_distribution():
    d = direction_distribution(["front", "front", "side", "unknown"])
    assert d == {"front": 2 / 3, "back": 0.0, "oblique": 0.0, "side": 1 / 3}


def test_intra_inter_hand_example():
    X = np.array([[0, 0], [2, 0], [10, 0], [12, 0]], float)
    r = intra_inter_variation(X, [1, 1, 2, 2])
    assert r["intra"] == pytest.approx(1.0)
    assert r["inter"] == pytest.approx(10.0)
    assert r["ratio"] == pytest.approx(0.1)


def test_tvd_and_coverage():
    assert total_variation_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert coverage(np.array([[0.0], [3.0]]), np.array([[1.0], [3.0]])) == pytest.approx(0.5)


def test_parsers():
    assert parse_market("0002_c3s1_000451_03.jpg") == (2, 3)
    assert parse_duke("0002_c7_f0044158.jpg") == (2, 7)
    assert parse_duke("0000_c14_0000_023_14_0303morning_0242_1.jpg") == (0, 14)


def test_analysis_recovers_camera_mix(tmp_path):
    mixes = {1: {"front": 1.0}, 2: {"side": 1.0}}
    kp = make_camera_set(mixes, 50, seed=1, noise=0.003)
    p = tmp_path / "kp.npz"
    save_keypoints(p, kp["files"], kp["pids"], kp["cams"], kp["bodies"], kp["scores"], kp["detected"])
    kp2 = load_keypoints(p)
    res = analyze_keypoints(kp2, (1.076, 1.351))
    assert res["cameras"]["1"]["direction"]["front"] > 0.9
    assert res["cameras"]["2"]["direction"]["side"] > 0.9
    assert res["variation"]["ratio"] > 0


def test_cli_on_cached_keypoints(tmp_path):
    kp_path = tmp_path / "kp.npz"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_keypoints.py"),
                    "--out", str(kp_path), "--per-camera", "40"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/analyze_cameras.py"), "--keypoints", str(kp_path),
                    "--parser", "market", "--name", "synthetic", "--out", str(tmp_path), "--no-figures"],
                   check=True)
    res = json.load(open(tmp_path / "directions_synthetic.json"))
    assert set(res["cameras"]) == {"1", "2", "3"}
