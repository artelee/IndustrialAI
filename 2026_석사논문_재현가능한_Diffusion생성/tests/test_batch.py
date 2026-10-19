import subprocess
import sys

from PIL import Image

from reprogen.batch import build_command, list_inputs, preflight, remaining_inputs, stage_inputs
from reprogen.demo import make_demo
from reprogen.sets import load_config, parse_config


def _touch_images(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (4, 8), (1, 2, 3)).save(d / n)


def test_sorted_inputs_and_remaining(tmp_path):
    ref = tmp_path / "ref"
    _touch_images(ref, ["c.jpg", "a.jpg", "b.jpg", "d.jpg"])
    (ref / "notes.txt").write_text("x")
    inputs = list_inputs(ref)
    assert [p.name for p in inputs] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    out = tmp_path / "out"
    assert [p.name for p in remaining_inputs(inputs, out)] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    _touch_images(out, ["b.jpg", "d.jpg", "zzz.jpg"])
    assert [p.name for p in remaining_inputs(inputs, out)] == ["a.jpg", "c.jpg"]
    _touch_images(out, ["a.jpg", "c.jpg"])
    assert remaining_inputs(inputs, out) == []


def test_stage_inputs(tmp_path):
    ref = tmp_path / "ref"
    _touch_images(ref, ["a.jpg", "b.jpg"])
    st = tmp_path / "stage"
    _touch_images(st, ["old.jpg"])
    stage_inputs([ref / "b.jpg"], st)
    assert [p.name for p in st.iterdir()] == ["b.jpg"]


def test_cams_expansion_and_command(tmp_path):
    cfg = parse_config({
        "ipg": {"ipg_dir": str(tmp_path / "IPG"), "python": "py"},
        "sets": [{"name": "s_c{cam}", "ref_dir": "r", "pose_dir": "p/c{cam}", "cams": [3, 1]}],
    })
    assert [s.name for s in cfg.sets] == ["s_c3", "s_c1"]
    assert str(cfg.sets[1].pose_dir) == "p/c1"
    cmd = build_command(cfg, cfg.sets[0], tmp_path / "st", tmp_path / "out")
    assert cmd[:2] == ["py", "inference.py"]
    assert cmd[cmd.index("--pose_dir") + 1].endswith("p/c3")
    assert cmd[cmd.index("--out_dir") + 1] == str((tmp_path / "out").resolve())


def test_preflight(tmp_path):
    cfg = load_config(make_demo(tmp_path, n_refs=3))
    assert preflight(cfg) == []
    # remove the seed patch -> refused
    inf = cfg.ipg.inference_py
    inf.write_text(inf.read_text().replace("zlib.crc32", "hash"))
    assert any("seed patch" in p for p in preflight(cfg))
    inf.write_text(inf.read_text().replace("hash", "zlib.crc32"))
    # 9 poses -> refused; non-numeric pose name -> refused
    pose_dir = cfg.sets[0].pose_dir
    Image.new("RGB", (4, 8)).save(pose_dir / "extra.png")
    probs = preflight(cfg)
    assert any("9 files" in p for p in probs) and any("integer prefix" in p for p in probs)
    (pose_dir / "extra.png").unlink()
    # wrong expected input count -> refused
    (cfg.sets[0].ref_dir / "0001_c1s1_000000_00.jpg").unlink()
    assert any("expected 3 inputs, found 2" in p for p in preflight(cfg))


def _run_batch(repo_root, *args):
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_batch.py"), *args],
        capture_output=True, text=True,
    )


def test_run_batch_end_to_end_resume(tmp_path, repo_root):
    cfg_path = make_demo(tmp_path, n_refs=4, n_cams=1)
    r = _run_batch(repo_root, "--config", str(cfg_path), "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "remaining 4" in r.stdout and not (tmp_path / "gen_seed" / "demo_std").exists()

    r = _run_batch(repo_root, "--config", str(cfg_path), "--sets", "demo_std")
    assert r.returncode == 0, r.stdout + r.stderr
    out = tmp_path / "gen_seed" / "demo_std"
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert len(first) == 4

    # simulate a crash: one panel lost, one truncated -> --repair + resume regenerates exactly those
    names = sorted(first)
    (out / names[0]).unlink()
    (out / names[2]).write_bytes(first[names[2]][:50])
    r = _run_batch(repo_root, "--config", str(cfg_path), "--sets", "demo_std", "--repair")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "remaining 2" in r.stdout
    again = {p.name: p.read_bytes() for p in out.iterdir()}
    assert again == first  # per-image seed -> byte-identical regeneration
