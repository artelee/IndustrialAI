# walking-direction-analyzer

Classify pedestrians' walking direction (front / back / oblique / side) from DWPose body keypoints, and analyze per-camera direction distributions and intra- vs. inter-camera pose variation in person re-identification (Re-ID) galleries.

[한국어 README](README.md)

## Problem

Pose-guided Re-ID augmentation (for example generating a gallery person in several poses) usually uses one fixed pose set for every camera. Cameras differ in where people walk relative to the lens, so each camera sees a different mix of front, back and side views. This repository measures that:

- how the walking-direction mix changes from camera to camera, and
- whether poses vary more **within** a camera than **between** cameras. If they do, one representative pose per camera is not enough.

Everything works from keypoint arrays. DWPose is only needed to extract the keypoints once; the keypoints can be cached and re-analyzed on CPU.

## Method

| Step | What happens | Module |
|---|---|---|
| 1. Keypoints | DWPose (via `controlnet_aux`) gives 18 OpenPose-layout body joints per image (first person, confidence > 0.3 = visible). | `walkdir/pose.py` |
| 2. Normalization | Neck moved to the origin, coordinates divided by torso length (neck to hip), invisible joints set to 0, giving a 36-dim pose vector. | `walkdir/pose.py` |
| 3. Quality filter | Rejects with a reason: `core` (fewer than 12 joints, or neck/shoulders/hips missing), `halfbody` (no ankle or no head joint), `small` (skeleton height < 0.55 of the image), `limb` (a bone > 1.6x torso length). `nodet` / `unknown` count failed detections and undecidable directions. | `walkdir/quality.py` |
| 4. Direction | Shoulder width `w = abs(x_rs - x_ls)` in torso lengths: `w < q33` side, `q33 <= w < q66` oblique, otherwise front/back by left/right shoulder order (person's right shoulder on the image left means front). `q33`/`q66` are the 1/3 and 2/3 quantiles of the gallery's shoulder widths. | `walkdir/direction.py` |
| 5. Per-camera share | Direction distribution per camera, with per-reason rejection counts. | `walkdir/stats.py` |
| 6. Variation | Intra-camera variation is the mean distance of a camera's poses to the camera mean. Inter-camera variation is the mean distance between camera means. Both are computed in the 36-dim space. | `walkdir/stats.py` |
| 7. Figures | Stacked direction bars and variation bars from statistics, plus skeleton-only drawings. | `walkdir/plot.py` |

![Direction rule on normalized skeletons](docs/figures/skeleton_directions.png)

*Synthetic skeletons (no person images). Shoulder width separates side, oblique and front/back views. The order of the shoulders (the ring marks the person's right shoulder) separates front from back. Face joints are not used because DWPose often detects eyes on people seen from behind.*

## Install

```bash
git clone https://github.com/artelee/IndustrialAI.git
cd IndustrialAI/2026_석사논문_보행방향분석
pip install -r requirements.txt      # core: numpy, matplotlib, pyyaml, pillow, pytest
pip install -e .                     # optional, makes `import walkdir` available everywhere
pytest                               # CPU only, no weights needed
```

Pose extraction is optional. These versions were tested with Python 3.10 and torch 2.1:

```bash
pip install opencv-python mmengine==0.10.7 mmcv==2.1.0 mmdet==3.3.0 mmpose==1.3.2
pip install --no-deps controlnet_aux==0.0.10
```

### Data and weights

- Datasets: [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html), DukeMTMC-reID (withdrawn by its authors and no longer distributed), and [MSMT17](https://www.pkuvmc.com/dataset.html). Obtain them from the official sources under their licenses. MSMT17 is expected in Market-style file names (`0000_c14_....jpg`). No images are included in this repository.
- DWPose weights are downloaded automatically by `controlnet_aux` on first use: [dw-ll_ucoco_384](https://huggingface.co/wanghaofan/dw-ll_ucoco_384) and [YOLOX-L](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox). To use local files, set `pose.det_ckpt` and `pose.pose_ckpt` in `configs/default.yaml`.

## Run

```bash
# 1) Extract keypoints with DWPose and analyze (keypoints are cached to outputs/keypoints_market.npz)
python scripts/analyze_cameras.py --gallery data/market1501/bounding_box_test --parser market --samples 500 --out outputs/
#    add --device cuda for a GPU run; --parser duke | msmt for the other datasets

# 2) Re-analyze cached keypoints without the pose model (e.g. other thresholds)
python scripts/analyze_cameras.py --keypoints outputs/keypoints_market.npz --parser market --thresholds auto --out outputs/

# 3) CPU-only demo with synthetic skeletons
python scripts/make_synthetic_keypoints.py --out outputs/keypoints_synthetic.npz
python scripts/analyze_cameras.py --keypoints outputs/keypoints_synthetic.npz --parser market --name synthetic --out outputs/

# 4) Redraw the README figures from summary statistics
python scripts/make_readme_figures.py --stats docs/figures/figure_data.json --out docs/figures/
```

The analysis writes these files to `outputs/`:

- `directions_<name>.json` and `.csv`: per-camera direction shares, rejection reasons, thresholds, and intra/inter variation
- `fig_directions_<name>.png` and `fig_variation_<name>.png`
- `keypoints_<name>.npz`: raw keypoints, written only when `--gallery` is used

`--thresholds` accepts `preset` (the default: the dataset values above), `auto` (tertiles of the analyzed sample), or `q33,q66`.

Library use:

```python
from walkdir.quality import filter_and_normalize
from walkdir.direction import classify_direction, PRESET_THRESHOLDS

res, reason = filter_and_normalize(body_18x2, scores_18)    # reason in {"core","halfbody","small","limb", ...}
if res is not None:
    norm_body, visible = res
    print(classify_direction(norm_body, visible, *PRESET_THRESHOLDS["market"]))
```

## Repository layout

```
walking-direction-analyzer/
├── configs/
│   ├── default.yaml          # filter thresholds, direction thresholds, pose-model settings
│   └── datasets.yaml         # example (placeholder) gallery paths
├── docs/figures/             # README figures + figure_data.json (statistics only)
├── scripts/
│   ├── analyze_cameras.py    # CLI: gallery or cached keypoints -> per-camera statistics + figures
│   ├── make_synthetic_keypoints.py
│   └── make_readme_figures.py
├── src/walkdir/
│   ├── pose.py               # DWPose wrapper (optional import), neck/torso normalization
│   ├── quality.py            # quality filter with rejection reasons
│   ├── direction.py          # shoulder-width / shoulder-order direction rule, thresholds
│   ├── stats.py              # direction shares, intra/inter variation, TVD, coverage
│   ├── plot.py               # stacked direction bars, variation bars, skeleton drawings
│   ├── analysis.py           # end-to-end analysis of a keypoint set
│   ├── data.py               # file-name parsers, per-camera sampling, keypoint cache
│   └── synthetic.py          # synthetic skeletons for tests / demo
└── tests/                    # pytest (CPU, synthetic keypoints)
```

`pose.py`, `quality.py` and `direction.py` depend only on numpy (and on each other: `quality` imports `pose`). They can be copied into other projects as a unit.

## Citation, license, related repositories

- DWPose: Z. Yang, A. Zeng, C. Yuan, Y. Li. *Effective Whole-body Pose Estimation with Two-stages Distillation.* ICCV Workshops 2023. Apache-2.0.
- controlnet_aux (Hugging Face), used as the DWPose wrapper. Apache-2.0.
- OpenPose 18-joint body layout: Z. Cao et al. *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.* TPAMI 2019.
- Market-1501: L. Zheng et al. *Scalable Person Re-identification: A Benchmark.* ICCV 2015.
- DukeMTMC-reID: E. Ristani et al. *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking.* ECCV Workshops 2016; Z. Zheng et al. *Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro.* ICCV 2017.
- MSMT17: L. Wei et al. *Person Transfer GAN to Bridge Domain Gap for Person Re-Identification.* CVPR 2018.

License: [MIT](LICENSE) (c) 2026 ehlee. The datasets and pose-model weights keep their own licenses and are not redistributed.

Related repositories:
- [2026_석사논문_ReID_평가벤치](../2026_석사논문_ReID_평가벤치/)
- [2026_석사논문_ReID_카메라특징보정](../2026_석사논문_ReID_카메라특징보정/)
- [2026_석사논문_보행방향분석](../2026_석사논문_보행방향분석/) (this repository)
- [2026_석사논문_재현가능한_Diffusion생성](../2026_석사논문_재현가능한_Diffusion생성/)
- [2026_석사논문_카메라별_자세선정](../2026_석사논문_카메라별_자세선정/)
- [2026_석사논문_MSMT17_변환기](../2026_석사논문_MSMT17_변환기/)
