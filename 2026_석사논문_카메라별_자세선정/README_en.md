# camera-aware-pose-selection

Select the 8 generation poses of each camera from that camera's own walking-direction
distribution, and render them exactly like the generator's reference pose images.

![Common poses vs. camera-specific poses](docs/figures/common_vs_camera_poses.png)

*Skeletons only. Top row: the 8 common reference poses of Pose2ID's IPG
(front/back/oblique/side = 1/1/3/3). Below: camera-specific poses selected and rendered by
this repository for Market-1501 C2 (3/1/2/2), Market-1501 C6 (1/2/3/2) and DukeMTMC-reID C4
(4/3/1/0, no side views in this camera).*

[한국어 README](README.md)

## Problem

Pose2ID-style feature centralization generates images of a gallery person in a set of poses
and fuses their features into the gallery feature. The common pose set is the same for every
camera, but cameras differ strongly in how people walk past them: a corridor camera sees mostly
front and back views, a camera beside a path sees mostly side views. Poses that never occur in a
camera add features that do not match its queries. This repository picks the poses per camera so
that their direction mix follows the camera, and draws them in the same format as the reference
poses so that they can replace them without any other change to the generator.

## Method

| Step | What | Module |
|---|---|---|
| 1 | Sample up to 500 gallery images per camera, DWPose keypoints, quality filter, neck/torso normalization | `posecore/` (copied from walking-direction-analyzer) |
| 2 | Walking direction per pose: shoulder width (in torso lengths) vs. dataset quantiles `q33`/`q66` -> side / oblique / front-back; front vs. back from the shoulder order | `posecore/direction.py` |
| 3 | **Allocation** of K = 8 poses in proportion to the direction shares; every direction with >= 5 sampled poses gets at least 1, rarer directions get 0; the total is fixed to 8 by removing from the most over-allocated / adding to the most under-allocated direction | `allocate.py` |
| 4 | **Per-direction K-means** (k = allocation of that direction) on the 36-dim normalized pose vectors; for every cluster center the **real pose nearest to the center** is selected | `select.py` |
| 5 | **Rendering matched to the reference poses**: place the pose so that its top end, bottom end and horizontal center match the median layout of the reference poses (aspect ratio kept); OpenPose `draw_bodypose` on a 320 x 160 canvas; LANCZOS downscale to **64 x 128 (W x H)**; written with `cv2.imwrite` from the RGB array, i.e. **stored in BGR order**, like the reference files (the generator reads them with `cv2.imread` + `Image.fromarray`) | `render.py` |
| 6 | Check figure: common poses and camera-specific poses side by side (skeletons only) | `compare.py` |

Direction order everywhere (allocation strings, JSON, tie-break): **front / back / oblique / side**.
The common Pose2ID poses 1-8 are front, oblique, side, side, back, side, oblique, oblique
(`configs/common_poses.yaml`), i.e. 1/1/3/3.

## Install

```bash
pip install -r requirements.txt      # or: pip install -e .
pytest                               # CPU only
```

Python 3.10. The core (allocation, selection, rendering, figures) needs only numpy,
scikit-learn, OpenCV, Pillow, matplotlib and PyYAML. Rendering uses controlnet_aux's
`draw_bodypose` when it is installed and otherwise an identical built-in copy.

Keypoint extraction from gallery images is optional and needs DWPose through controlnet_aux:

```bash
pip install --no-deps controlnet_aux==0.0.10
pip install torch mmengine==0.10.7 mmcv==2.1.0 mmdet==3.3.0 mmpose==1.3.2
```

### Data and weights

- Market-1501, DukeMTMC-reID (withdrawn by its authors; use only if you already hold a licensed
  copy) and MSMT17: obtain them from the official sources under their licenses. Paths in
  `configs/*.yaml` are placeholders (`data/market1501/bounding_box_test`, ...).
- DWPose / YOLOX weights are downloaded by controlnet_aux on first use
  (DWPose: https://github.com/IDEA-Research/DWPose).
- Reference poses (`standard_poses/`) of Pose2ID's IPG: https://github.com/yuanc3/Pose2ID

## Run

```bash
# 1) Select 8 poses per camera from gallery images (DWPose on CPU; add --device cuda for GPU)
python scripts/select_poses.py --config configs/market.yaml \
    --gallery data/market1501/bounding_box_test --out outputs/selection/market \
    --save-keypoints outputs/keypoints/market.json

#    ... or from a saved keypoint JSON (no pose model needed)
python scripts/select_poses.py --config configs/market.yaml \
    --keypoints outputs/keypoints/market.json --out outputs/selection/market

# 2) Render them like the reference poses -> outputs/poses/market/c<cam>/<k>.jpg
python scripts/render_poses.py --selection outputs/selection/market --out outputs/poses/market
#    (optional) re-measure the reference layout: --reference-dir path/to/Pose2ID/IPG/standard_poses

# 3) Check figure: common poses vs. camera-specific poses
python scripts/compare_poses.py --common-dir path/to/Pose2ID/IPG/standard_poses \
    --camera "Market C2=outputs/poses/market/c2,outputs/selection/market/c2.json" \
    --camera "Market C6=outputs/poses/market/c6,outputs/selection/market/c6.json" \
    --out outputs/compare_market.png

# CPU-only demo with synthetic skeletons (no dataset, no pose model)
python scripts/make_synthetic_keypoints.py --out outputs/demo/keypoints.json
python scripts/select_poses.py --config configs/market.yaml \
    --keypoints outputs/demo/keypoints.json --out outputs/demo/selection
python scripts/render_poses.py --selection outputs/demo/selection --out outputs/demo/poses
python scripts/compare_poses.py \
    --camera "Demo C2=outputs/demo/poses/c2,outputs/demo/selection/c2.json" \
    --camera "Demo C3=outputs/demo/poses/c3,outputs/demo/selection/c3.json" \
    --out outputs/demo/compare.png
```

The rendered `outputs/poses/<dataset>/c<cam>/` folders are used as the pose directory of the
generator for the gallery images of that camera. Each `c<cam>.json` keeps the direction shares,
the allocation, rejection counts and the selected poses with their keypoints.

## Repository layout

```
camera-aware-pose-selection/
├── configs/
│   ├── market.yaml, duke.yaml, msmt.yaml   # gallery placeholder, cameras, thresholds, K, seed
│   ├── render.yaml                         # canvas 320, 64x128, reference layout, backend
│   └── common_poses.yaml                   # directions of the 8 common reference poses
├── docs/figures/common_vs_camera_poses.png
├── scripts/
│   ├── select_poses.py                     # sampling -> keypoints -> allocation -> K-means
│   ├── render_poses.py                     # reference-matched rendering (BGR files)
│   ├── compare_poses.py                    # side-by-side check figure
│   └── make_synthetic_keypoints.py         # CPU demo input
├── src/campose/
│   ├── allocate.py                         # direction shares -> 8 poses
│   ├── select.py                           # per-direction K-means, nearest real pose
│   ├── render.py                           # place / render_rgb / save_like_reference / draw_bodypose
│   ├── compare.py                          # comparison figure
│   ├── data.py                             # file-name parsers, sampling, keypoint JSON
│   ├── synthetic.py                        # synthetic skeletons (tests, demo)
│   └── posecore/                           # pose.py, quality.py, direction.py (copied from walking-direction-analyzer)
└── tests/
```

## Citation, license, related repositories

- Pose2ID: C. Yuan et al. *From Poses to Identity: Training-Free Person Re-Identification via
  Feature Centralization.* CVPR 2025. https://github.com/yuanc3/Pose2ID (MIT, (c) 2025 yuanc3).
  The figure's top row shows its reference pose images (skeletons).
- DWPose: Z. Yang, A. Zeng, C. Yuan, Y. Li. *Effective Whole-body Pose Estimation with Two-stages
  Distillation.* ICCV Workshops 2023. Apache-2.0.
- OpenPose: Z. Cao et al. *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity
  Fields.* TPAMI 2019 (18-joint body layout and skeleton drawing style).
- controlnet_aux (Hugging Face), Apache-2.0: DWPose wrapper; `campose/render.py::draw_bodypose_builtin`
  is a copy of its `dwpose/util.py::draw_bodypose` (itself derived from ControlNet and
  pytorch-openpose), kept unchanged for pixel-identical output.
- CLIP-ReID: S. Li, L. Sun, Q. Li. *CLIP-ReID: Exploiting Vision-Language Model for Image
  Re-Identification without Concrete Text Labels.* AAAI 2023. https://github.com/Syliz517/CLIP-ReID
  (MIT, (c) 2023 Syliz517). Re-ID model of the reported results.
- Market-1501: L. Zheng et al. ICCV 2015. DukeMTMC-reID: E. Ristani et al. ECCV Workshops 2016;
  Z. Zheng et al. ICCV 2017. MSMT17: L. Wei et al. CVPR 2018.

License: [MIT](LICENSE) (c) 2026 ehlee, except the copied `draw_bodypose` (Apache-2.0, see the
header in `src/campose/render.py`). Datasets, pose-model weights and Pose2ID's reference poses keep
their own licenses and are not redistributed (apart from the skeleton figure above).

Related repositories:
- [2026_석사논문_ReID_평가벤치](../2026_석사논문_ReID_평가벤치/)
- [2026_석사논문_ReID_카메라특징보정](../2026_석사논문_ReID_카메라특징보정/)
- [2026_석사논문_보행방향분석](../2026_석사논문_보행방향분석/)
- [2026_석사논문_재현가능한_Diffusion생성](../2026_석사논문_재현가능한_Diffusion생성/)
- [2026_석사논문_카메라별_자세선정](../2026_석사논문_카메라별_자세선정/) (this repository)
- [2026_석사논문_MSMT17_변환기](../2026_석사논문_MSMT17_변환기/)
