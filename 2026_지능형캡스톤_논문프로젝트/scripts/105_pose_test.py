#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
105_pose_test.py ─ 자세 변환 sanity check (3 모델 × 2 조합 grid 비교)

목적:
  44b/44c 의 strength=0.4 가 자세 변환 실패 → strength/cn_scale 재조정 필요.
  ID 3명으로 빠르게 눈 검증.

대표 자세:
  각 타겟 카메라(c1~c5)에서 평가 ID 제외 100명의 OpenPose skeleton 수집 →
  medoid 1개 선정 (다른 모든 skeleton 과의 평균 코사인 유사도 최대).

비교 조합 (각 모델별로):
  curr : strength=0.4, cn_scale=0.8 (현재 망한 값)
  rec  : strength=0.65, cn_scale=1.0 (추천)

비교 모델: SD1.5 / SD1.5+IPA / SDXL

출력:
  outputs/pose_test/representative_skeletons/ : medoid skeleton + 원본
  outputs/pose_test/{model}_{combo}/grid_pid*.png : 비교 그리드
    레이아웃: [c6 원본] [c1 real] [c1 gen] [c2 real] [c2 gen] ... [c5 real] [c5 gen]
"""

import os, sys, glob, gc, csv, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel, DDIMScheduler, AutoencoderKL,
)
from controlnet_aux import OpenposeDetector

import torchvision.transforms as T
import torch.nn as nn

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OUT_DIR = f"{PROJECT_DIR}/outputs/pose_test"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
dtype = torch.float16

SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
NUM_TEST_IDS = 3
POSE_POOL_SIZE = 100
SEED = 42

COMBOS = [
    dict(name="curr", strength=0.4,  cn_scale=0.8),
    dict(name="rec",  strength=0.65, cn_scale=1.0),
]
MODELS = ["sd15", "sd15_ipa", "sdxl"]

os.makedirs(f"{OUT_DIR}/representative_skeletons", exist_ok=True)
for m in MODELS:
    for c in COMBOS:
        os.makedirs(f"{OUT_DIR}/{m}_{c['name']}", exist_ok=True)


def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]


# =====================================================================
# 1. 데이터 로드 + valid IDs
# =====================================================================
print("데이터 로드...")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1', '0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# 5 cam 모두에 정답 query 있는 ID만 (grid 비교 완성도 위해)
valid_ids = [pid for pid in sorted(gallery_by_id.keys())
             if SOURCE_CAM in gallery_by_id[pid]
             and all(tc in query_by_id[pid] for tc in TARGET_CAMS)]
valid_ids = valid_ids[:NUM_TEST_IDS]
print(f"테스트 ID: {valid_ids}\n")


# =====================================================================
# 2. OpenPose 안전 호출 + medoid 자세 선정
# =====================================================================
print("OpenPose 로드...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

def detect_skel(pil_img):
    """OpenPose 안전 호출. tuple/None/ndarray 다 처리."""
    try:
        out = openpose(pil_img)
        if isinstance(out, tuple): out = out[0]
        if out is None: return None
        if not isinstance(out, Image.Image):
            out = Image.fromarray(out)
        return out
    except Exception:
        return None


def select_medoid(skel_list):
    """skeleton 이미지 리스트 → medoid 인덱스.
    축소 이미지(48x96) pixel 공간에서 평균 코사인 유사도 최대인 자세."""
    small = np.stack([
        np.array(s.resize((48, 96))).astype(np.float32).flatten()
        for s in skel_list
    ])
    norms = np.linalg.norm(small, axis=1, keepdims=True) + 1e-8
    small = small / norms
    sims = small @ small.T
    return int(sims.mean(axis=1).argmax())


valid_set = set(valid_ids)
print(f"각 타겟 카메라에서 {POSE_POOL_SIZE}명 자세 수집 + medoid 선정...")
medoid_paths = {}
for tc in TARGET_CAMS:
    # pool: 평가 ID 제외
    pool = []
    for pid in sorted(gallery_by_id.keys()):
        if pid in valid_set: continue
        if tc in gallery_by_id[pid]:
            pool.append(gallery_by_id[pid][tc][0])
        if len(pool) >= POSE_POOL_SIZE * 2: break  # 일부 detect 실패 대비

    skels, srcs = [], []
    for p in pool:
        img = Image.open(p).convert("RGB").resize((384, 768), Image.LANCZOS)
        s = detect_skel(img)
        if s is None: continue
        skels.append(s); srcs.append(p)
        if len(skels) >= POSE_POOL_SIZE: break

    if not skels:
        print(f"  {tc}: medoid 추출 실패!"); continue
    midx = select_medoid(skels)
    medoid_paths[tc] = srcs[midx]
    # 시각화 저장
    skels[midx].save(f"{OUT_DIR}/representative_skeletons/{tc}_medoid_skel.png")
    Image.open(srcs[midx]).convert("RGB").save(
        f"{OUT_DIR}/representative_skeletons/{tc}_medoid_source.png")
    print(f"  {tc}: medoid = {os.path.basename(srcs[midx])} "
          f"(pool {len(skels)}/{POSE_POOL_SIZE})")
print()


# =====================================================================
# 2b. CLIP-ReID 로드 (gen ↔ real ↔ c6 cosine 유사도 측정용)
# =====================================================================
print("CLIP-ReID 로드 (feature 유사도 측정용)...")
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg as reid_cfg
from model.make_model_clipreid import make_model

REID_WEIGHT = f"{CACHE_DIR}/clipreid_duke_nosie.pth"
reid_cfg.MODEL.NAME = "ViT-B-16"; reid_cfg.MODEL.STRIDE_SIZE = [16, 16]
reid_cfg.MODEL.SIE_CAMERA = False; reid_cfg.MODEL.SIE_COE = 0.0
reid_cfg.MODEL.ID_LOSS_TYPE = "softmax"
reid_cfg.INPUT.SIZE_TRAIN = [256, 128]; reid_cfg.INPUT.SIZE_TEST = [256, 128]
reid_cfg.INPUT.PIXEL_MEAN = [0.5]*3; reid_cfg.INPUT.PIXEL_STD = [0.5]*3
reid_cfg.DATASETS.NAMES = "market1501"
reid_cfg.TEST.WEIGHT = REID_WEIGHT; reid_cfg.TEST.NECK_FEAT = "before"
try:    _reid_b = make_model(reid_cfg, num_class=702, camera_num=0, view_num=1)
except Exception: _reid_b = make_model(reid_cfg, num_class=702, camera_num=6, view_num=1)
_reid_b.load_param(REID_WEIGHT)
_reid_b = _reid_b.to(device).eval()

reid_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def reid_feat(pil_img):
    """PIL.Image → L2 정규화된 feature numpy 벡터"""
    t = reid_tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    f = _reid_b(t, cam_label=None)
    if isinstance(f, (list, tuple)):
        f = f[0] if isinstance(f[0], torch.Tensor) else f[-1]
    if f.dim() > 2: f = f.view(f.size(0), -1)
    f = nn.functional.normalize(f.float(), dim=1)
    return f.cpu().numpy().flatten()
print("  ✅ CLIP-ReID 로드 완료\n")


# =====================================================================
# 3. 모델 로더 (각각 메모리 절약)
# =====================================================================
def load_sd15():
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        cache_dir=CACHE_DIR, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe, (384, 768)

def load_sd15_ipa():
    pipe, size = load_sd15()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.8)
    # attention_slicing은 IPA와 충돌 → 빼기
    return pipe, size

def load_sdxl():
    cn = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        cache_dir=CACHE_DIR, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        cache_dir=CACHE_DIR, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=cn, vae=vae,
        cache_dir=CACHE_DIR, torch_dtype=dtype, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe, (512, 1024)

LOADERS = {"sd15": load_sd15, "sd15_ipa": load_sd15_ipa, "sdxl": load_sdxl}


# =====================================================================
# 4. 생성
# =====================================================================
def generate_one(pipe, model_name, base_img, pose_ref_img, strength, cn_scale, size):
    skel = detect_skel(pose_ref_img.resize(size, Image.LANCZOS))
    if skel is None: return None
    skel = skel.resize(size, Image.LANCZOS)
    base = base_img.resize(size, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(SEED)
    kwargs = dict(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=base, control_image=skel,
        strength=strength,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=cn_scale,
        generator=gen,
        width=size[0], height=size[1],
    )
    if model_name == "sd15_ipa":
        kwargs["ip_adapter_image"] = base
    return pipe(**kwargs).images[0]


# =====================================================================
# 5. Grid 만들기 (한 ID = 한 줄)
#    [c6] [c1 real] [c1 gen] [c2 real] [c2 gen] ... [c5 real] [c5 gen]
# =====================================================================
def make_grid(pid, c6_path, real_paths, gen_imgs, sim_table, label):
    """sim_table: dict {tc: (gen_real_sim, gen_c6_sim)} or None"""
    cell_w, cell_h = 180, 360
    title_h, label_h = 32, 26
    sim_h = 30   # gen 컷 아래 수치 표시용
    n_cols = 11
    total_w = cell_w * n_cols + 10
    total_h = title_h + label_h + cell_h + sim_h + 10
    grid = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(grid)
    try:
        font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = font_s = font_t = ImageFont.load_default()
    draw.text((10, 6), f"PID {pid}  |  {label}", fill='black', font=font_t)

    cols = [("c6 source", Image.open(c6_path).convert("RGB"), None)]
    for tc in TARGET_CAMS:
        real = Image.open(real_paths[tc]).convert("RGB") if tc in real_paths else None
        cols.append((f"{tc} real", real, None))
        gen = gen_imgs.get(tc)
        sims = sim_table.get(tc) if sim_table else None
        cols.append((f"{tc} gen", gen, sims))

    for i, (lbl, img, sims) in enumerate(cols):
        x = i * cell_w + 5
        color = 'navy' if 'gen' in lbl else ('darkred' if 'real' in lbl else 'black')
        draw.text((x, title_h + 2), lbl, fill=color, font=font)
        if img is not None:
            img_r = img.resize((cell_w - 5, cell_h), Image.LANCZOS)
            grid.paste(img_r, (x, title_h + label_h))
        if sims is not None:
            gen_real, gen_c6 = sims
            y = title_h + label_h + cell_h + 4
            # gen↔real: 클수록 좋음(파랑), gen↔c6: 작을수록 좋음(자세변환↑, 회색)
            draw.text((x, y),       f"vs real: {gen_real:.3f}", fill='navy', font=font_s)
            draw.text((x, y + 14),  f"vs c6  : {gen_c6:.3f}",   fill='gray', font=font_s)
    return grid


# =====================================================================
# 6. 실행
# =====================================================================
csv_rows = []   # 전체 수치표 (CSV로 저장)

for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    try:
        pipe, size = LOADERS[model_name]()
    except Exception as e:
        print(f"  [!] {model_name} 로드 실패: {e}")
        continue

    for combo in COMBOS:
        out_subdir = f"{OUT_DIR}/{model_name}_{combo['name']}"
        print(f"\n  조합 {combo['name']}: strength={combo['strength']}, cn_scale={combo['cn_scale']}")
        for pid in valid_ids:
            c6_path = sorted(gallery_by_id[pid][SOURCE_CAM])[0]
            base_img = Image.open(c6_path).convert("RGB")
            real_paths = {tc: sorted(query_by_id[pid][tc])[0] for tc in TARGET_CAMS}

            # 미리 feature 추출 (재사용)
            c6_f   = reid_feat(base_img)
            real_f = {tc: reid_feat(Image.open(real_paths[tc])) for tc in TARGET_CAMS}

            gen_imgs, sim_table = {}, {}
            for tc in TARGET_CAMS:
                if tc not in medoid_paths: continue
                pose_ref = Image.open(medoid_paths[tc]).convert("RGB")
                try:
                    g = generate_one(pipe, model_name, base_img, pose_ref,
                                     combo['strength'], combo['cn_scale'], size)
                    if g is not None:
                        gen_imgs[tc] = g
                        g.save(f"{out_subdir}/{pid}_gen_{tc}.png")
                        # ── 수치 측정 ──
                        gf = reid_feat(g)
                        gen_real_sim = float(gf @ real_f[tc])
                        gen_c6_sim   = float(gf @ c6_f)
                        real_c6_sim  = float(real_f[tc] @ c6_f)
                        sim_table[tc] = (gen_real_sim, gen_c6_sim)
                        csv_rows.append([model_name, combo['name'], pid, tc,
                                         combo['strength'], combo['cn_scale'],
                                         round(real_c6_sim, 4),
                                         round(gen_real_sim, 4),
                                         round(gen_c6_sim, 4)])
                except Exception as e:
                    print(f"    [!] pid={pid} tc={tc}: {e}")

            label = f"{model_name} | str={combo['strength']}, cn={combo['cn_scale']}"
            grid = make_grid(pid, c6_path, real_paths, gen_imgs, sim_table, label)
            grid.save(f"{out_subdir}/grid_pid{pid}.png")
            print(f"    pid={pid}: grid 저장 (gen-real sim 평균 "
                  f"{np.mean([s[0] for s in sim_table.values()]):.3f})")

    del pipe; gc.collect(); torch.cuda.empty_cache()

# ===== CSV 저장 + 요약 표 =====
csv_path = f"{OUT_DIR}/sim_table.csv"
with open(csv_path, "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["model", "combo", "pid", "tc", "strength", "cn_scale",
                "real_vs_c6", "gen_vs_real", "gen_vs_c6"])
    w.writerows(csv_rows)

# 모델×조합 평균 요약
print("\n" + "=" * 70)
print("요약 (조합별 평균)")
print("=" * 70)
print(f"{'model':<10}{'combo':<6}{'gen_vs_real ↑':<16}{'gen_vs_c6 ↓':<16}{'(real_vs_c6)':<14}")
print("-" * 70)
arr = np.array([[r[6], r[7], r[8]] for r in csv_rows], dtype=float) if csv_rows else None
keys = sorted(set((r[0], r[1]) for r in csv_rows))
for mn, cn in keys:
    sel = [r for r in csv_rows if r[0] == mn and r[1] == cn]
    if not sel: continue
    rvc = np.mean([r[6] for r in sel])
    gvr = np.mean([r[7] for r in sel])
    gvc = np.mean([r[8] for r in sel])
    print(f"{mn:<10}{cn:<6}{gvr:<16.3f}{gvc:<16.3f}{rvc:<14.3f}")
print("-" * 70)
print("해석:")
print("  gen_vs_real 클수록 좋음 (생성이 정답에 가까움)")
print("  gen_vs_c6 작을수록 좋음 (자세 변환 일어났다 = c6 원본에서 멀어짐)")
print("  real_vs_c6 = 정답조차 c6와 얼마나 다른가 (참고 기준선)")
print(f"\n✅ CSV: {csv_path}")
print(f"✅ Grid 18장: {OUT_DIR}/<model>_<combo>/grid_pid*.png")