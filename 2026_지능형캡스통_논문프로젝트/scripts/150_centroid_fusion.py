#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
150_centroid_fusion.py ─ Pose2ID식 Identity-Guided Centroid 융합 (실 환경 정합본)

핵심 가설:
  query 1장을 "타깃 카메라(c6)의 여러 대표 자세"로 생성 → 생성 피처 centroid →
  자세 성분 상쇄, identity 성분 강화 → 자세 갭 큰 쌍에서 cosine/mAP 향상.
  (기존 '갤러리 추가/단순 concat' ΔmAP≈0 방식과 근본적으로 다름)

검증 1단계: 5명 cosine 측정 (이전 140 스크립트와 동일 프로토콜)
  - alpha 스윕으로 원본 query vs 생성 centroid 혼합 비율별 cosine 관찰
  - 목표선 = base_q (정답↔쿼리). centroid 융합이 이를 넘으면 효과.

모든 가중치 공개 사전학습 (Training-Free). 평가/융합 백본 동일(CLIP-ReID).
"""
import os, sys, glob, traceback, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OUT = f"{PROJECT_DIR}/outputs/centroid_fusion"; os.makedirs(OUT, exist_ok=True)
MARKET = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR = f"{MARKET}/bounding_box_test"; QRY_DIR = f"{MARKET}/query"
W = f"{CACHE_DIR}/clipreid_duke_nosie.pth"
device, dtype = "cuda", torch.float16
SRC_CAM = "c1"; N = 5

# centroid 생성 파라미터
N_POSES = 5          # query당 생성할 타깃 자세 개수
N_PER_POSE = 2       # 자세당 생성 샘플 수 (best 1개 채택)
IP_SCALE = 0.8
CN_SCALE = 1.0
SIM_KEEP = 0.0       # 자세별 생성본 중 query와 cosine 이 값 미만이면 centroid에서 제외 (0=전부 사용)
ALPHAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]  # 1.0=원본만(baseline), 0.0=생성centroid만


def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]


print("=" * 60)
print("데이터 + 5명 + 모델 로드")
print("=" * 60)
gby = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1', '0000'): continue
    gby[pid][cam].append(f)
qby = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid, cam = parse(f); qby[pid][cam].append(f)
ids = [p for p in sorted(qby) if "c6" in qby[p] and SRC_CAM in gby[p]][:N]
print(f"5명: {ids}")

# ── CLIP-ReID (140 스크립트와 동일 로딩) ─────────────────────────────
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg as rc
from model.make_model_clipreid import make_model
rc.MODEL.NAME = "ViT-B-16"; rc.MODEL.STRIDE_SIZE = [16, 16]; rc.MODEL.SIE_CAMERA = False
rc.MODEL.SIE_COE = 0.0; rc.MODEL.ID_LOSS_TYPE = "softmax"
rc.INPUT.SIZE_TRAIN = [256, 128]; rc.INPUT.SIZE_TEST = [256, 128]
rc.INPUT.PIXEL_MEAN = [0.5] * 3; rc.INPUT.PIXEL_STD = [0.5] * 3
rc.DATASETS.NAMES = "market1501"; rc.TEST.WEIGHT = W; rc.TEST.NECK_FEAT = "before"
try:
    _r = make_model(rc, num_class=702, camera_num=0, view_num=1)
except Exception:
    _r = make_model(rc, num_class=702, camera_num=6, view_num=1)
_r.load_param(W); _r = _r.to(device).eval()
rtf = T.Compose([T.Resize([256, 128]), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


@torch.no_grad()
def feat_img(img):
    t = rtf(img.convert("RGB")).unsqueeze(0).to(device)
    f = _r(t, cam_label=None)
    if isinstance(f, (list, tuple)): f = f[0] if isinstance(f[0], torch.Tensor) else f[-1]
    if f.dim() > 2: f = f.view(f.size(0), -1)
    return nn.functional.normalize(f.float(), dim=1).cpu().numpy().flatten()


def feat_path(p): return feat_img(Image.open(p))


c1_real = {pid: feat_path(gby[pid][SRC_CAM][0]) for pid in ids}      # 정답(c1 갤러리)
c6_query = {pid: feat_path(qby[pid]["c6"][0]) for pid in ids}        # 쿼리(c6)
base_q = np.mean([c1_real[p] @ c6_query[p] for p in ids])
print(f"기준선 정답↔쿼리: {base_q:.3f}  ← 목표선")

# ── OpenPose: 타깃 카메라(c6) 자세 N개를 query별로 다양하게 확보 ──────
from controlnet_aux import OpenposeDetector
op = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)


def to_pose(o):
    if isinstance(o, tuple): o = o[0]
    if o is None: return None
    if not isinstance(o, Image.Image): o = Image.fromarray(o)
    return o.resize((512, 768), Image.LANCZOS)


def sample_c6_poses(pid, k=N_POSES):
    """해당 pid 의 c6 이미지들에서 자세 추출. 부족하면 다른 사람 c6 자세로 보충."""
    paths = list(qby[pid].get("c6", [])) + list(gby[pid].get("c6", []))
    # 다양성 위해 다른 id의 c6 갤러리 자세도 풀에 추가
    if len(paths) < k:
        for opid in ids:
            if opid == pid: continue
            paths += list(gby[opid].get("c6", []))[:2]
    paths = paths[:max(k, 1)]
    out = []
    for p in paths[:k]:
        try:
            img = Image.open(p).convert("RGB").resize((512, 768), Image.LANCZOS)
            ps = to_pose(op(img))
            if ps is not None: out.append(ps)
        except Exception:
            continue
    return out


# ── 생성 파이프라인 (140 스크립트와 동일) ────────────────────────────
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                     cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, cache_dir=CACHE_DIR,
    torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IP_SCALE); pipe = pipe.to(device)
PROMPT = "RAW photo of a person, full body, standing, photorealistic, correct anatomy"
NEG = "blurry, deformed, extra limbs, bad anatomy"


@torch.no_grad()
def gen_one(query_img, pose_img):
    return pipe(prompt=PROMPT, negative_prompt=NEG, image=pose_img,
                ip_adapter_image=query_img,
                controlnet_conditioning_scale=CN_SCALE,
                num_inference_steps=30, guidance_scale=7.5,
                width=512, height=768,
                num_images_per_prompt=N_PER_POSE).images


# ── query별 생성 centroid 계산 ───────────────────────────────────────
print("\n" + "=" * 60)
print("생성 centroid 계산 (query를 c6 자세 N개로 생성)")
print("=" * 60)
gen_centroid = {}      # pid -> 생성 피처 centroid (정규화)
for pid in ids:
    src = gby[pid][SRC_CAM][0]
    qimg = Image.open(src).convert("RGB").resize((512, 1024), Image.LANCZOS)
    q_orig_f = c1_real[pid]
    poses = sample_c6_poses(pid)
    if not poses:
        print(f"  {pid}: c6 자세 추출 실패 → 원본 fallback")
        gen_centroid[pid] = q_orig_f
        continue
    feats = []
    for j, ps in enumerate(poses):
        try:
            imgs = gen_one(qimg, ps)
        except Exception:
            print(f"  {pid} pose{j} 생성 실패:\n{traceback.format_exc()[:300]}")
            continue
        best_f, best_s = None, -1
        for k, im in enumerate(imgs):
            f = feat_img(im)
            s = float(q_orig_f @ f)
            if s > best_s:
                best_s, best_f = s, f
                best_im = im
        if best_f is not None and best_s >= SIM_KEEP:
            feats.append(best_f)
            best_im.save(f"{OUT}/{pid}_pose{j}.png")
    if not feats:
        gen_centroid[pid] = q_orig_f
        print(f"  {pid}: 채택 생성본 없음 → 원본 fallback")
    else:
        c = np.mean(feats, axis=0); c = c / (np.linalg.norm(c) + 1e-12)
        gen_centroid[pid] = c
        print(f"  {pid}: {len(feats)}개 자세 융합 완료")

del pipe, cn; torch.cuda.empty_cache()

# ── alpha 스윕: fused = normalize(alpha*원본 + (1-alpha)*centroid) ────
print("\n" + "=" * 60)
print("alpha 스윕 결과 (5명 평균 cosine)")
print("=" * 60)
print(f"{'alpha':<8}{'융합↔쿼리':<14}{'vs 목표선':<12}")
print("-" * 34)
print(f"{'[기준]':<8}{base_q:<14.3f}{'(목표)':<12}")
print("-" * 34)
sweep = {}
for a in ALPHAS:
    sims = []
    for pid in ids:
        fused = a * c1_real[pid] + (1 - a) * gen_centroid[pid]
        fused = fused / (np.linalg.norm(fused) + 1e-12)
        sims.append(float(fused @ c6_query[pid]))
    m = np.mean(sims); sweep[a] = m
    flag = "✓ 효과" if m > base_q else ""
    print(f"{a:<8.1f}{m:<14.3f}{flag:<12}")
print("-" * 34)

best_a = max(sweep, key=sweep.get)
print(f"\n최고 alpha={best_a:.1f}  cosine={sweep[best_a]:.3f}  (목표선 {base_q:.3f})")
if sweep[best_a] > base_q:
    print("→ centroid 융합이 원본 쿼리 매칭을 개선. 다음 단계: 전체 query mAP 평가.")
else:
    print("→ centroid 가 목표선 미달. 자세 정렬만으로는 한계 → 원인 규명 프레임 권장.")
    print("  (sanity: alpha=1.0 값이 base_q와 동일해야 정상. 다르면 코드 점검)")
print(f"\n생성 이미지: {OUT}/")