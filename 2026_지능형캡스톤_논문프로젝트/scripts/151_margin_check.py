#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
151_margin_check.py ─ centroid 융합의 margin 검증 (5명, mAP 평가 전 사전 점검)

150에서 cosine(정답↔쿼리)은 alpha=0.6에서 +0.011 올랐음.
그러나 mAP가 오르려면 정답뿐 아니라 '오답과의 cosine'도 같이 안 올라야 함.
핵심 지표 = margin = (정답 cosine) - (오답 평균 cosine).
margin 이 alpha 따라 벌어지면 → mAP 향상 기대. 안 벌어지면 → mAP 평가 무의미.

추가로 mini-rank: 각 query를 5명 갤러리(각자 c1 원본 = gen_centroid로 보강) 중
순위 매겨 Rank-1 적중도 함께 출력.

150과 동일 환경/로딩. 생성은 다시 돌림(또는 150 캐시 재사용 가능하면 그쪽).
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

N_POSES = 5; N_PER_POSE = 2
IP_SCALE = 0.8; CN_SCALE = 1.0; SIM_KEEP = 0.0
ALPHAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
# 150에서 생성해둔 이미지를 재사용해 시간 절약 (없으면 재생성)
REUSE_CACHED_GEN = True


def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]


print("=" * 60); print("데이터 + 모델 로드"); print("=" * 60)
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


c1_real = {pid: feat_path(gby[pid][SRC_CAM][0]) for pid in ids}     # 정답(c1 갤러리)
c6_query = {pid: feat_path(qby[pid]["c6"][0]) for pid in ids}       # 쿼리(c6)
base_q = np.mean([c1_real[p] @ c6_query[p] for p in ids])
print(f"기준선 정답↔쿼리: {base_q:.3f}")

# ── 생성 centroid: 150 캐시 재사용 우선, 없으면 재생성 ────────────────
from controlnet_aux import OpenposeDetector
gen_centroid = {}
need_gen = []
if REUSE_CACHED_GEN:
    for pid in ids:
        cached = sorted(glob.glob(f"{OUT}/{pid}_pose*.png"))
        if cached:
            feats = [feat_path(p) for p in cached]
            c = np.mean(feats, axis=0); c /= (np.linalg.norm(c) + 1e-12)
            gen_centroid[pid] = c
            print(f"  {pid}: 캐시 {len(cached)}장 재사용")
        else:
            need_gen.append(pid)
else:
    need_gen = list(ids)

if need_gen:
    op = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

    def to_pose(o):
        if isinstance(o, tuple): o = o[0]
        if o is None: return None
        if not isinstance(o, Image.Image): o = Image.fromarray(o)
        return o.resize((512, 768), Image.LANCZOS)

    def sample_c6_poses(pid, k=N_POSES):
        paths = list(qby[pid].get("c6", [])) + list(gby[pid].get("c6", []))
        if len(paths) < k:
            for opid in ids:
                if opid == pid: continue
                paths += list(gby[opid].get("c6", []))[:2]
        out = []
        for p in paths[:k]:
            try:
                img = Image.open(p).convert("RGB").resize((512, 768), Image.LANCZOS)
                ps = to_pose(op(img))
                if ps is not None: out.append(ps)
            except Exception:
                continue
        return out

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

    for pid in need_gen:
        qimg = Image.open(gby[pid][SRC_CAM][0]).convert("RGB").resize((512, 1024), Image.LANCZOS)
        q_orig_f = c1_real[pid]; poses = sample_c6_poses(pid); feats = []
        for j, ps in enumerate(poses):
            try:
                imgs = pipe(prompt=PROMPT, negative_prompt=NEG, image=ps, ip_adapter_image=qimg,
                            controlnet_conditioning_scale=CN_SCALE, num_inference_steps=30,
                            guidance_scale=7.5, width=512, height=768,
                            num_images_per_prompt=N_PER_POSE).images
            except Exception:
                continue
            best_f, best_s, best_im = None, -1, None
            for im in imgs:
                f = feat_img(im); s = float(q_orig_f @ f)
                if s > best_s: best_s, best_f, best_im = s, f, im
            if best_f is not None and best_s >= SIM_KEEP:
                feats.append(best_f); best_im.save(f"{OUT}/{pid}_pose{j}.png")
        if feats:
            c = np.mean(feats, axis=0); c /= (np.linalg.norm(c) + 1e-12); gen_centroid[pid] = c
        else:
            gen_centroid[pid] = c1_real[pid]
        print(f"  {pid}: {len(feats)}장 융합")
    del pipe, cn; torch.cuda.empty_cache()

# ── margin + mini-rank 측정 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("margin 검증 (정답 - 오답). margin 이 벌어져야 mAP 향상 기대")
print("=" * 60)
print(f"{'alpha':<7}{'정답cos':<10}{'오답cos':<10}{'margin':<10}{'Rank1':<8}")
print("-" * 45)

# baseline margin (원본 c1 갤러리만, 융합 없음)
def eval_alpha(a):
    # 각 pid 의 갤러리 표현 = alpha*원본 + (1-a)*centroid
    gallery = {}
    for pid in ids:
        g = a * c1_real[pid] + (1 - a) * gen_centroid[pid]
        gallery[pid] = g / (np.linalg.norm(g) + 1e-12)
    pos_list, neg_list, r1 = [], [], 0
    for q in ids:
        qf = c6_query[q]
        sims = {pid: float(qf @ gallery[pid]) for pid in ids}
        pos = sims[q]
        neg = np.mean([sims[p] for p in ids if p != q])
        pos_list.append(pos); neg_list.append(neg)
        ranked = sorted(ids, key=lambda p: -sims[p])
        if ranked[0] == q: r1 += 1
    return np.mean(pos_list), np.mean(neg_list), r1 / len(ids)


base_pos, base_neg, base_r1 = eval_alpha(1.0)
rows = []
for a in ALPHAS:
    pos, neg, r1 = eval_alpha(a)
    margin = pos - neg
    rows.append((a, pos, neg, margin, r1))
    print(f"{a:<7.1f}{pos:<10.3f}{neg:<10.3f}{margin:<10.3f}{r1*100:<8.0f}")

print("-" * 45)
base_margin = base_pos - base_neg
best = max(rows, key=lambda r: r[3])  # margin 최대
print(f"\nbaseline(alpha=1.0) margin = {base_margin:.3f}")
print(f"최고 margin: alpha={best[0]:.1f}, margin={best[3]:.3f} (Δ={best[3]-base_margin:+.3f})")
if best[3] > base_margin + 1e-4 and best[0] < 1.0:
    print("→ 융합이 정답-오답 간격을 벌림. mAP 향상 기대. 전체 query 평가로 진행 권장.")
else:
    print("→ margin 이 안 벌어짐(또는 alpha=1.0이 최고). cosine 상승은 전역적이라 mAP 무의미.")
    print("  자세 융합은 접고 '원인 규명' 발표 프레임 권장.")
print("\n주의: 5명 mini-rank는 통계적 신뢰 낮음. margin 방향만 참고하고")
print("      양성이면 반드시 전체 query 정식 mAP로 확인할 것.")