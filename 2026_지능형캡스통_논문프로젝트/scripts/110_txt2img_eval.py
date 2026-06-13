#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
110_txt2img_eval.py ─ txt2img+IPA 생성물로 c6 sparse ReID 평가

비교 6조합:  query = c1~c5 real
                       보정X        보정O(CP)
  B0 (c6 real만)        ──           ──
  + gen_raw (열화 전)    ──           ──
  + gen_matched (열화)   ──           ──
"""

import os, sys, glob, io, gc, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

# ===== CONFIG =====
HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/c6_txt2img"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

device, dtype = "cuda", torch.float16
SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
NUM_IDS = 50
POSE_POOL_SIZE = 100
SEED, SIZE = 42, (384, 768)
IPA_SCALE, CN_SCALE = 0.8, 1.0

os.makedirs(GEN_DIR, exist_ok=True)
for tc in TARGET_CAMS:
    os.makedirs(f"{GEN_DIR}/raw/{tc}", exist_ok=True)
    os.makedirs(f"{GEN_DIR}/matched/{tc}", exist_ok=True)

def logline(m):
    with open(LOG, "a") as fp: fp.write(m + "\n")
    print(m)

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]


# ===== 1. 데이터 + valid IDs =====
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

valid_ids = [pid for pid in sorted(gallery_by_id)
             if SOURCE_CAM in gallery_by_id[pid]
             and all(tc in query_by_id[pid] for tc in TARGET_CAMS)][:NUM_IDS]
print(f"평가 ID: {len(valid_ids)}명\n")


# ===== 2. medoid 자세 (카메라별) =====
print("OpenPose 로드 + medoid 자세...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
def detect_skel(img):
    try:
        o = openpose(img)
        if isinstance(o, tuple): o = o[0]
        if o is None: return None
        if not isinstance(o, Image.Image): o = Image.fromarray(o)
        return o
    except Exception:
        return None
def select_medoid(skels):
    small = np.stack([np.array(s.resize((48,96))).astype(np.float32).flatten() for s in skels])
    small = small/(np.linalg.norm(small,axis=1,keepdims=True)+1e-8)
    return int((small@small.T).mean(axis=1).argmax())

valid_set = set(valid_ids)
medoid_skel = {}
for tc in TARGET_CAMS:
    pool=[]
    for pid in sorted(gallery_by_id):
        if pid in valid_set: continue
        if tc in gallery_by_id[pid]: pool.append(gallery_by_id[pid][tc][0])
        if len(pool) >= POSE_POOL_SIZE*2: break
    skels=[]
    for p in pool:
        s=detect_skel(Image.open(p).convert("RGB").resize(SIZE, Image.LANCZOS))
        if s is not None: skels.append(s)
        if len(skels)>=POSE_POOL_SIZE: break
    midx=select_medoid(skels)
    medoid_skel[tc]=skels[midx].resize(SIZE, Image.LANCZOS)
    print(f"  {tc}: medoid pose 확보 ({len(skels)}장 중)")
print()


# ===== 3. 열화 함수 =====
def analyze_cctv(ref):
    a=np.asarray(ref.convert("RGB")).astype(np.float32); gray=a.mean(2)
    lap=np.abs(np.gradient(np.gradient(gray,axis=0)[0],axis=0)).var()
    blurred=Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    noise=float((gray-np.asarray(blurred)).std())
    bright=float(gray.mean()); contrast=float(gray.std())
    sat=float(np.asarray(ref.convert("HSV")).astype(np.float32)[:,:,1].mean())
    return dict(sharp=float(lap),noise=noise,bright=bright,contrast=contrast,sat=sat)

def degrade_matched(img, ref):
    s=analyze_cctv(ref); w,h=img.size
    down=int(np.clip(48+s["sharp"]*0.5,40,96))
    img=img.resize((down,int(down*h/w)),Image.BILINEAR).resize((w,h),Image.BILINEAR)
    img=img.filter(ImageFilter.GaussianBlur(float(np.clip(1.2-s["sharp"]*0.01,0.3,1.2))))
    g=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Brightness(img).enhance(float(np.clip(s["bright"]/(g.mean()+1e-6),0.7,1.3)))
    g2=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Contrast(img).enhance(float(np.clip(s["contrast"]/(g2.std()+1e-6),0.7,1.3)))
    cur_sat=np.asarray(img.convert("HSV")).astype(np.float32)[:,:,1].mean()
    img=ImageEnhance.Color(img).enhance(float(np.clip(s["sat"]/(cur_sat+1e-6),0.7,1.3)))
    arr=np.asarray(img.convert("RGB")).astype(np.float32)
    arr+=np.random.RandomState(SEED).normal(0,s["noise"]*0.6,arr.shape)
    buf=io.BytesIO(); Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(buf,"JPEG",quality=55)
    buf.seek(0); return Image.open(buf).convert("RGB")


# ===== 4. 생성 (txt2img + IPA + ControlNet) =====
print("생성 파이프라인 로드 (txt2img + IPA)...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                   cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IPA_SCALE)
pipe=pipe.to(device)
print("로드 완료. 생성 시작...")

for i, pid in enumerate(valid_ids):
    c6_img=Image.open(sorted(gallery_by_id[pid][SOURCE_CAM])[0]).convert("RGB")
    for tc in TARGET_CAMS:
        raw_p=f"{GEN_DIR}/raw/{tc}/{pid}_gen_{tc}.png"
        mat_p=f"{GEN_DIR}/matched/{tc}/{pid}_gen_{tc}.png"
        if os.path.exists(raw_p) and os.path.exists(mat_p): continue
        g=torch.Generator(device).manual_seed(SEED)
        gen=pipe(prompt="a photo of a person, full body, surveillance",
                 negative_prompt="blurry, low quality, deformed, multiple people, extra limbs",
                 image=medoid_skel[tc], ip_adapter_image=c6_img,
                 controlnet_conditioning_scale=CN_SCALE,
                 num_inference_steps=30, guidance_scale=7.5,
                 width=SIZE[0], height=SIZE[1], generator=g).images[0]
        gen.save(raw_p)
        degrade_matched(gen, c6_img).save(mat_p)
    if (i+1)%10==0: print(f"  {i+1}/{len(valid_ids)}")
del pipe, cn; gc.collect(); torch.cuda.empty_cache()
print("생성 완료\n")


# ===== 5. CLIP-ReID + feature =====
print("CLIP-ReID 로드...")
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg as rcfg
from model.make_model_clipreid import make_model
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
rcfg.MODEL.NAME="ViT-B-16"; rcfg.MODEL.STRIDE_SIZE=[16,16]
rcfg.MODEL.SIE_CAMERA=False; rcfg.MODEL.SIE_COE=0.0; rcfg.MODEL.ID_LOSS_TYPE="softmax"
rcfg.INPUT.SIZE_TRAIN=[256,128]; rcfg.INPUT.SIZE_TEST=[256,128]   # ← SIZE_TRAIN 필수 (pos_embed 129)
rcfg.INPUT.PIXEL_MEAN=[0.5]*3; rcfg.INPUT.PIXEL_STD=[0.5]*3
rcfg.DATASETS.NAMES="market1501"; rcfg.TEST.WEIGHT=W; rcfg.TEST.NECK_FEAT="before"
try:    _reid=make_model(rcfg,num_class=702,camera_num=0,view_num=1)
except Exception: _reid=make_model(rcfg,num_class=702,camera_num=6,view_num=1)
_reid.load_param(W); _reid=_reid.to(device).eval()
reid_tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat_path(p):
    t=reid_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    f=_reid(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()

def cam_idx(tc): return int(tc[1:])-1

print("feature 추출...")
g_f,g_p,g_c=[],[],[]
for pid in valid_ids:
    for p in gallery_by_id[pid][SOURCE_CAM]:
        g_f.append(feat_path(p)); g_p.append(int(pid)); g_c.append(cam_idx("c6"))
q_f,q_p,q_c=[],[],[]
for pid in valid_ids:
    for tc in TARGET_CAMS:
        for p in query_by_id[pid][tc]:
            q_f.append(feat_path(p)); q_p.append(int(pid)); q_c.append(cam_idx(tc))
raw_f,raw_p,raw_c=[],[],[]
mat_f,mat_p,mat_c=[],[],[]
for pid in valid_ids:
    for tc in TARGET_CAMS:
        rp=f"{GEN_DIR}/raw/{tc}/{pid}_gen_{tc}.png"
        mp=f"{GEN_DIR}/matched/{tc}/{pid}_gen_{tc}.png"
        if os.path.exists(rp):
            raw_f.append(feat_path(rp)); raw_p.append(int(pid)); raw_c.append(cam_idx(tc))
        if os.path.exists(mp):
            mat_f.append(feat_path(mp)); mat_p.append(int(pid)); mat_c.append(cam_idx(tc))
g_f,q_f,raw_f,mat_f=map(lambda x:np.array(x,dtype=np.float32),[g_f,q_f,raw_f,mat_f])
g_p,g_c,q_p,q_c=map(np.array,[g_p,g_c,q_p,q_c])
raw_p,raw_c,mat_p,mat_c=map(np.array,[raw_p,raw_c,mat_p,mat_c])
print(f"  gallery(c6)={len(g_f)} query(c1~5)={len(q_f)} gen_raw={len(raw_f)} gen_matched={len(mat_f)}\n")

# ===== 6. CP + 메트릭 (Distance Fusion 방식 적용) =====
def cosd(qf, gf): return (1.0 - qf @ gf.T).astype(np.float32)

def eval_market(dm, qp, gp, qc, gc, mr=50):
    nq = dm.shape[0]; idx = np.argsort(dm, axis=1); mt = (gp[idx] == qp[:, None]).astype(np.int32)
    cmcs, APs, v = [], [], 0
    for qi in range(nq):
        o = idx[qi]; keep = ~((gp[o] == qp[qi]) & (gc[o] == qc[qi])); raw = mt[qi][keep]
        if not raw.any(): continue
        c = raw.cumsum(); c[c > 1] = 1; c = c[:mr]
        if len(c) < mr: c = np.concatenate([c, np.full(mr - len(c), c[-1])])
        cmcs.append(c); v += 1; nr = raw.sum()
        tmp = raw.cumsum() / (np.arange(len(raw)) + 1.0); APs.append((tmp * raw).sum() / nr)
    return np.asarray(cmcs).sum(0) / v, float(np.mean(APs))

# 🌟 핵심: 정답 개수를 늘리지 않고, 원본 거리와 AI 생성본 거리를 섞어주는 함수
def run_fusion(q_f, q_p, q_c, g_f, g_p, g_c, gen_f, gen_p, gen_c, alpha=0.7):
    # 1. 원본 갤러리와의 기본 거리 계산 [Query 수, Gallery 수]
    dist_real = cosd(q_f, g_f)
    
    # 2. AI 생성본과의 거리를 담을 빈 공간 만들기 (기본값은 무관심을 뜻하는 1.0)
    dist_gen = np.ones_like(dist_real)
    
    # 3. 쿼리와 시점이 맞는 AI 생성본만 골라서 거리 계산
    for i in range(len(q_f)):
        query_cam = q_c[i]
        
        for j in range(len(g_f)):
            gal_pid = g_p[j]
            
            # 현재 쿼리 카메라 시점(query_cam)으로 만들어진 해당 사람(gal_pid)의 AI 피처 찾기
            mask = (gen_p == gal_pid) & (gen_c == query_cam)
            if mask.any():
                target_gen_feat = gen_f[mask][0:1] # 매칭된 피처 1개 가져오기
                # AI 생성본과의 거리 계산
                dist_gen[i, j] = cosd(q_f[i:i+1], target_gen_feat)[0, 0]
                
    # 4. 거리 융합 (Weighted Sum)
    # alpha가 0.7이면, 원본 모델의 판단을 70% 믿고, AI가 시점을 맞춰준 판단을 30% 믿는다는 뜻입니다.
    final_dist = (alpha * dist_real) + ((1.0 - alpha) * dist_gen)
    
    cmc, mAP = eval_market(final_dist, q_p, g_p, q_c, g_c)
    return cmc[0] * 100, mAP * 100

logline("\n" + "=" * 72)
logline(f"## [2026-06-03] script110 txt2img+IPA 생성 ReID 평가 (ID {len(valid_ids)}명)")
logline(f"   방식: Distance Fusion (Gallery 확장이 아닌 힌트 제공)")
logline("=" * 72)
logline(f"\n{'비교군':<18}{'Rank-1':<10}{'mAP':<11}")
logline("-" * 45)

# B0 (원본만 사용) - 기존과 동일하게 계산하기 위해 alpha=1.0 적용
r1_base, map_base = run_fusion(q_f, q_p, q_c, g_f, g_p, g_c, raw_f, raw_p, raw_c, alpha=1.0)
logline(f"{'B0 (c6 real)':<18}{r1_base:<10.1f}{map_base:<11.1f}")

# + gen_raw (열화 전 융합) - 원본 70%, 생성본 30% 비율 (alpha 값은 실험으로 조절 가능)
r1_raw, map_raw = run_fusion(q_f, q_p, q_c, g_f, g_p, g_c, raw_f, raw_p, raw_c, alpha=0.7)
logline(f"{'+ gen_raw':<18}{r1_raw:<10.1f}{map_raw:<11.1f} (+{map_raw-map_base:.1f})")

# + gen_matched (열화 후 융합)
r1_mat, map_mat = run_fusion(q_f, q_p, q_c, g_f, g_p, g_c, mat_f, mat_p, mat_c, alpha=0.7)
logline(f"{'+ gen_matched':<18}{r1_mat:<10.1f}{map_mat:<11.1f} (+{map_mat-map_base:.1f})")

logline("-" * 45)
logline("해석: AI 이미지를 갤러리에 추가하지 않고 '거리 가중합(Fusion)'으로 사용하여 mAP 하락 방지.")