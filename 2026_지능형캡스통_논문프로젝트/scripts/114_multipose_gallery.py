#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
114_multipose_gallery.py ─ C1→C6 다중자세 갤러리 보강 (1단계 단일쌍)

[설계]  데이터=Market, 모델=Duke학습본 → Duke→Market 크로스도메인
  Query (고정):  C6 real
  Gallery:
    [Baseline]   C1 real 만
    [Proposed]   C1 real + (C1 인물을 C6 대표자세 K개로 변환한 생성본 K장)
  변환:  IPA(외형)=C1 인물(512x1024 리사이즈), ControlNet(포즈)=C6 대표자세 K개
  생성본 ID = 원래 C1 사람 (정답 인정)

[생성 설정] (113 검증 통과)
  IPA scale 0.8, IPA 입력 512x1024 리사이즈, SIZE 384x768,
  CCTV top-down 프롬프트, 강화 negative
  C6 대표자세 K개 = OpenPose skeleton K-means++ 클러스터 medoid

[지표]  Rank-1/5/10 주지표 + mAP, CP 유무
"""

import os, sys, glob, io, gc, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

# ===== CONFIG =====
HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_DIR=f"{PROJECT_DIR}/outputs/c1_multipose_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
LOG=f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
device,dtype="cuda",torch.float16
GALLERY_CAM="c1"; QUERY_CAM="c6"
K_POSES=3                          # C6 대표 자세 개수
POSE_POOL_SIZE=100
SEED=42; SIZE=(384,768); IPA_INPUT=(512,1024); IPA_SCALE=0.8; CN_SCALE=1.0
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
PROMPT="a photo of a person, full body shot, standing, taken from a surveillance CCTV camera, top-down angle, highly detailed, photorealistic"
NEG="cropped, close up, partial body, missing limbs, extra limbs, bad anatomy, deformed, blurry, low resolution, face portrait"

for k in range(K_POSES): os.makedirs(f"{GEN_DIR}/raw/pose{k}",exist_ok=True)
def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)
def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]
def cam_idx(tc): return int(tc[1:])-1


# ===== 1. 데이터 =====
print("데이터 로드...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
query_ids=[p for p in sorted(qby) if QUERY_CAM in qby[p] and GALLERY_CAM in gby[p]]
gallery_ids=[p for p in sorted(gby) if GALLERY_CAM in gby[p]]
print(f"Query(C6, 정답 C1보유): {len(query_ids)}명")
print(f"Gallery(C1 전체): {len(gallery_ids)}명\n")


# ===== 2. C6 대표 자세 K개 (K-means++) =====
print(f"OpenPose + C6 대표자세 {K_POSES}개...")
op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
def detect(img):
    o=op(img)
    if isinstance(o,tuple): o=o[0]
    if o is None: return None
    if not isinstance(o,Image.Image): o=Image.fromarray(o)
    return o
def select_k(vecs,K,seed=42):
    rng=np.random.RandomState(seed); N=len(vecs)
    centers=[vecs[rng.randint(N)]]
    for _ in range(K-1):
        d=np.min([((vecs-c)**2).sum(-1) for c in centers],axis=0)
        centers.append(vecs[rng.choice(N,p=d/d.sum())])
    centers=np.array(centers)
    for _ in range(30):
        a=((vecs[:,None]-centers[None])**2).sum(-1).argmin(1)
        new=np.array([vecs[a==k].mean(0) if (a==k).any() else centers[k] for k in range(K)])
        if np.allclose(new,centers): break
        centers=new
    reps=[]
    for k in range(K):
        m=a==k
        if not m.any(): continue
        idxs=np.where(m)[0]; reps.append(int(idxs[((vecs[idxs]-centers[k])**2).sum(-1).argmin()]))
    return reps

q_set=set(query_ids); pool=[]
for p in sorted(gby):
    if p in q_set: continue
    if QUERY_CAM in gby[p]: pool.append(gby[p][QUERY_CAM][0])
    if len(pool)>=POSE_POOL_SIZE*2: break
skels,vecs=[],[]
for p in pool:
    s=detect(Image.open(p).convert("RGB").resize(SIZE,Image.LANCZOS))
    if s is None: continue
    skels.append(s.resize(SIZE,Image.LANCZOS))
    vecs.append(np.array(s.resize((48,96))).astype(np.float32).flatten())
    if len(skels)>=POSE_POOL_SIZE: break
vecs=np.array(vecs); vecs=vecs/(np.linalg.norm(vecs,axis=1,keepdims=True)+1e-8)
rep_idx=select_k(vecs,K_POSES,SEED)
c6_poses=[skels[i] for i in rep_idx]
print(f"  대표자세 {len(c6_poses)}개 선정 (pool {len(skels)})\n")


# ===== 3. 열화 =====
def analyze(ref):
    a=np.asarray(ref.convert("RGB")).astype(np.float32); g=a.mean(2)
    lap=np.abs(np.gradient(np.gradient(g,axis=0)[0],axis=0)).var()
    bl=Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return dict(sharp=float(lap),noise=float((g-np.asarray(bl)).std()),
                bright=float(g.mean()),contrast=float(g.std()),
                sat=float(np.asarray(ref.convert("HSV")).astype(np.float32)[:,:,1].mean()))
def degrade(img,ref):
    s=analyze(ref); w,h=img.size
    down=int(np.clip(48+s["sharp"]*0.5,40,96))
    img=img.resize((down,int(down*h/w)),Image.BILINEAR).resize((w,h),Image.BILINEAR)
    img=img.filter(ImageFilter.GaussianBlur(float(np.clip(1.2-s["sharp"]*0.01,0.3,1.2))))
    g=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Brightness(img).enhance(float(np.clip(s["bright"]/(g.mean()+1e-6),0.7,1.3)))
    g2=np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img=ImageEnhance.Contrast(img).enhance(float(np.clip(s["contrast"]/(g2.std()+1e-6),0.7,1.3)))
    cs=np.asarray(img.convert("HSV")).astype(np.float32)[:,:,1].mean()
    img=ImageEnhance.Color(img).enhance(float(np.clip(s["sat"]/(cs+1e-6),0.7,1.3)))
    arr=np.asarray(img.convert("RGB")).astype(np.float32)
    arr+=np.random.RandomState(SEED).normal(0,s["noise"]*0.6,arr.shape)
    buf=io.BytesIO(); Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(buf,"JPEG",quality=55)
    buf.seek(0); return Image.open(buf).convert("RGB")


# ===== 4. 생성: C1 인물 → C6 대표자세 K개 =====
print("생성 파이프라인 로드 (txt2img + IPA)...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=cn,cache_dir=CACHE_DIR,
    torch_dtype=dtype,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IPA_SCALE); pipe=pipe.to(device)
print(f"로드 완료. 생성 시작 (C1 {len(gallery_ids)}명 × {K_POSES}자세)...")

for i,pid in enumerate(gallery_ids):
    c1=Image.open(sorted(gby[pid][GALLERY_CAM])[0]).convert("RGB")
    c1_big=c1.resize(IPA_INPUT,Image.LANCZOS)              # 조언①
    for k in range(K_POSES):
        rp=f"{GEN_DIR}/raw/pose{k}/{pid}_gen.png"
        if os.path.exists(rp): continue
        g=torch.Generator(device).manual_seed(SEED+k)
        gen=pipe(prompt=PROMPT,negative_prompt=NEG,image=c6_poses[k],ip_adapter_image=c1_big,
                 controlnet_conditioning_scale=CN_SCALE,num_inference_steps=30,guidance_scale=7.5,
                 width=SIZE[0],height=SIZE[1],generator=g).images[0]
        gen.save(rp)
    if (i+1)%50==0: print(f"  {i+1}/{len(gallery_ids)}")
del pipe,cn; gc.collect(); torch.cuda.empty_cache()
print("생성 완료\n")


# ===== 5. CLIP-ReID (Duke 학습본) =====
print("CLIP-ReID 로드 (Duke→Market)...")
sys.path.insert(0,"/home/ubuntu/CLIP-ReID")
from config import cfg as rc
from model.make_model_clipreid import make_model
rc.MODEL.NAME="ViT-B-16"; rc.MODEL.STRIDE_SIZE=[16,16]; rc.MODEL.SIE_CAMERA=False
rc.MODEL.SIE_COE=0.0; rc.MODEL.ID_LOSS_TYPE="softmax"
rc.INPUT.SIZE_TRAIN=[256,128]; rc.INPUT.SIZE_TEST=[256,128]
rc.INPUT.PIXEL_MEAN=[0.5]*3; rc.INPUT.PIXEL_STD=[0.5]*3
rc.DATASETS.NAMES="market1501"; rc.TEST.WEIGHT=W; rc.TEST.NECK_FEAT="before"
try: _r=make_model(rc,num_class=702,camera_num=0,view_num=1)
except Exception: _r=make_model(rc,num_class=702,camera_num=6,view_num=1)
_r.load_param(W); _r=_r.to(device).eval()
tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat(p):
    t=tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    f=_r(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()

print("feature 추출...")
q_f,q_p,q_c=[],[],[]
for pid in query_ids:
    for p in qby[pid][QUERY_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cam_idx(QUERY_CAM))
g_f,g_p,g_c=[],[],[]
for pid in gallery_ids:
    for p in gby[pid][GALLERY_CAM]:
        g_f.append(feat(p)); g_p.append(int(pid)); g_c.append(cam_idx(GALLERY_CAM))
# 생성본 K개 (전부 같은 사람 ID, cam=c1 소속)
gen_f,gen_p,gen_c=[],[],[]
for pid in gallery_ids:
    for k in range(K_POSES):
        rp=f"{GEN_DIR}/raw/pose{k}/{pid}_gen.png"
        if os.path.exists(rp):
            gen_f.append(feat(rp)); gen_p.append(int(pid)); gen_c.append(cam_idx(GALLERY_CAM))
A=lambda x:np.array(x,dtype=np.float32)
q_f,g_f,gen_f=map(A,[q_f,g_f,gen_f])
q_p,q_c,g_p,g_c,gen_p,gen_c=map(np.array,[q_p,q_c,g_p,g_c,gen_p,gen_c])
print(f"  query(C6)={len(q_f)} gallery(C1)={len(g_f)} gen={len(gen_f)} (={len(gallery_ids)}×{K_POSES})\n")


# ===== 6. Rank-k + mAP (Min-Distance 융합 방식으로 변경!) =====
def cosd(qf,gf): return (1.0-qf@gf.T).astype(np.float32)

def eval_ranks(dm,qp,gp,qc,gc,topk=(1,5,10),mr=50):
    nq=dm.shape[0]; idx=np.argsort(dm,axis=1); mt=(gp[idx]==qp[:,None]).astype(np.int32)
    cmcs,APs,v=[],[],0
    for qi in range(nq):
        o=idx[qi]; keep=~((gp[o]==qp[qi])&(gc[o]==qc[qi])); raw=mt[qi][keep]
        if not raw.any(): continue
        c=raw.cumsum(); c[c>1]=1; c=c[:mr]
        if len(c)<mr: c=np.concatenate([c,np.full(mr-len(c),c[-1])])
        cmcs.append(c); v+=1; nr=raw.sum()
        tmp=raw.cumsum()/(np.arange(len(raw))+1.0); APs.append((tmp*raw).sum()/nr)
    cmc=np.asarray(cmcs).sum(0)/v
    return {f"R{k}":float(cmc[k-1]*100) for k in topk}, float(np.mean(APs))*100

# 🌟 수정된 융합 함수 (갤러리 크기는 절대 늘리지 않고 힌트만 쏙 빼먹음)
def run_min_distance(extra_f, extra_p, extra_c):
    # 1. 쿼리 vs 원본 갤러리(C1) 거리 계산
    dist_base = cosd(q_f, g_f)
    
    if extra_f is None or len(extra_f) == 0:
        return eval_ranks(dist_base, q_p, g_p, q_c, g_c)
    
    # 2. 쿼리 vs 생성본 전체 거리 계산
    dist_gen = cosd(q_f, extra_f)
    
    # 최종 거리를 담을 배열 (시작은 원본 거리로)
    dist_final = dist_base.copy()
    
    # 3. 갤러리 사람 1명씩 돌면서 힌트(생성본) 적용
    for j in range(len(g_f)):
        pid = g_p[j]
        # extra_p(생성본) 중에 이 사람(pid)의 사진이 있는지 인덱스 찾기
        gen_idx = np.where(extra_p == pid)[0]
        
        if len(gen_idx) > 0:
            # 다중자세(K=3)일 경우, 3장 중 쿼리와 가장 비슷한(거리가 짧은) 1장만 고름
            min_gen_dist = np.min(dist_gen[:, gen_idx], axis=1)
            
            # 원본 거리와 생성본 최고 기록 중 더 가까운 값을 최종 점수로 채택!
            dist_final[:, j] = np.minimum(dist_final[:, j], min_gen_dist)
            
    return eval_ranks(dist_final, q_p, g_p, q_c, g_c)

logline("\n"+"="*72)
logline(f"## [2026-06-03] script114 Min-Distance 융합 평가")
logline(f"   다중자세(K=3) 중 최적 힌트만 반영 / 갤러리 모수(2114) 완벽 고정")
logline("="*72)
logline(f"\n{'Gallery 구성':<20}{'Rank-1':<9}{'Rank-5':<9}{'Rank-10':<10}{'mAP':<8}")
logline("-"*60)

# Baseline 평가
base_r, base_m = run_min_distance(None, None, None)
logline(f"{'Baseline (C1)':<20}{base_r['R1']:<9.1f}{base_r['R5']:<9.1f}{base_r['R10']:<10.1f}{base_m:<8.1f}")

# 생성본 융합 평가
for name, (ef, ep, ec) in [("+ gen_matched", (gen_f, gen_p, gen_c))]:
    r, m = run_min_distance(ef, ep, ec)
    d1 = r['R1'] - base_r['R1']
    dm = m - base_m
    logline(f"{name:<20}{r['R1']:<9.1f}{r['R5']:<9.1f}{r['R10']:<10.1f}{m:<8.1f} (R1{d1:+.1f}, mAP{dm:+.1f})")
logline("-"*60)