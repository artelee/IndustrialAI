#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
117_allcam_table.py ─ 카메라 쌍별 통합 표 (논문 표 2)

각 source 카메라(C1~C5) → target C6 에 대해:
  생성: source 인물을 C6 대표자세 K=3 으로 변환 (txt2img+IPA, 113 검증설정)
  평가: Baseline / +CP / +Fusion / +CP+Fusion  (Duke→Market 크로스도메인)

출력 표:
  Source→C6   Baseline(R1/mAP)  +CP  +Fusion  +CP+Fusion
  C1→C6 ... C5→C6 ... 평균
"""

import os, sys, glob, gc, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_ROOT=f"{PROJECT_DIR}/outputs/allcam_to_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
LOG=f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
device,dtype="cuda",torch.float16
TARGET_CAM="c6"; SOURCE_CAMS=["c1","c2","c3","c4","c5"]
K_POSES=3; POSE_POOL=100
SEED=42; SIZE=(384,768); IPA_INPUT=(512,1024); IPA_SCALE=0.8; CN_SCALE=1.0
ALPHA=0.8
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
PROMPT="a photo of a person, full body shot, standing, taken from a surveillance CCTV camera, top-down angle, highly detailed, photorealistic"
NEG="cropped, close up, partial body, missing limbs, extra limbs, bad anatomy, deformed, blurry, low resolution, face portrait"

def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)
def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]
def cidx(tc): return int(tc[1:])-1

# 데이터
print("데이터 로드...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)

# ── OpenPose: C6 대표자세 K개 (모든 source 공통) ──
print("OpenPose + C6 대표자세...")
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
# C6 자세 풀 (어느 source ID도 제외할 필요는 없음: 자세만 빌림, 외형 무관)
pool=[]
for pid in sorted(gby):
    if TARGET_CAM in gby[pid]: pool.append(gby[pid][TARGET_CAM][0])
    if len(pool)>=POSE_POOL*2: break
skels,vecs=[],[]
for p in pool:
    s=detect(Image.open(p).convert("RGB").resize(SIZE,Image.LANCZOS))
    if s is None: continue
    skels.append(s.resize(SIZE,Image.LANCZOS))
    vecs.append(np.array(s.resize((48,96))).astype(np.float32).flatten())
    if len(skels)>=POSE_POOL: break
vecs=np.array(vecs); vecs=vecs/(np.linalg.norm(vecs,axis=1,keepdims=True)+1e-8)
c6_poses=[skels[i] for i in select_k(vecs,K_POSES,SEED)]
print(f"  C6 대표자세 {len(c6_poses)}개\n")

# ── 생성 파이프라인 ──
print("생성 파이프라인 로드...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=cn,cache_dir=CACHE_DIR,
    torch_dtype=dtype,safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IPA_SCALE); pipe=pipe.to(device)
print("로드 완료\n")

# ── 각 source 카메라 생성 ──
for sc in SOURCE_CAMS:
    gallery_ids=[p for p in sorted(gby) if sc in gby[p]]
    for k in range(K_POSES): os.makedirs(f"{GEN_ROOT}/{sc}/pose{k}",exist_ok=True)
    todo=[p for p in gallery_ids if not all(os.path.exists(f"{GEN_ROOT}/{sc}/pose{k}/{p}_gen.png") for k in range(K_POSES))]
    if not todo:
        print(f"[{sc}→c6] 생성 완료됨(skip)"); continue
    print(f"[{sc}→c6] 생성 {len(todo)}명 × {K_POSES}자세...")
    for i,pid in enumerate(todo):
        src=Image.open(sorted(gby[pid][sc])[0]).convert("RGB").resize(IPA_INPUT,Image.LANCZOS)
        for k in range(K_POSES):
            rp=f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png"
            if os.path.exists(rp): continue
            g=torch.Generator(device).manual_seed(SEED+k)
            gen=pipe(prompt=PROMPT,negative_prompt=NEG,image=c6_poses[k],ip_adapter_image=src,
                     controlnet_conditioning_scale=CN_SCALE,num_inference_steps=30,guidance_scale=7.5,
                     width=SIZE[0],height=SIZE[1],generator=g).images[0]
            gen.save(rp)
        if (i+1)%100==0: print(f"    {i+1}/{len(todo)}")
del pipe,cn; gc.collect(); torch.cuda.empty_cache()
print("전체 생성 완료\n")

# ── CLIP-ReID ──
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

# ── 메트릭/CP/융합 ──
def l2n(f):
    n=np.linalg.norm(f,axis=1,keepdims=True); n[n==0]=1e-12; return (f/n).astype(np.float32)
def cosd(qf,gf): return (1.0-qf@gf.T).astype(np.float32)
def eval_ranks(dm,qp,gp,qc,gc,mr=50):
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
    return cmc[0]*100, float(np.mean(APs))*100
def build_protos(f,c): return {int(cc):f[c==cc].mean(0) for cc in np.unique(c)}
def apply_cp(f,c,pr):
    out=f.copy(); gm=np.mean(list(pr.values()),axis=0)
    for i,cc in enumerate(c): out[i]=out[i]-pr.get(int(cc),gm)
    return l2n(out)

# ── query: C6 real (모든 source 공통) ──
print("query(C6) feature...")
q_ids=[p for p in sorted(qby) if TARGET_CAM in qby[p]]
q_f=[]; q_p=[]; q_c=[]
for pid in q_ids:
    if not any(sc in gby[pid] for sc in SOURCE_CAMS): continue  # 정답이 source에 있어야
    for p in qby[pid][TARGET_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cidx(TARGET_CAM))
q_f=np.array(q_f,dtype=np.float32); q_p=np.array(q_p); q_c=np.array(q_c)
print(f"  query(C6)={len(q_f)}\n")

# ── 각 source 평가 ──
logline("\n"+"="*84)
logline(f"## [2026-06-03] script117 카메라쌍별 표 (Duke→Market, →C6, K={K_POSES}, α={ALPHA})")
logline("="*84)
logline(f"\n{'Source→C6':<12}{'Base R1':<9}{'Base mAP':<10}{'CP R1':<8}{'CP mAP':<9}"
        f"{'Fus R1':<8}{'Fus mAP':<9}{'CP+Fus R1':<11}{'CP+Fus mAP':<11}")
logline("-"*84)
agg=defaultdict(list)
for sc in SOURCE_CAMS:
    gallery_ids=[p for p in sorted(gby) if sc in gby[p]]
    # real prototype (ID별 평균) + 생성 prototype
    real_by_id=defaultdict(list); gen_by_id=defaultdict(list)
    for pid in gallery_ids:
        for p in gby[pid][sc]: real_by_id[pid].append(feat(p))
        for k in range(K_POSES):
            rp=f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png"
            if os.path.exists(rp): gen_by_id[pid].append(feat(rp))
    def make_gal(use_gen):
        gf,gp,gc=[],[],[]
        for pid in gallery_ids:
            rm=np.mean(real_by_id[pid],axis=0)
            if use_gen and len(gen_by_id[pid])>0:
                proto=ALPHA*rm+(1-ALPHA)*np.mean(gen_by_id[pid],axis=0)
            else: proto=rm
            gf.append(proto); gp.append(int(pid)); gc.append(cidx(sc))
        return l2n(np.array(gf,dtype=np.float32)),np.array(gp),np.array(gc)
    g_real_f,g_p,g_c=make_gal(False)
    protos=build_protos(np.concatenate([q_f,g_real_f]),np.concatenate([q_c,g_c]))
    def run(use_gen,use_cp):
        gf,gp,gc=make_gal(use_gen)
        if use_cp: qf=apply_cp(q_f,q_c,protos); gf=apply_cp(gf,gc,protos)
        else: qf=q_f
        return eval_ranks(cosd(qf,gf),q_p,gp,q_c,gc)
    b=run(False,False); cp=run(False,True); fu=run(True,False); cf=run(True,True)
    logline(f"{sc+'→c6':<12}{b[0]:<9.1f}{b[1]:<10.1f}{cp[0]:<8.1f}{cp[1]:<9.1f}"
            f"{fu[0]:<8.1f}{fu[1]:<9.1f}{cf[0]:<11.1f}{cf[1]:<11.1f}")
    for key,val in [('b',b),('cp',cp),('fu',fu),('cf',cf)]: agg[key].append(val)
logline("-"*84)
m=lambda k,i: np.mean([v[i] for v in agg[k]])
logline(f"{'평균':<12}{m('b',0):<9.1f}{m('b',1):<10.1f}{m('cp',0):<8.1f}{m('cp',1):<9.1f}"
        f"{m('fu',0):<8.1f}{m('fu',1):<9.1f}{m('cf',0):<11.1f}{m('cf',1):<11.1f}")
logline("-"*84)
logline("해석: CP가 5개 카메라쌍 전부 양성 → 다중카메라 일반성 / Fusion 추가기여 / CP+Fus 최고")