#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
122_domaingap_map.py ─ domain gap vector 보정의 진짜 mAP 효과 (멀티카메라)

121에서 cosine이 0.895→0.903 올랐으나, domain gap은 모든 생성본에 같은 벡터를
더하므로 cosine 일괄 상승(착시)일 수 있음. 진짜 판단은 mAP.

비교 (멀티카메라 Gallery=C1~C5, Query=C6):
  real + CP                            (기준, 67.6)
  gen raw 융합 + CP
  gen + domain_gap 보정 융합 + CP       ← 121에서 cosine 오른 그 방법
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_ROOT=f"{PROJECT_DIR}/outputs/allcam_to_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
LOG=f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
device="cuda"
TARGET_CAM="c6"; SOURCE_CAMS=["c1","c2","c3","c4","c5"]; K_POSES=3
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"

def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)
def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]
def cidx(tc): return int(tc[1:])-1

print("데이터 로드...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)

print("CLIP-ReID 로드...")
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

print("feature 추출...")
q_ids=[p for p in sorted(qby) if TARGET_CAM in qby[p] and any(sc in gby[p] for sc in SOURCE_CAMS)]
q_f=[]; q_p=[]; q_c=[]
for pid in q_ids:
    for p in qby[pid][TARGET_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cidx(TARGET_CAM))
q_f=np.array(q_f,dtype=np.float32); q_p=np.array(q_p); q_c=np.array(q_c)
# =================================================================
# 1. 피처 평균 후 L2 정규화 (발언권 회복)
# =================================================================
g_p=[]; g_c=[]; g_real=[]; g_gen=[]
for sc in SOURCE_CAMS:
    for pid in [p for p in sorted(gby) if sc in gby[p]]:
        # 원본 평균 후 정규화
        rf_mean = np.mean([feat(x) for x in gby[pid][sc]], axis=0)
        rf = l2n(np.expand_dims(rf_mean, 0))[0] 
        
        # 생성본 평균 후 정규화
        gens = [feat(f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png") for k in range(K_POSES)
                if os.path.exists(f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png")]
        if gens:
            gen_mean = np.mean(gens, axis=0)
            gf = l2n(np.expand_dims(gen_mean, 0))[0]
        else:
            gf = rf
            
        g_p.append(int(pid)); g_c.append(cidx(sc))
        g_real.append(rf); g_gen.append(gf)

g_p=np.array(g_p); g_c=np.array(g_c)
g_real=np.array(g_real, dtype=np.float32); g_gen=np.array(g_gen, dtype=np.float32)

# =================================================================
# 2. 카메라별 도메인 보정 (Camera-Specific Calibration)
# =================================================================
g_gen_calib = np.zeros_like(g_gen)

for c_idx in np.unique(g_c):
    mask = (g_c == c_idx)
    if not mask.any(): continue
    
    # 해당 카메라(예: C1)만의 원본/생성본 평균 도출
    cam_mean_real = g_real[mask].mean(0)
    cam_mean_gen = g_gen[mask].mean(0)
    
    # 해당 카메라 전용 도메인 갭 벡터
    cam_domain_gap = cam_mean_real - cam_mean_gen
    
    # 해당 카메라의 생성본 피처들에만 전용 갭 벡터를 더해줌
    g_gen_calib[mask] = g_gen[mask] + cam_domain_gap

# 보정 후 또 정규화! (매우 중요)
g_gen_calib = l2n(g_gen_calib)

# =================================================================
# 3. Adaptive Fusion (적응형 가중합) 적용 평가
# =================================================================
# 기존 준비물 (CP 등)
g_base = l2n(g_real)
protos = build_protos(np.concatenate([q_f, g_base]), np.concatenate([q_c, g_c]))

def run(gallery_f):
    gf = apply_cp(l2n(gallery_f), g_c, protos)
    qf = apply_cp(q_f, q_c, protos)
    return eval_ranks(cosd(qf, gf), q_p, g_p, q_c, g_c)

logline("\n"+"="*64)
logline(f"## [2026-06-03] 효과 극대화: 카메라별 보정 + 적응형 융합")
logline("="*64)

# 1. 기준 (Real + CP)
cp_r, cp_m = run(g_real)
logline(f"{'real + CP (기준)':<32}{cp_r:<9.1f}{cp_m:<9.1f}{'—':<8}")

# 2. 기존 고정 Alpha (비교용)
alpha = 0.8
r, m = run(alpha * g_real + (1 - alpha) * g_gen_calib)
logline(f"{'고정 보정 융합 (α=0.8)':<32}{r:<9.1f}{m:<9.1f}{m-cp_m:+.1f}")

# 3. 마법의 Adaptive Fusion
# 원본과 생성본이 얼마나 비슷한지(품질) 내적(Dot product)으로 측정
quality_scores = np.sum(g_real * g_gen_calib, axis=1, keepdims=True)

# 품질이 낮으면(비슷하지 않으면) alpha를 0.95까지 올려 원본을 지키고,
# 품질이 높으면(유사하면) alpha를 0.7까지 내려 힌트를 듬뿍 수용함
adaptive_alphas = 0.95 - (quality_scores * 0.25) 
adaptive_alphas = np.clip(adaptive_alphas, 0.6, 0.95)

# 각자 다른 alpha 값으로 융합
adaptive_fused_g = (adaptive_alphas * g_real) + ((1.0 - adaptive_alphas) * g_gen_calib)

r_adapt, m_adapt = run(adaptive_fused_g)
logline(f"{'🚀 적응형 보정 융합 (Adaptive)':<32}{r_adapt:<9.1f}{m_adapt:<9.1f}{m_adapt-cp_m:+.1f}")
logline("-"*64)