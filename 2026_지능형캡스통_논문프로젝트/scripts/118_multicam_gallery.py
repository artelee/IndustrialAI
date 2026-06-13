#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
118_multicam_gallery.py ─ 진짜 멀티카메라 평가 (논문 메인 표)

표준 멀티카메라 프로토콜:
  Gallery = C1~C5 전부 한 갤러리에 (각 인물은 등장한 카메라마다 prototype)
  Query   = C6
  → C6 인물을, 5개 카메라가 섞인 갤러리에서 검색

비교군:
  Baseline : 원본 feature만
  Proposed : 생성본 융합(α=0.8) + 카메라 단위 보정(CP)

117 생성물(outputs/allcam_to_c6/{c1~c5}/pose0~2) 재활용, 평가만.
Duke→Market 크로스도메인.
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
TARGET_CAM="c6"; SOURCE_CAMS=["c1","c2","c3","c4","c5"]
K_POSES=3; ALPHA=0.8
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"

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

# CLIP-ReID
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

# 메트릭/CP/융합
def l2n(f):
    n=np.linalg.norm(f,axis=1,keepdims=True); n[n==0]=1e-12; return (f/n).astype(np.float32)
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
def build_protos(f,c): return {int(cc):f[c==cc].mean(0) for cc in np.unique(c)}
def apply_cp(f,c,pr):
    out=f.copy(); gm=np.mean(list(pr.values()),axis=0)
    for i,cc in enumerate(c): out[i]=out[i]-pr.get(int(cc),gm)
    return l2n(out)

# ── 쿼리: C6 (정답이 소스 카메라 중 하나라도 있어야) ──
print("query(C6) feature...")
q_ids=[p for p in sorted(qby) if TARGET_CAM in qby[p] and any(sc in gby[p] for sc in SOURCE_CAMS)]
q_f=[]; q_p=[]; q_c=[]
for pid in q_ids:
    for p in qby[pid][TARGET_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cidx(TARGET_CAM))
q_f=np.array(q_f,dtype=np.float32); q_p=np.array(q_p); q_c=np.array(q_c)
print(f"  query(C6)={len(q_f)}\n")

# ── 갤러리: C1~C5 전부 한 갤러리에 ──
# 각 (인물, 카메라) 조합마다 하나의 entry. real prototype + 생성본 융합.
print("멀티카메라 갤러리 구성 (C1~C5)...")
real_by_idcam=defaultdict(list)   # (pid,cam) -> real feats
gen_by_idcam=defaultdict(list)    # (pid,cam) -> gen feats
for sc in SOURCE_CAMS:
    gallery_ids=[p for p in sorted(gby) if sc in gby[p]]
    for pid in gallery_ids:
        for p in gby[pid][sc]:
            real_by_idcam[(pid,sc)].append(feat(p))
        for k in range(K_POSES):
            rp=f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png"
            if os.path.exists(rp): gen_by_idcam[(pid,sc)].append(feat(rp))
    print(f"  {sc}: {len(gallery_ids)}명")

def make_gallery(use_gen):
    gf,gp,gc=[],[],[]
    for (pid,sc),reals in real_by_idcam.items():
        rm=np.mean(reals,axis=0)
        gens=gen_by_idcam.get((pid,sc),[])
        if use_gen and len(gens)>0:
            proto=ALPHA*rm+(1-ALPHA)*np.mean(gens,axis=0)
        else:
            proto=rm
        gf.append(proto); gp.append(int(pid)); gc.append(cidx(sc))
    return l2n(np.array(gf,dtype=np.float32)), np.array(gp), np.array(gc)

g_real_f,g_p,g_c=make_gallery(False)
print(f"  갤러리 총 entry={len(g_real_f)} (카메라 분포={np.bincount(g_c)})\n")

# CP prototype: real query + real gallery
protos=build_protos(np.concatenate([q_f,g_real_f]),np.concatenate([q_c,g_c]))

def run(use_gen,use_cp):
    gf,gp,gc=make_gallery(use_gen)
    if use_cp: qf=apply_cp(q_f,q_c,protos); gf=apply_cp(gf,gc,protos)
    else: qf=q_f
    return eval_ranks(cosd(qf,gf),q_p,gp,q_c,gc)

# ── 평가 ──
logline("\n"+"="*72)
logline(f"## [2026-06-03] script118 멀티카메라 평가 (Gallery=C1~C5, Query=C6, Duke→Market)")
logline(f"   query={len(q_f)}, gallery entry={len(g_real_f)}, α={ALPHA}")
logline("="*72)
logline(f"\n{'구성':<22}{'R1':<9}{'R5':<9}{'R10':<9}{'mAP':<9}{'ΔmAP':<8}")
logline("-"*66)
b_r,b_m=run(False,False)
logline(f"{'Baseline':<22}{b_r['R1']:<9.1f}{b_r['R5']:<9.1f}{b_r['R10']:<9.1f}{b_m:<9.1f}{'—':<8}")
p_r,p_m=run(True,True)
logline(f"{'Proposed (Fus+CP)':<22}{p_r['R1']:<9.1f}{p_r['R5']:<9.1f}{p_r['R10']:<9.1f}{p_m:<9.1f}{p_m-b_m:+.1f}")
logline("-"*66)
# 참고: CP 단독, Fusion 단독도
cp_r,cp_m=run(False,True); fu_r,fu_m=run(True,False)
logline(f"(참고) +CP only      R1={cp_r['R1']:.1f} mAP={cp_m:.1f} (Δ{cp_m-b_m:+.1f})")
logline(f"(참고) +Fusion only  R1={fu_r['R1']:.1f} mAP={fu_m:.1f} (Δ{fu_m-b_m:+.1f})")
logline("-"*66)
logline("해석: C1~C5 전부 한 갤러리(진짜 멀티카메라). Proposed가 Baseline 대비 양성이면 멀티카메라 일반성 입증")