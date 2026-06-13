#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
115_fusion_cp.py ─ Feature 융합 + 카메라보정(CP) 결합 (114 생성물 재사용, 평가만)

[배경] 갤러리 "추가"는 5번 음성(94/97/110/112/114). 대신:
  - feature 융합(98): ID당 prototype = α·c1real + (1-α)·생성평균  → 갤러리 크기=ID수, mAP폭락 없음
  - CP(114): 카메라 분포 보정, +8.6 mAP 확인
  → 둘을 결합해서 시너지 보는 게 목표

[데이터] 114가 만든 생성물 재사용: outputs/c1_multipose_c6/raw/pose0~2/
[설정]  Duke→Market 크로스도메인, Query=C6, Gallery=C1(융합), 새 생성 0
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_DIR=f"{PROJECT_DIR}/outputs/c1_multipose_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
LOG=f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
device="cuda"
GALLERY_CAM="c1"; QUERY_CAM="c6"; K_POSES=3
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
ALPHAS=[0.7,0.8,0.9]

def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)
def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]
def cam_idx(tc): return int(tc[1:])-1

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
query_ids=[p for p in sorted(qby) if QUERY_CAM in qby[p] and GALLERY_CAM in gby[p]]
gallery_ids=[p for p in sorted(gby) if GALLERY_CAM in gby[p]]

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

print("feature 추출...")
q_f,q_p,q_c=[],[],[]
for pid in query_ids:
    for p in qby[pid][QUERY_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cam_idx(QUERY_CAM))
# c1 real: ID별로 모음 (융합용)
real_by_id=defaultdict(list)
for pid in gallery_ids:
    for p in gby[pid][GALLERY_CAM]: real_by_id[pid].append(feat(p))
# 생성본: ID별 평균
gen_by_id=defaultdict(list)
for pid in gallery_ids:
    for k in range(K_POSES):
        rp=f"{GEN_DIR}/raw/pose{k}/{pid}_gen.png"
        if os.path.exists(rp): gen_by_id[pid].append(feat(rp))
q_f=np.array(q_f,dtype=np.float32); q_p=np.array(q_p); q_c=np.array(q_c)
print(f"  query(C6)={len(q_f)} gallery ID={len(real_by_id)} gen 보유 ID={len(gen_by_id)}\n")

# ===== 메트릭/CP/융합 =====
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

# 갤러리 prototype 구성 (융합 여부, alpha)
def make_gallery(alpha, use_gen):
    gf,gp,gc=[],[],[]
    for pid in gallery_ids:
        rmean=np.mean(real_by_id[pid],axis=0)
        if use_gen and len(gen_by_id[pid])>0:
            proto=alpha*rmean+(1-alpha)*np.mean(gen_by_id[pid],axis=0)
        else:
            proto=rmean
        gf.append(proto); gp.append(int(pid)); gc.append(cam_idx(GALLERY_CAM))
    return l2n(np.array(gf,dtype=np.float32)), np.array(gp), np.array(gc)

# CP proto: real query + real gallery prototype
g_real_f,g_real_p,g_real_c=make_gallery(1.0,False)
protos=build_protos(np.concatenate([q_f,g_real_f]),np.concatenate([q_c,g_real_c]))

def run(alpha,use_gen,use_cp):
    gf,gp,gc=make_gallery(alpha,use_gen)
    if use_cp: qf=apply_cp(q_f,q_c,protos); gf=apply_cp(gf,gc,protos)
    else: qf=q_f
    return eval_ranks(cosd(qf,gf),q_p,gp,q_c,gc)

logline("\n"+"="*72)
logline(f"## [2026-06-03] script115 Feature융합 + CP 결합 (Duke→Market, C1→C6)")
logline(f"   Query=C6({len(q_f)}), Gallery=C1 ID prototype({len(gallery_ids)}), 융합=txt2img생성 K={K_POSES}")
logline("="*72)
logline(f"\n{'구성':<26}{'R1':<8}{'R5':<8}{'R10':<8}{'mAP':<8}")
logline("-"*58)
# 1) Baseline (real prototype, no CP)
r,m=run(1.0,False,False); base_m=m
logline(f"{'Baseline (real)':<26}{r['R1']:<8.1f}{r['R5']:<8.1f}{r['R10']:<8.1f}{m:<8.1f}")
# 2) +CP only
r,m=run(1.0,False,True); cp_m=m
logline(f"{'+ CP':<26}{r['R1']:<8.1f}{r['R5']:<8.1f}{r['R10']:<8.1f}{m:<8.1f} (mAP{m-base_m:+.1f})")
logline("-"*58)
# 3) 융합 단독 + 융합+CP (alpha 별)
for a in ALPHAS:
    r,m=run(a,True,False)
    logline(f"{'+ fusion α='+str(a):<26}{r['R1']:<8.1f}{r['R5']:<8.1f}{r['R10']:<8.1f}{m:<8.1f} (mAP{m-base_m:+.1f})")
    r2,m2=run(a,True,True)
    logline(f"{'+ fusion α='+str(a)+' + CP':<26}{r2['R1']:<8.1f}{r2['R5']:<8.1f}{r2['R10']:<8.1f}{m2:<8.1f} (vs CP{m2-cp_m:+.1f})")
logline("-"*58)
logline("해석: '+CP' 대비 '+fusion+CP' 가 더 높으면 → 융합이 CP 위에 추가 기여 (시너지)")