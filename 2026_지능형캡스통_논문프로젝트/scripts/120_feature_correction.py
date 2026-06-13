#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
120_feature_correction.py ─ 생성본 feature 보정으로 융합 살리기 (멀티카메라)

가설: 현재는 생성본 raw feature가 융합되어 SD 도메인 편향이 prototype을 흔듦.
     생성본 feature를 보정하면 융합이 살아날 수 있음.

비교:
  Baseline                         : real만
  현재 Proposed (gen raw 융합 + CP)
  방법 B (gen 도메인 보정 후 융합 + CP) : gen − μ_gen
  방법 C (real/gen 각자 보정 후 융합+CP): 각자 평균 차감
  + α 민감도 (0.7/0.8/0.9)

멀티카메라: Gallery=C1~C5, Query=C6. 118 생성물 재활용. Duke→Market.
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

# query
print("feature 추출...")
q_ids=[p for p in sorted(qby) if TARGET_CAM in qby[p] and any(sc in gby[p] for sc in SOURCE_CAMS)]
q_f=[]; q_p=[]; q_c=[]
for pid in q_ids:
    for p in qby[pid][TARGET_CAM]:
        q_f.append(feat(p)); q_p.append(int(pid)); q_c.append(cidx(TARGET_CAM))
q_f=np.array(q_f,dtype=np.float32); q_p=np.array(q_p); q_c=np.array(q_c)

# 갤러리 entry별 real / gen (raw, 정규화 전)
g_p=[]; g_c=[]; g_real=[]; g_gen=[]; has_gen=[]
for sc in SOURCE_CAMS:
    for pid in [p for p in sorted(gby) if sc in gby[p]]:
        rf=np.mean([feat(x) for x in gby[pid][sc]],axis=0)
        gens=[feat(f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png") for k in range(K_POSES)
              if os.path.exists(f"{GEN_ROOT}/{sc}/pose{k}/{pid}_gen.png")]
        g_p.append(int(pid)); g_c.append(cidx(sc)); g_real.append(rf)
        if gens: g_gen.append(np.mean(gens,axis=0)); has_gen.append(True)
        else: g_gen.append(rf); has_gen.append(False)
g_p=np.array(g_p); g_c=np.array(g_c)
g_real=np.array(g_real,dtype=np.float32); g_gen=np.array(g_gen,dtype=np.float32)
print(f"  query={len(q_f)}, gallery={len(g_p)}, 생성보유={sum(has_gen)}\n")

# 도메인 평균 (보정용)
gen_mean = g_gen[has_gen].mean(0)      # 생성 도메인 편향
real_mean = g_real.mean(0)             # real 도메인 편향

# CP proto (real 기반)
g_base=l2n(g_real)
protos=build_protos(np.concatenate([q_f,g_base]),np.concatenate([q_c,g_c]))

def run(gallery_f, use_cp):
    gf=l2n(gallery_f)
    if use_cp:
        qf=apply_cp(q_f,q_c,protos); gf=apply_cp(gf,g_c,protos)
    else:
        qf=q_f
    return eval_ranks(cosd(qf,gf),q_p,g_p,q_c,g_c)

def gallery_A(a): return a*g_real+(1-a)*g_gen                      # 현재(raw)
def gallery_B(a): return a*g_real+(1-a)*(g_gen-gen_mean)           # gen 도메인 보정
def gallery_C(a): return a*(g_real-real_mean)+(1-a)*(g_gen-gen_mean) # 각자 보정

logline("\n"+"="*72)
logline(f"## [2026-06-03] script120 생성본 feature 보정 융합 (멀티카메라)")
logline(f"   Gallery=C1~C5, Query=C6, query={len(q_f)}, gallery={len(g_p)}")
logline("="*72)
b_r,b_m=run(g_real,False); cp_r,cp_m=run(g_real,True)
logline(f"\n{'구성':<34}{'R1':<9}{'mAP':<9}{'ΔmAP':<8}")
logline("-"*60)
logline(f"{'Baseline (real, no CP)':<34}{b_r:<9.1f}{b_m:<9.1f}{'—':<8}")
logline(f"{'real + CP':<34}{cp_r:<9.1f}{cp_m:<9.1f}{cp_m-b_m:+.1f}")
logline("-"*60)
for a in [0.7,0.8,0.9]:
    r,m=run(gallery_A(a),True)
    logline(f"{'A) gen raw 융합 α='+str(a)+' +CP':<34}{r:<9.1f}{m:<9.1f}{m-cp_m:+.1f} (vs CP)")
logline("-"*60)
for a in [0.7,0.8,0.9]:
    r,m=run(gallery_B(a),True)
    logline(f"{'B) gen 보정 융합 α='+str(a)+' +CP':<34}{r:<9.1f}{m:<9.1f}{m-cp_m:+.1f} (vs CP)")
logline("-"*60)
for a in [0.7,0.8,0.9]:
    r,m=run(gallery_C(a),True)
    logline(f"{'C) 각자보정 융합 α='+str(a)+' +CP':<34}{r:<9.1f}{m:<9.1f}{m-cp_m:+.1f} (vs CP)")
logline("-"*60)
logline("해석: B/C 가 'real+CP'(cp_m) 보다 높으면 → feature 보정이 융합을 살림. 생성형 기여 입증")