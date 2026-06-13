#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
96_gallery_camera_sweep.py  ―  갤러리 카메라 6-way 전수 비교 (Duke→Market)

[이 실험이 뭐냐]
  지금까지 sparse 실험은 c1 을 갤러리로 고정했다. 75번에서 c1→c2 만 유독
  음수였던 점, c1 의 화각이 다른 카메라와 많이 다른 점을 고려하면
  'c1 불리' 효과가 결과를 깎아왔을 가능성이 있다.

  → 6개 카메라 각각을 갤러리로 두고 baseline + 보정(CP) 측정.

[세팅]
  gallery   = bounding_box_test 에서 gallery_cam 만
  query     = query 폴더에서 gallery_cam 이 아닌 모든 카메라
  weight    = Duke 학습 → Market 평가 (cross-domain)

[알아내는 것]
  ① 어느 카메라가 갤러리로 가장 유리한가  → 이후 생성 실험의 새 기준점
  ② 보정(CP)이 6 카메라 모두에서 baseline 을 이기는가 → 주 기여 robustness
  ③ c1 의 상대적 불리함이 실제 존재했는가

* 93번 표준평가 캐시(feat_cache_std/duke2market_*.npy) 그대로 재사용 → 추출 0.
"""

import os, re, glob, datetime
import numpy as np

# ===== CONFIG =====
HOME        = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT        = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR  = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
CLIP_REID   = "/home/ubuntu/CLIP-ReID"
LOG         = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
WEIGHT      = f"{CKPT}/clipreid_duke/ViT-B-16_60.pth"
NUM_CLASS   = 702
DEVICE      = "cuda"
BATCH       = 128

# 93번 캐시 재사용
CACHE_Q = f"{PROJECT_DIR}/feat_cache_std/duke2market_query.npy"
CACHE_G = f"{PROJECT_DIR}/feat_cache_std/duke2market_gallery.npy"

CAMERAS = [0, 1, 2, 3, 4, 5]   # c1~c6 (0-index)

def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)


# ===== 모델/추출/CP/메트릭 (93·94·95 동일) =====
def build_model(weight_path, num_class):
    import sys, torch, torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, CLIP_REID)
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME="ViT-B-16"; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE="softmax"
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5]*3; cfg.INPUT.PIXEL_STD=[0.5]*3
    cfg.DATASETS.NAMES="market1501"; cfg.TEST.WEIGHT=weight_path; cfg.TEST.NECK_FEAT="before"
    try:    b=make_model(cfg,num_class=num_class,camera_num=0,view_num=1)
    except Exception: b=make_model(cfg,num_class=num_class,camera_num=6,view_num=1)
    b.load_param(weight_path); b=b.to(DEVICE).eval()
    class BB(nn.Module):
        def __init__(s,bb): super().__init__(); s.bb=bb
        @torch.no_grad()
        def forward(s,x):
            f=s.bb(x,cam_label=None)
            if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
            if f.dim()>2: f=f.view(f.size(0),-1)
            return f
    pre=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
    return BB(b).eval(), pre

def extract_features(model,preprocess,paths,cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"[cache] {os.path.basename(cache_path)}"); return np.load(cache_path)
    import torch
    from PIL import Image
    feats=[]
    with torch.no_grad():
        for i in range(0,len(paths),BATCH):
            bp=paths[i:i+BATCH]
            imgs=torch.stack([preprocess(Image.open(p).convert("RGB")) for p in bp]).to(DEVICE)
            f=torch.nn.functional.normalize(model(imgs).float(),dim=1)
            feats.append(f.cpu().numpy())
            print(f"\r[extract] {min(i+BATCH,len(paths))}/{len(paths)}",end="")
    print(); feats=np.concatenate(feats,0).astype(np.float32)
    if cache_path: np.save(cache_path,feats)
    return feats

_PAT=re.compile(r"([-\d]+)_c(\d+)")
def parse_dir(d):
    paths,pids,cams=[],[],[]
    for p in sorted(glob.glob(os.path.join(d,"*.jpg"))):
        m=_PAT.search(os.path.basename(p))
        if m is None: continue
        pid,cam=int(m.group(1)),int(m.group(2))
        if pid==-1: continue
        paths.append(p); pids.append(pid); cams.append(cam-1)
    return paths,np.asarray(pids),np.asarray(cams)

def cosine_distmat(qf,gf): return (1.0-qf@gf.T).astype(np.float32)

def build_prototypes(f,c):
    protos={}
    for cc in np.unique(c):
        sel=np.where(c==cc)[0]; protos[int(cc)]=f[sel].mean(0)
    return protos

def apply_cp(f,c,protos):
    out=f.copy(); gmean=np.mean(list(protos.values()),axis=0)
    for i,cc in enumerate(c): out[i]=out[i]-protos.get(int(cc),gmean)
    n=np.linalg.norm(out,axis=1,keepdims=True); n[n==0]=1e-12
    return (out/n).astype(np.float32)

def eval_market(distmat,q_pids,g_pids,q_cams,g_cams,max_rank=50):
    num_q=distmat.shape[0]; indices=np.argsort(distmat,axis=1)
    matches=(g_pids[indices]==q_pids[:,None]).astype(np.int32)
    all_cmc,all_AP,valid=[],[],0
    for qi in range(num_q):
        order=indices[qi]
        keep=~((g_pids[order]==q_pids[qi])&(g_cams[order]==q_cams[qi]))
        raw=matches[qi][keep]
        if not raw.any(): continue
        cmc=raw.cumsum(); cmc[cmc>1]=1
        c=cmc[:max_rank]
        if len(c)<max_rank: c=np.concatenate([c,np.full(max_rank-len(c),c[-1])])
        all_cmc.append(c); valid+=1
        nr=raw.sum(); tmp=raw.cumsum()/(np.arange(len(raw))+1.0)
        all_AP.append((tmp*raw).sum()/nr)
    return np.asarray(all_cmc).sum(0)/valid, float(np.mean(all_AP)), valid


# ===== 한 카메라 sweep =====
def run_one(gallery_cam, g_f, g_p, g_c, q_f, q_p, q_c):
    gmask=(g_c==gallery_cam)
    gf,gp,gc=g_f[gmask],g_p[gmask],g_c[gmask]
    qmask=(q_c!=gallery_cam)
    qf,qp,qc=q_f[qmask],q_p[qmask],q_c[qmask]
    # baseline
    cmc_b,mAP_b,vq=eval_market(cosine_distmat(qf,gf),qp,gp,qc,gc)
    # CP: 갤러리(1cam) + query 실제 카메라 통계로 percam proto
    src_f=np.concatenate([gf,qf]); src_c=np.concatenate([gc,qc])
    protos=build_prototypes(src_f,src_c)
    qf_c=apply_cp(qf,qc,protos); gf_c=apply_cp(gf,gc,protos)
    cmc_cp,mAP_cp,_=eval_market(cosine_distmat(qf_c,gf_c),qp,gp,qc,gc)
    return dict(gids=len(np.unique(gp)), gn=len(gf), qn=len(qf), vq=vq,
                b_r1=cmc_b[0]*100, b_map=mAP_b*100,
                cp_r1=cmc_cp[0]*100, cp_map=mAP_cp*100)


def main():
    logline("\n"+"="*92)
    logline(f"## [{datetime.date.today()}] script96 갤러리 카메라 6-way sweep (Duke→Market)")
    logline("="*92)
    qp_paths,qp_pids,qp_cams=parse_dir(f"{MARKET_DIR}/query")
    gp_paths,gp_pids,gp_cams=parse_dir(f"{MARKET_DIR}/bounding_box_test")
    print(f"[data] query={len(qp_paths)}, gallery={len(gp_paths)}")

    model,pre=build_model(WEIGHT,NUM_CLASS)
    q_f=extract_features(model,pre,qp_paths,CACHE_Q)
    g_f=extract_features(model,pre,gp_paths,CACHE_G)

    logline(f"\n{'gallery':<9}{'gIDs':<6}{'gN':<7}{'qN':<7}{'B0_R1':<8}{'B0_mAP':<8}"
            f"{'CP_R1':<8}{'CP_mAP':<8}{'ΔR1':<8}{'ΔmAP':<8}")
    logline("-"*92)
    rows=[]
    for cam in CAMERAS:
        r=run_one(cam,g_f,gp_pids,gp_cams,q_f,qp_pids,qp_cams)
        d_r1=r['cp_r1']-r['b_r1']; d_map=r['cp_map']-r['b_map']
        logline(f"c{cam+1:<8}{r['gids']:<6}{r['gn']:<7}{r['qn']:<7}"
                f"{r['b_r1']:<8.1f}{r['b_map']:<8.1f}{r['cp_r1']:<8.1f}{r['cp_map']:<8.1f}"
                f"{d_r1:<+8.1f}{d_map:<+8.1f}")
        rows.append((cam,r,d_r1,d_map))

    avg=lambda k: np.mean([rr[k] for _,rr,_,_ in rows])
    logline("-"*92)
    logline(f"{'평균':<9}{'-':<6}{'-':<7}{'-':<7}"
            f"{avg('b_r1'):<8.1f}{avg('b_map'):<8.1f}{avg('cp_r1'):<8.1f}{avg('cp_map'):<8.1f}"
            f"{avg('cp_r1')-avg('b_r1'):<+8.1f}{avg('cp_map')-avg('b_map'):<+8.1f}")

    best_b=max(rows,key=lambda x:x[1]['b_map'])
    best_cp=max(rows,key=lambda x:x[1]['cp_map'])
    worst_b=min(rows,key=lambda x:x[1]['b_map'])
    n_win=sum(1 for _,r,_,_ in rows if r['cp_map']>r['b_map'])
    logline(f"\n→ baseline mAP 최고 갤러리: c{best_b[0]+1} ({best_b[1]['b_map']:.1f}) / "
            f"최저: c{worst_b[0]+1} ({worst_b[1]['b_map']:.1f})")
    logline(f"→ CP       mAP 최고 갤러리: c{best_cp[0]+1} ({best_cp[1]['cp_map']:.1f})")
    logline(f"→ CP > baseline mAP : {n_win}/6 카메라에서 우위 (주 기여 robustness)")
    logline(f"\n해석: c1이 최저면 → c1 불리 가설 확정, 이후 생성 실험은 best_b 카메라 기준으로 재실행")

if __name__=="__main__":
    main()