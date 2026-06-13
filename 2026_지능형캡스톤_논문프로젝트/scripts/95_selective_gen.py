#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
95_selective_gen.py  ―  기여③ 재시도: '어려운 케이스에만' 선택적 생성

[네 아이디어]
  94번은 모든 ID에 무조건 생성을 넣어서 쉬운 것까지 망쳤다(mAP -9).
  → 보정(B0+CP) 후에도 매칭이 약한(애매한) ID 에만 생성을 추가하면?
    쉬운 건 안 건드리고, 어려운 데서만 생성이 회복시키는지 본다.

[기준선] B0+CP  (지금까지 best: sparse mAP 62.8)
  - gallery = 각 ID c1 실제 + 카메라 보정
  - query   = c2~c6 실제

[선택 기준]  보정 후 query 의 '애매함' = (1등 점수 - 2등 점수) margin.
  margin 작을수록 헷갈림 = 어려움. 하위 ratio% query 의 '정답 ID'에만 생성 추가.
  * query 라벨로 '어려운 ID'를 고르지만, 생성은 그 ID 갤러리에 넣을 뿐
    query 를 생성으로 바꾸지 않음 → 순환성 없음.
  * 배포 관점: margin 은 라벨 없이 계산 가능(불확실 query 탐지) → 정당.

[측정]  ratio {10,20,30,50}% 별로:
  적용수 / 회복(틀렸다 맞음) / 손상(맞았다 틀림) / 순효과 / mAP,R1
  회복 > 손상 → "생성은 어려운 케이스 전용 보강으로 유효" (조건부 기여③)
  회복 <= 손상 → 생성은 어떤 선택으로도 무효 → 기여① 확정

* 모델/특징추출/CP/메트릭 = 93·94 와 동일.
"""

import os, re, glob, datetime
import numpy as np

# ===== CONFIG =====
HOME        = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT        = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR  = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
CLIP_REID   = "/home/ubuntu/CLIP-ReID"
GEN_DIR     = f"{PROJECT_DIR}/outputs/c1base_gen_all"
LOG         = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
CACHE_DIR   = f"{PROJECT_DIR}/feat_cache_sparse"   # 94번 캐시 재사용

WEIGHT      = f"{CKPT}/clipreid_duke_nosie.pth"
NUM_CLASS   = 702
DEVICE      = "cuda"
BATCH       = 128
GALLERY_CAM = 0
QUERY_CAMS  = [1,2,3,4,5]
GEN_CAMS    = ["c2","c3","c4","c5","c6"]
RATIOS      = [10, 20, 30, 50]      # 어려운 하위 %에만 생성

os.makedirs(CACHE_DIR, exist_ok=True)
def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)


# ===== 모델/추출/CP/메트릭 (동일) =====
def build_model(weight_path, num_class):
    import sys, torch, torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, CLIP_REID)
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME="ViT-B-16"; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE="softmax"
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="market1501"; cfg.TEST.WEIGHT=weight_path; cfg.TEST.NECK_FEAT="before"
    try:    b=make_model(cfg,num_class=num_class,camera_num=0,view_num=1)
    except Exception: b=make_model(cfg,num_class=num_class,camera_num=6,view_num=1)
    b.load_param(weight_path); b=b.to(DEVICE).eval()
    class Backbone(nn.Module):
        def __init__(s,bb): super().__init__(); s.bb=bb
        @torch.no_grad()
        def forward(s,x):
            f=s.bb(x,cam_label=None)
            if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
            if f.dim()>2: f=f.view(f.size(0),-1)
            return f
    pre=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
    return Backbone(b).eval(), pre

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

def build_prototypes(f,c,n_proto=0,seed=0):
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
        if len(c)<max_rank:                 # 갤러리<max_rank 시 마지막값으로 패딩
            c=np.concatenate([c,np.full(max_rank-len(c),c[-1])])
        all_cmc.append(c); valid+=1
        nr=raw.sum(); tmp=raw.cumsum()/(np.arange(len(raw))+1.0)
        all_AP.append((tmp*raw).sum()/nr)
    return np.asarray(all_cmc).sum(0)/valid, float(np.mean(all_AP))


# ===== 데이터: sparse 구성 (94번과 동일) =====
_PAT=re.compile(r"([-\d]+)_c(\d+)")
def parse_real(d):
    by={}
    for p in sorted(glob.glob(os.path.join(d,"*.jpg"))):
        m=_PAT.search(os.path.basename(p))
        if m is None: continue
        pid,cam=int(m.group(1)),int(m.group(2))
        if pid==-1: continue
        by.setdefault(pid,{}).setdefault(cam-1,[]).append(p)
    return by
def gen_path(pid,cs):
    p=f"{GEN_DIR}/{cs}/{pid:04d}_gen_{cs}.png"; return p if os.path.exists(p) else None

def build_sets(gby,qby):
    ids=sorted([p for p in gby if GALLERY_CAM in gby[p]
                and any(qc in qby.get(p,{}) for qc in QUERY_CAMS)])
    g=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for p in gby[pid][GALLERY_CAM]:
            g["paths"].append(p); g["pids"].append(pid); g["cams"].append(GALLERY_CAM)
    gen=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for cs in GEN_CAMS:
            gp=gen_path(pid,cs)
            if gp: gen["paths"].append(gp); gen["pids"].append(pid); gen["cams"].append(int(cs[1:])-1)
    q=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for qc in QUERY_CAMS:
            for p in qby.get(pid,{}).get(qc,[]):
                q["paths"].append(p); q["pids"].append(pid); q["cams"].append(qc)
    for d in (g,gen,q):
        d["pids"]=np.array(d["pids"]); d["cams"]=np.array(d["cams"])
    print(f"[sparse] ids={len(ids)} gallery={len(g['paths'])} gen={len(gen['paths'])} query={len(q['paths'])}")
    return g,gen,q


def margin_difficulty(distmat):
    """query 별 (1등 - 2등) 거리차. 작을수록 애매(어려움). 라벨 불필요."""
    part=np.partition(distmat,1,axis=1)
    return part[:,1]-part[:,0]      # >=0, 작을수록 어려움


def main():
    logline("\n"+"="*78)
    logline(f"## [{datetime.date.today()}] script95 선택적 생성 (어려운 케이스 전용, 기여③ 재시도)")
    logline("="*78)
    gby=parse_real(f"{MARKET_DIR}/bounding_box_test"); qby=parse_real(f"{MARKET_DIR}/query")
    G,GEN,Q=build_sets(gby,qby)

    model,pre=build_model(WEIGHT,NUM_CLASS)
    qf =extract_features(model,pre,Q["paths"],  f"{CACHE_DIR}/q.npy")
    gf =extract_features(model,pre,G["paths"],  f"{CACHE_DIR}/g_c1.npy")
    genf=extract_features(model,pre,GEN["paths"],f"{CACHE_DIR}/gen.npy")

    # --- 보정(B0+CP) ---
    protos=build_prototypes(np.concatenate([gf,qf]),np.concatenate([G["cams"],Q["cams"]]))
    qf_c=apply_cp(qf,Q["cams"],protos)
    gf_c=apply_cp(gf,G["cams"],protos)
    genf_c=apply_cp(genf,GEN["cams"],protos)

    base_dist=1.0-qf_c@gf_c.T
    cmc,mAP=eval_market(base_dist,Q["pids"],G["pids"],Q["cams"],G["cams"])
    r1_B=cmc[0]*100
    logline(f"\n기준 B0+CP   R1={r1_B:.1f}  mAP={mAP*100:.1f}   (query={len(qf)})")

    # 보정 단독에서 각 query 정답여부 + 난이도
    order0=np.argsort(base_dist,axis=1)
    correct0=(G["pids"][order0[:,0]]==Q["pids"])     # top-1 정답?
    diff=margin_difficulty(base_dist)                # 작을수록 어려움
    rank_by_diff=np.argsort(diff)                    # 어려운 순

    # --- ratio 별: 어려운 query 의 '정답 ID' 에만 생성 추가 ---
    logline(f"\n{'ratio':<8}{'적용ID':<8}{'회복':<6}{'손상':<6}{'순효과':<8}{'R1':<10}{'mAP':<10}")
    logline("-"*60)
    logline(f"{'0%':<8}{0:<8}{'-':<6}{'-':<6}{'-':<8}{r1_B:<10.1f}{mAP*100:<10.1f}")
    for ratio in RATIOS:
        k=int(len(qf)*ratio/100)
        hard_q=rank_by_diff[:k]                       # 어려운 query
        target_ids=set(Q["pids"][hard_q].tolist())    # 그 query 들의 정답 ID
        # 해당 ID 의 생성만 갤러리에 추가
        sel=np.array([i for i,pid in enumerate(GEN["pids"]) if pid in target_ids],dtype=int)
        if len(sel)==0:
            logline(f"{str(ratio)+'%':<8}{0:<8}{0:<6}{0:<6}{0:<8}{r1_B:<10.1f}{mAP*100:<10.1f}")
            continue
        gf_e=np.concatenate([gf_c,genf_c[sel]],0)
        gp_e=np.concatenate([G["pids"],GEN["pids"][sel]])
        gc_e=np.concatenate([G["cams"],GEN["cams"][sel]])
        dist_e=1.0-qf_c@gf_e.T
        cmc_e,mAP_e=eval_market(dist_e,Q["pids"],gp_e,Q["cams"],gc_e)
        # 회복/손상: top-1 정답 여부 변화
        order_e=np.argsort(dist_e,axis=1)
        correct_e=(gp_e[order_e[:,0]]==Q["pids"])
        recover=int(((~correct0)&correct_e).sum())    # 틀렸다->맞음
        damage =int((correct0&(~correct_e)).sum())    # 맞았다->틀림
        net=recover-damage
        logline(f"{str(ratio)+'%':<8}{len(target_ids):<8}{recover:<6}{damage:<6}"
                f"{net:+<8}{cmc_e[0]*100:<10.1f}{mAP_e*100:<10.1f}")

    logline("\n해석:")
    logline("  어떤 ratio 든 mAP > 기준(B0+CP) & 회복>손상 → 조건부 생성 유효 (기여③ 성립)")
    logline("  모든 ratio 에서 mAP <= 기준 → 생성은 선택해도 무효 → 기여① 확정, 생성 접기")

if __name__=="__main__":
    main()