#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
94_sparse_gen_eval.py  ―  기여③: "생성이 이득 나는 조건" 표준 메트릭으로 증명

[이 실험이 뭐냐]
  표준 full 갤러리엔 각 ID의 여러 카메라 실제 사진이 이미 다 있어서 생성이 묻힌다
  (= 기여①, 강한 모델에서 생성 무효). 그래서 갤러리를 'ID당 c1 한 카메라'로
  희박하게(sparse) 만든 뒤, 빈 시점(c2~c6)을 생성 이미지로 채웠을 때
  mAP/R1 이 오르는지 본다. 오르면 → "생성이 이득 나는 조건" 증명(기여③).

[비교 구성]  (Market 평가, Duke학습 weight, frozen)
  B0           : 갤러리 = 각 ID의 c1 실제 이미지만        (희소 baseline)
  E1           : B0 + 생성 시점(c1base_gen_all) 추가      (생성 확장)
  B0+CP        : B0 에 카메라 보정                          (보정 단독)
  E1+CP        : 생성 + 보정                                (최종 결합)

  query = 각 ID의 c2~c6 '실제' 이미지 (생성은 query에 절대 안 넣음 → 순환성 차단)

[무엇을 증명하나]
  E1 > B0        → 생성이 빈 시점을 채워 이득 (기여③ 성립)
  E1 ~= / < B0   → 강한 모델에선 sparse 에서도 생성이 약함 → 기여① 강화 + 보정으로 해결
  E1+CP 최고     → 생성·보정 결합이 베스트

* 모델/특징추출/메트릭/CP = 93번과 동일. 데이터 구성만 sparse 로 바뀜.
* 생성물: outputs/c1base_gen_all/{cam}/{pid}_gen_{cam}.png  (있는 것만 추가)
"""

import os, re, glob, datetime
import numpy as np

# =====================================================================
# 0. CONFIG
# =====================================================================
HOME        = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT        = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR  = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
CLIP_REID   = "/home/ubuntu/CLIP-ReID"
GEN_DIR     = f"{PROJECT_DIR}/outputs/c1base_gen_all"   # {cam}/{pid}_gen_{cam}.png
LOG         = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

WEIGHT      = f"{CKPT}/clipreid_duke_nosie.pth"          # Duke학습 → Market평가 (cross-domain)
NUM_CLASS   = 702
DEVICE      = "cuda"
BATCH       = 128
CACHE_DIR   = f"{PROJECT_DIR}/feat_cache_sparse"

GALLERY_CAM = 0            # c1 (0-index) = 갤러리에 남길 단일 카메라
QUERY_CAMS  = [1,2,3,4,5]  # c2~c6 = query 로 쓸 카메라
GEN_CAMS    = ["c2","c3","c4","c5","c6"]

os.makedirs(CACHE_DIR, exist_ok=True)


def logline(m):
    with open(LOG, "a") as fp: fp.write(m + "\n")
    print(m)


# =====================================================================
# 1. 모델 + 특징추출  (93번과 동일)
# =====================================================================
def build_model(weight_path, num_class):
    import sys, torch
    import torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, CLIP_REID)
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME="ViT-B-16"; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE="softmax"
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="market1501"
    cfg.TEST.WEIGHT=weight_path; cfg.TEST.NECK_FEAT="before"
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
    pre=T.Compose([T.Resize([256,128]),T.ToTensor(),
                   T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    return Backbone(b).eval(), pre


def extract_features(model, preprocess, paths, cache_path=None):
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
    print()
    feats=np.concatenate(feats,0).astype(np.float32)
    if cache_path: np.save(cache_path,feats)
    return feats


# =====================================================================
# 2. CP / 메트릭  (93번과 동일)
# =====================================================================
def build_prototypes(g_feats, g_cams, mode="percam", n_proto=0, seed=0):
    rng=np.random.RandomState(seed)
    if mode=="global": return {"__global__": g_feats.mean(0)}
    protos={}
    for c in np.unique(g_cams):
        sel=np.where(g_cams==c)[0]
        if n_proto>0 and len(sel)>n_proto: sel=rng.choice(sel,n_proto,replace=False)
        protos[int(c)]=g_feats[sel].mean(0)
    return protos

def apply_cp(feats, cams, protos, mode="percam"):
    out=feats.copy()
    if mode=="global": out=out-protos["__global__"][None,:]
    else:
        gmean=np.mean(list(protos.values()),axis=0)
        for i,c in enumerate(cams): out[i]=out[i]-protos.get(int(c),gmean)
    n=np.linalg.norm(out,axis=1,keepdims=True); n[n==0]=1e-12
    return (out/n).astype(np.float32)

def cosine_distmat(qf,gf): return (1.0-qf@gf.T).astype(np.float32)

def eval_market(distmat,q_pids,g_pids,q_cams,g_cams,max_rank=50):
    num_q=distmat.shape[0]
    indices=np.argsort(distmat,axis=1)
    matches=(g_pids[indices]==q_pids[:,None]).astype(np.int32)
    all_cmc,all_AP,valid=[],[],0
    for qi in range(num_q):
        order=indices[qi]
        remove=(g_pids[order]==q_pids[qi])&(g_cams[order]==q_cams[qi])
        keep=~remove; raw=matches[qi][keep]
        if not raw.any(): continue
        cmc=raw.cumsum(); cmc[cmc>1]=1
        all_cmc.append(cmc[:max_rank]); valid+=1
        num_rel=raw.sum()
        tmp=raw.cumsum()/(np.arange(len(raw))+1.0)
        all_AP.append((tmp*raw).sum()/num_rel)
    cmc=np.asarray(all_cmc).sum(0)/valid
    return cmc, float(np.mean(all_AP))


# =====================================================================
# 3. 데이터 구성  ―  sparse 갤러리 + 생성
#    파일명: Market real 0001_c1s1_...jpg / gen 0001_gen_c2.png
# =====================================================================
_PAT = re.compile(r"([-\d]+)_c(\d+)")

def parse_real(dirpath):
    """returns dict[pid] -> dict[camid] -> [paths]"""
    by={}
    for p in sorted(glob.glob(os.path.join(dirpath,"*.jpg"))):
        m=_PAT.search(os.path.basename(p))
        if m is None: continue
        pid,cam=int(m.group(1)),int(m.group(2))
        if pid==-1: continue
        by.setdefault(pid,{}).setdefault(cam-1,[]).append(p)
    return by

def gen_path(pid, cam_str):
    p=f"{GEN_DIR}/{cam_str}/{pid:04d}_gen_{cam_str}.png"
    return p if os.path.exists(p) else None

def build_sparse_sets(gallery_by, query_by):
    """
    갤러리: 각 ID 의 c1 실제 이미지만.
    query : 각 ID 의 c2~c6 실제 이미지 (full gallery 의 test set 그대로 사용).
    생성  : 각 ID 의 c2~c6 생성 이미지(있는 것만).
    평가 대상 ID = c1 갤러리 + c2~c6 query 가 모두 있는 ID.
    """
    # 평가 ID: 갤러리에 c1 있고, query 카메라가 하나라도 있는 ID
    valid_ids=[pid for pid in gallery_by
               if GALLERY_CAM in gallery_by[pid]
               and any(qc in query_by.get(pid,{}) for qc in QUERY_CAMS)]
    valid_ids=sorted(valid_ids)

    g_paths,g_pids,g_cams,g_isgen=[],[],[],[]
    for pid in valid_ids:                       # --- 갤러리: c1 실제 ---
        for p in gallery_by[pid][GALLERY_CAM]:
            g_paths.append(p); g_pids.append(pid); g_cams.append(GALLERY_CAM); g_isgen.append(0)

    gen_paths,gen_pids,gen_cams=[],[],[]         # --- 생성 시점(있는 것만) ---
    n_gen=0
    for pid in valid_ids:
        for cam_str in GEN_CAMS:
            gp=gen_path(pid,cam_str)
            if gp:
                gen_paths.append(gp); gen_pids.append(pid)
                gen_cams.append(int(cam_str[1:])-1); n_gen+=1

    q_paths,q_pids,q_cams=[],[],[]               # --- query: c2~c6 실제 ---
    for pid in valid_ids:
        for qc in QUERY_CAMS:
            for p in query_by.get(pid,{}).get(qc,[]):
                q_paths.append(p); q_pids.append(pid); q_cams.append(qc)

    print(f"[sparse] ids={len(valid_ids)}  gallery(c1 real)={len(g_paths)}  "
          f"gen={n_gen}  query(c2~c6 real)={len(q_paths)}")
    return (dict(paths=g_paths,pids=np.array(g_pids),cams=np.array(g_cams)),
            dict(paths=gen_paths,pids=np.array(gen_pids),cams=np.array(gen_cams)),
            dict(paths=q_paths,pids=np.array(q_pids),cams=np.array(q_cams)))


# =====================================================================
# 4. 실행
# =====================================================================
def report(tag,cmc,mAP,base=None):
    r1,r5,r10=cmc[0]*100,cmc[4]*100,cmc[9]*100
    s=f"{tag:<14} R1={r1:5.1f}  R5={r5:5.1f}  R10={r10:5.1f}  mAP={mAP*100:5.1f}"
    if base is not None: s+=f"   (R1 {r1-base[0]:+.1f}, mAP {mAP*100-base[1]:+.1f})"
    logline(s); return (r1,mAP*100)

def main():
    logline("\n"+"="*78)
    logline(f"## [{datetime.date.today()}] script94 sparse 갤러리 + 생성 (기여③)")
    logline(f"   weight={os.path.basename(WEIGHT)}  gallery=c1 only  query=c2~c6 real")
    logline("="*78)

    gallery_by=parse_real(f"{MARKET_DIR}/bounding_box_test")
    query_by  =parse_real(f"{MARKET_DIR}/query")
    G,GEN,Q=build_sparse_sets(gallery_by,query_by)

    model,pre=build_model(WEIGHT,NUM_CLASS)
    qf =extract_features(model,pre,Q["paths"],  f"{CACHE_DIR}/q.npy")
    gf =extract_features(model,pre,G["paths"],  f"{CACHE_DIR}/g_c1.npy")
    genf=extract_features(model,pre,GEN["paths"],f"{CACHE_DIR}/gen.npy") if len(GEN["paths"]) else np.empty((0,qf.shape[1]),np.float32)

    # ---- B0: c1 실제만 ----
    cmc,mAP=eval_market(cosine_distmat(qf,gf),Q["pids"],G["pids"],Q["cams"],G["cams"])
    base=report("B0 (c1 real)",cmc,mAP)

    # ---- E1: + 생성 ----
    if len(genf):
        gf_e =np.concatenate([gf,genf],0)
        gp_e =np.concatenate([G["pids"],GEN["pids"]])
        gc_e =np.concatenate([G["cams"],GEN["cams"]])
        cmc,mAP=eval_market(cosine_distmat(qf,gf_e),Q["pids"],gp_e,Q["cams"],gc_e)
        report("E1 (+gen)",cmc,mAP,base)

    # ---- B0+CP / E1+CP : 카메라 보정 ----
    # proto 는 sparse 갤러리(c1) + query 카메라 통계가 필요하므로, 평가에 쓰는
    # 실제 이미지 카메라별 평균으로 구성(label-free, query-agnostic).
    proto_src_f=np.concatenate([gf,qf],0)
    proto_src_c=np.concatenate([G["cams"],Q["cams"]])
    protos=build_prototypes(proto_src_f,proto_src_c,mode="percam")
    qf_c=apply_cp(qf,Q["cams"],protos)
    gf_c=apply_cp(gf,G["cams"],protos)
    cmc,mAP=eval_market(cosine_distmat(qf_c,gf_c),Q["pids"],G["pids"],Q["cams"],G["cams"])
    report("B0+CP",cmc,mAP,base)

    if len(genf):
        genf_c=apply_cp(genf,GEN["cams"],protos)
        gf_ec =np.concatenate([gf_c,genf_c],0)
        gp_e  =np.concatenate([G["pids"],GEN["pids"]])
        gc_e  =np.concatenate([G["cams"],GEN["cams"]])
        cmc,mAP=eval_market(cosine_distmat(qf_c,gf_ec),Q["pids"],gp_e,Q["cams"],gc_e)
        report("E1+CP",cmc,mAP,base)

    logline("\n해석: E1>B0 → 생성이 빈 시점 채워 이득(기여③ 성립) / "
            "E1<=B0 → sparse 에서도 생성 약함→기여① 강화 / E1+CP 최고 → 결합이 베스트")

if __name__=="__main__":
    main()