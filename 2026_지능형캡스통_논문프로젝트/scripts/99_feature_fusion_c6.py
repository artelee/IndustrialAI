#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
99_feature_fusion_c6.py  ―  트랙② 일반성 검증: c6base 로 동일 α sweep

[이 실험이 뭐냐]
  98번(c1base) 에서 α=0.8 가 미세하게 양성(+0.3 mAP) + α 단조 감소 곡선 확인.
  같은 패턴이 갤러리 카메라 바꾼 c6base 에서도 재현되는지 본다.
  재현되면 → 현상의 일반성 확보 (논문에 c1/c6 표 나란히).

[설정]   c1base → c6base 만 다름. 평가 구성·메트릭·α 동일.
  gallery_cam = c6, query = c1~c5, gen = c6base_gen_all
"""

import os, re, glob, datetime
import numpy as np

# ===== CONFIG =====
HOME        = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT        = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR  = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
CLIP_REID   = "/home/ubuntu/CLIP-ReID"
GEN_DIR     = f"{PROJECT_DIR}/outputs/c6base_gen_all"          # ← c6base
LOG         = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
CACHE_DIR   = f"{PROJECT_DIR}/feat_cache_c6sparse"             # ← 97번 캐시 재사용

WEIGHT      = f"{CKPT}/clipreid_duke_nosie.pth"
NUM_CLASS   = 702
DEVICE      = "cuda"
BATCH       = 128

GALLERY_CAM = 5                                                # ← c6
QUERY_CAMS  = [0, 1, 2, 3, 4]                                  # ← c1~c5
GEN_CAMS    = ["c1", "c2", "c3", "c4", "c5"]                   # ← c1~c5 생성
ALPHAS      = [1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.0]         # 0.7,0.9 추가 (sweet spot 더 정확히)

def logline(m):
    with open(LOG, "a") as fp: fp.write(m + "\n")
    print(m)


# ===== 모델/추출/CP/메트릭 =====
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
def cosine_distmat(qf,gf): return (1.0-qf@gf.T).astype(np.float32)

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
    return np.asarray(all_cmc).sum(0)/valid, float(np.mean(all_AP))


# ===== 데이터 =====
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
    print(f"[c6 sparse] ids={len(ids)} gallery(c6)={len(g['paths'])} "
          f"gen={len(gen['paths'])} query(c1~c5)={len(q['paths'])}")
    return g,gen,q


def fuse_prototypes(g_feats, g_pids, gen_feats, gen_pids, alpha):
    proto_f, proto_p = [], []
    for pid in np.unique(g_pids):
        real_mean = g_feats[g_pids == pid].mean(0)
        gen_sel = gen_feats[gen_pids == pid] if gen_feats is not None else None
        if alpha >= 1.0 or gen_sel is None or len(gen_sel) == 0:
            p = real_mean
        elif alpha <= 0.0:
            p = gen_sel.mean(0)
        else:
            p = alpha * real_mean + (1 - alpha) * gen_sel.mean(0)
        p = p / (np.linalg.norm(p) + 1e-12)
        proto_f.append(p); proto_p.append(pid)
    return np.array(proto_f, dtype=np.float32), np.array(proto_p)


def report(tag, cmc, mAP, base=None):
    r1, r5, r10 = cmc[0]*100, cmc[4]*100, cmc[9]*100
    s = f"{tag:<22} R1={r1:5.1f}  R5={r5:5.1f}  R10={r10:5.1f}  mAP={mAP*100:5.1f}"
    if base is not None: s += f"   (R1 {r1-base[0]:+.1f}, mAP {mAP*100-base[1]:+.1f})"
    logline(s); return (r1, mAP*100)


def main():
    if not os.path.isdir(GEN_DIR):
        logline(f"[!] {GEN_DIR} 없음. 44번 c6 base 생성 먼저 끝내야 함.")
        return

    logline("\n" + "=" * 78)
    logline(f"## [{datetime.date.today()}] script99 feature 융합 (트랙② , c6base) — 일반성 검증")
    logline(f"   gallery=c6, query=c1~c5, gen=c6→c1~c5 | α sweep {ALPHAS}")
    logline("=" * 78)

    gby = parse_real(f"{MARKET_DIR}/bounding_box_test")
    qby = parse_real(f"{MARKET_DIR}/query")
    G, GEN, Q = build_sets(gby, qby)
    if len(GEN["paths"]) == 0:
        logline("[!] 생성 이미지 0장. 44번 먼저 돌려서 c6base_gen_all/ 채우기.")
        return

    model, pre = build_model(WEIGHT, NUM_CLASS)
    qf   = extract_features(model, pre, Q["paths"],   f"{CACHE_DIR}/q.npy")
    gf   = extract_features(model, pre, G["paths"],   f"{CACHE_DIR}/g_c6.npy")
    genf = extract_features(model, pre, GEN["paths"], f"{CACHE_DIR}/gen.npy")

    proto_src_f = np.concatenate([gf, qf])
    proto_src_c = np.concatenate([G["cams"], Q["cams"]])
    protos_cp = build_prototypes(proto_src_f, proto_src_c)

    # === 보정 없음 ===
    logline("\n[보정 없음]")
    base_a1 = None
    for a in ALPHAS:
        gen_for_fuse = genf if a < 1.0 else None
        pf, pp = fuse_prototypes(gf, G["pids"], gen_for_fuse, GEN["pids"], a)
        pc = np.full(len(pf), -1, dtype=np.int64)
        cmc, mAP = eval_market(cosine_distmat(qf, pf), Q["pids"], pp, Q["cams"], pc)
        tag = f"F α={a:.1f}" + (" (=원본만)" if a == 1.0 else " (=생성만)" if a == 0.0 else "")
        b = report(tag, cmc, mAP, base_a1)
        if a == 1.0: base_a1 = b

    # === CP 보정 ===
    logline("\n[CP 보정]")
    qf_c   = apply_cp(qf,   Q["cams"],   protos_cp)
    gf_c   = apply_cp(gf,   G["cams"],   protos_cp)
    genf_c = apply_cp(genf, GEN["cams"], protos_cp)
    base_a1_cp = None
    for a in ALPHAS:
        gen_for_fuse = genf_c if a < 1.0 else None
        pf, pp = fuse_prototypes(gf_c, G["pids"], gen_for_fuse, GEN["pids"], a)
        pc = np.full(len(pf), -1, dtype=np.int64)
        cmc, mAP = eval_market(cosine_distmat(qf_c, pf), Q["pids"], pp, Q["cams"], pc)
        tag = f"F α={a:.1f}+CP" + (" (=원본만)" if a == 1.0 else " (=생성만)" if a == 0.0 else "")
        b = report(tag, cmc, mAP, base_a1_cp)
        if a == 1.0: base_a1_cp = b

    logline("\n해석:")
    logline("  c1base 와 c6base 에서 같은 α 부근(0.7~0.9)에서 양성 → 현상 일반성 확보")
    logline("  → 논문에 c1/c6 표 나란히 + α 곡선 그림 한 장")


if __name__ == "__main__":
    main()