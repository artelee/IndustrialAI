#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
100_selective_fusion.py  ―  confidence-gated feature 융합
                            (98/99 트랙② + 95번 발상의 진화)

[아이디어]
  98/99 에서 확인: feature 융합(α=0.8~0.9)은 작지만 양성.
  근데 모든 ID 에 똑같이 적용하면 쉬운 ID 까지 살짝 깎임 → 작은 양성에 머묾.

  → 자신 없는 query 에만 융합 prototype 으로 재매칭.
     자신 있는 query 는 원본 그대로 두기.

[confidence 기준]
  1등 코사인 유사도 절대값 (낮을수록 헤맴 → 생성 도움 필요)
  margin(1-2등) 보다 직관적: '원본만으론 답을 못 찾는' 케이스를 직접 가리킴

[비교 공간 2개 한 번에]
  A. c1 갤러리, 보정 없음     (98번에서 α=0.8 에 +0.3 확인됨)
  B. c6 갤러리, +CP            (99번에서 α=0.9+CP 에 +0.3, R1 +0.5 확인됨)

[프로토콜]
  1) α=1.0 prototype 갤러리로 1차 매칭 → query 별 1등 유사도(top1_sim) 기록
  2) 하위 X% (top1_sim 낮은 query) 선택
  3) 그 query 만 α=0.8/0.9 융합 prototype 갤러리로 재매칭
  4) X ∈ {10,20,30,50}, α ∈ {0.8, 0.9}, 공간 ∈ {A, B}

* 94/97 캐시 모두 재사용 → 새 추출 0
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

WEIGHT      = f"{CKPT}/clipreid_duke_nosie.pth"
NUM_CLASS   = 702
DEVICE      = "cuda"
BATCH       = 128

# 두 공간 정의
SPACES = {
    "A_c1_noCP": dict(
        gen_dir = f"{PROJECT_DIR}/outputs/c1base_gen_all",
        cache   = f"{PROJECT_DIR}/feat_cache_sparse",
        g_cache_name = "g_c1.npy",
        gallery_cam = 0,
        query_cams  = [1, 2, 3, 4, 5],
        gen_cams    = ["c2", "c3", "c4", "c5", "c6"],
        use_cp      = False,
    ),
    "B_c6_CP": dict(
        gen_dir = f"{PROJECT_DIR}/outputs/c6base_gen_all",
        cache   = f"{PROJECT_DIR}/feat_cache_c6sparse",
        g_cache_name = "g_c6.npy",
        gallery_cam = 5,
        query_cams  = [0, 1, 2, 3, 4],
        gen_cams    = ["c1", "c2", "c3", "c4", "c5"],
        use_cp      = True,
    ),
}
#RATIOS = [10, 20, 30, 50]      # 자신 없는 하위 X%
RATIOS = [10, 20, 30, 50, 70, 100]
ALPHAS = [0.8, 0.9]            # 98/99 의 sweet spot

def logline(m):
    with open(LOG, "a") as fp: fp.write(m + "\n")
    print(m)


# ===== 모델/추출/CP/메트릭 =====
def build_model():
    import sys, torch, torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, CLIP_REID)
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME="ViT-B-16"; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE="softmax"
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5]*3; cfg.INPUT.PIXEL_STD=[0.5]*3
    cfg.DATASETS.NAMES="market1501"; cfg.TEST.WEIGHT=WEIGHT; cfg.TEST.NECK_FEAT="before"
    try:    b=make_model(cfg,num_class=NUM_CLASS,camera_num=0,view_num=1)
    except Exception: b=make_model(cfg,num_class=NUM_CLASS,camera_num=6,view_num=1)
    b.load_param(WEIGHT); b=b.to(DEVICE).eval()
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

def build_cp_protos(f,c):
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
def gen_path(gd,pid,cs):
    p=f"{gd}/{cs}/{pid:04d}_gen_{cs}.png"; return p if os.path.exists(p) else None

def build_sets(gby,qby,cfg_space):
    GC=cfg_space["gallery_cam"]; QCs=cfg_space["query_cams"]; GENCs=cfg_space["gen_cams"]; gd=cfg_space["gen_dir"]
    ids=sorted([p for p in gby if GC in gby[p] and any(qc in qby.get(p,{}) for qc in QCs)])
    g=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for p in gby[pid][GC]:
            g["paths"].append(p); g["pids"].append(pid); g["cams"].append(GC)
    gen=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for cs in GENCs:
            gp=gen_path(gd,pid,cs)
            if gp: gen["paths"].append(gp); gen["pids"].append(pid); gen["cams"].append(int(cs[1:])-1)
    q=dict(paths=[],pids=[],cams=[])
    for pid in ids:
        for qc in QCs:
            for p in qby.get(pid,{}).get(qc,[]):
                q["paths"].append(p); q["pids"].append(pid); q["cams"].append(qc)
    for d in (g,gen,q):
        d["pids"]=np.array(d["pids"]); d["cams"]=np.array(d["cams"])
    return ids,g,gen,q


# ===== 핵심: prototype 융합 =====
def fuse_prototypes(g_feats, g_pids, gen_feats, gen_pids, alpha):
    proto_f, proto_p = [], []
    for pid in np.unique(g_pids):
        real_mean = g_feats[g_pids == pid].mean(0)
        gen_sel = gen_feats[gen_pids == pid] if gen_feats is not None and len(gen_feats)>0 else None
        if alpha >= 1.0 or gen_sel is None or len(gen_sel) == 0:
            p = real_mean
        elif alpha <= 0.0:
            p = gen_sel.mean(0)
        else:
            p = alpha * real_mean + (1 - alpha) * gen_sel.mean(0)
        p = p / (np.linalg.norm(p) + 1e-12)
        proto_f.append(p); proto_p.append(pid)
    return np.array(proto_f, dtype=np.float32), np.array(proto_p)


def hybrid_distmat(qf, pf_orig, pf_fused, hard_mask):
    """
    쉬운 query 는 pf_orig 로, 어려운 query(hard_mask=True) 는 pf_fused 로 매칭.
    실제로는 둘 다 동일 prototype 갤러리지만 vector 만 다름.
    """
    d_orig = 1.0 - qf @ pf_orig.T
    d_fused= 1.0 - qf @ pf_fused.T
    d = d_orig.copy()
    d[hard_mask] = d_fused[hard_mask]
    return d.astype(np.float32)


def run_space(name, cfg_space, model, pre, gby, qby):
    logline("\n" + "=" * 78)
    logline(f"### 공간 {name}  (gallery_cam=c{cfg_space['gallery_cam']+1}, "
            f"CP={'O' if cfg_space['use_cp'] else 'X'})")
    logline("=" * 78)
    ids, G, GEN, Q = build_sets(gby, qby, cfg_space)
    print(f"[{name}] ids={len(ids)} gallery={len(G['paths'])} gen={len(GEN['paths'])} query={len(Q['paths'])}")
    if len(GEN["paths"]) == 0:
        logline(f"[!] {cfg_space['gen_dir']} 비어있음. skip.")
        return

    qf  = extract_features(model, pre, Q["paths"],   f"{cfg_space['cache']}/q.npy")
    gf  = extract_features(model, pre, G["paths"],   f"{cfg_space['cache']}/{cfg_space['g_cache_name']}")
    genf= extract_features(model, pre, GEN["paths"], f"{cfg_space['cache']}/gen.npy")

    if cfg_space["use_cp"]:
        src_f = np.concatenate([gf, qf]); src_c = np.concatenate([G["cams"], Q["cams"]])
        cp = build_cp_protos(src_f, src_c)
        qf = apply_cp(qf, Q["cams"], cp)
        gf = apply_cp(gf, G["cams"], cp)
        genf = apply_cp(genf, GEN["cams"], cp)

    # 1차 매칭용 prototype (α=1.0, 원본만)
    pf_orig, pp = fuse_prototypes(gf, G["pids"], None, None, 1.0)
    pc = np.full(len(pf_orig), -1, dtype=np.int64)
    d_orig = 1.0 - qf @ pf_orig.T
    cmc_b, mAP_b = eval_market(d_orig, Q["pids"], pp, Q["cams"], pc)
    r1_b, map_b = cmc_b[0]*100, mAP_b*100
    logline(f"\n  baseline (원본 prototype)   R1={r1_b:.1f}  mAP={map_b:.1f}")

    # 1등 유사도 = (1 - 1등 거리). 낮을수록 헤맴.
    top1_sim = 1.0 - d_orig.min(axis=1)
    rank = np.argsort(top1_sim)   # 작은(=헷갈리는) 순

    # X% × α sweep
    logline(f"\n  {'X%':<6}{'α':<6}{'적용수':<8}{'R1':<10}{'mAP':<10}{'ΔR1':<10}{'ΔmAP':<10}")
    logline("  " + "-" * 60)
    for alpha in ALPHAS:
        pf_fused, _ = fuse_prototypes(gf, G["pids"], genf, GEN["pids"], alpha)
        for x in RATIOS:
            k = int(len(qf) * x / 100)
            hard = np.zeros(len(qf), dtype=bool); hard[rank[:k]] = True
            d_hyb = hybrid_distmat(qf, pf_orig, pf_fused, hard)
            cmc, mAP = eval_market(d_hyb, Q["pids"], pp, Q["cams"], pc)
            r1, mp = cmc[0]*100, mAP*100
            logline(f"  {x:<6}{alpha:<6}{k:<8}{r1:<10.1f}{mp:<10.1f}"
                    f"{r1-r1_b:<+10.1f}{mp-map_b:<+10.1f}")


def main():
    logline("\n" + "=" * 78)
    logline(f"## [{datetime.date.today()}] script100 confidence-gated feature 융합")
    logline(f"   하위 X% (1등 유사도 낮은 query) 만 α 융합 prototype 으로 재매칭")
    logline("=" * 78)

    model, pre = build_model()
    gby = parse_real(f"{MARKET_DIR}/bounding_box_test")
    qby = parse_real(f"{MARKET_DIR}/query")

    for name, cfg_space in SPACES.items():
        if not os.path.isdir(cfg_space["gen_dir"]):
            logline(f"[skip] {name}: {cfg_space['gen_dir']} 없음")
            continue
        try:
            run_space(name, cfg_space, model, pre, gby, qby)
        except Exception as e:
            logline(f"[error] {name}: {e}")

    logline("\n해석:")
    logline("  특정 X% 에서 ΔmAP > 98/99 의 +0.3 → confidence 게이팅이 융합 효과 증폭 ✅")
    logline("  모든 X% 에서 ΔmAP ≤ +0.3 → 게이팅은 추가 이득 없음, 융합 단독으로 충분")
    logline("  ΔmAP < 0 인 X% 가 많으면 → 게이팅 잘못 작동 (어려운 query 정의 재검토)")


if __name__ == "__main__":
    main()