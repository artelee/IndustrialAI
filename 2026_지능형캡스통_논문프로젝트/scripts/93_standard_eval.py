#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
93_standard_eval.py  ―  표준 cross-domain 평가로 'Training-Free 카메라 보정(CP)' 검증

[이 실험이 뭐냐]
  네 주 기여(CP)를 '표준 full-gallery' 프로토콜에서 측정해 학계 비교 가능한
  mAP/Rank-1 를 뽑는 실험. per-pair·NUM_IDS=100 같은 비표준 세팅이 아니라,
  CLIP-ReID·OSNet-AIN 논문들이 보고하는 그 표준 Duke<->Market 세팅이다.

[한 번 돌리면 나오는 것]
  방향별(Duke->Market, Market->Duke)로:
    - baseline                  : 보정 없는 표준 수치 (= 논문 비교 베이스)
    - CP-global                 : 전체 평균 1개 차감 (= 단순 도메인 centering)
    - CP-percam                 : 카메라별 평균 차감 (= 너의 제안)
    - (옵션) +rerank            : k-reciprocal 재정렬
    - N_PROTO x seed 안정성      : '소량 사진(5~50장)'으로도 되는지

[무엇을 증명하려는가]
  CP-percam > baseline          → "학습 0으로 cross-domain 정확도 향상" (주 주장)
  CP-percam > CP-global         → 효과가 '카메라별' 보정에서 옴 (open Q1)
  CP-global ~= CP-percam        → 그냥 도메인 centering 효과 → 기여 문장 재프레이밍
  N_PROTO 10~20 에서 안정        → "타겟 카메라 소량 사진" 주장 정당화

* 모델 로딩/전처리/특징추출 = 네 92번 코드 그대로. (CLIP-ReID make_model, load_param, extract)
* 메트릭 = torchreid eval_market1501 과 동일 규칙(같은 cam-같은 id 제거, pid=-1 제외).
* 특징은 방향별로 1번만 추출하고 캐싱 → 모든 변형(CP/rerank/N_PROTO)이 재사용.
"""

import os
import re
import glob
import datetime
import numpy as np

# =====================================================================
# 0. CONFIG
# =====================================================================
DIRECTIONS   = ["duke2market", "market2duke"]   # 둘 다 자동 실행. 하나만 보려면 줄여.

HOME         = os.path.expanduser("~")
PROJECT_DIR  = f"{HOME}/reid-gallery-expansion"
CKPT         = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR   = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
DUKE_DIR     = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
CLIP_REID    = "/home/ubuntu/CLIP-ReID"
LOG          = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

DEVICE       = "cuda"
BATCH        = 128
CACHE_DIR    = f"{PROJECT_DIR}/feat_cache_std"

RUN_RERANK   = False                            # full gallery 에선 메모리 큼. 기본 off.
RERANK_PARAM = dict(k1=20, k2=6, lambda_value=0.3)

NPROTO_SWEEP = [0, 5, 10, 20, 50]               # 0 = 카메라별 전체 사용
SEEDS        = [0, 1, 2, 3, 4]

# 방향 -> (평가 도메인 root, 학습 weight, 학습도메인 클래스수)
DIR_CFG = {
    # 모델은 학습 도메인, 평가는 타겟 도메인 (cross-domain)
    "duke2market": dict(eval_root=MARKET_DIR, weight=f"{CKPT}/clipreid_duke_nosie.pth",   num_class=702),
    "market2duke": dict(eval_root=DUKE_DIR,   weight=f"{CKPT}/clipreid_market_nosie.pth", num_class=751),
}

os.makedirs(CACHE_DIR, exist_ok=True)


def logline(m):
    with open(LOG, "a") as fp:
        fp.write(m + "\n")
    print(m)


# =====================================================================
# 1. 데이터 로딩  (Market/Duke 공통 파일명: 0001_c1s1_... / 0001_c2_...)
#    torchreid 규칙: pid=-1 제외, pid=0(배경/junk) 은 distractor 로 유지, camid 0-index
# =====================================================================
_PAT = re.compile(r"([-\d]+)_c(\d+)")

def parse_dir(dirpath):
    paths, pids, camids = [], [], []
    for p in sorted(glob.glob(os.path.join(dirpath, "*.jpg"))):
        m = _PAT.search(os.path.basename(p))
        if m is None:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:                  # distractor 제외 (표준)
            continue
        paths.append(p); pids.append(pid); camids.append(cam - 1)
    return paths, np.asarray(pids), np.asarray(camids)


def load_eval_data(eval_root):
    q = parse_dir(os.path.join(eval_root, "query"))
    g = parse_dir(os.path.join(eval_root, "bounding_box_test"))
    print(f"[data] eval_root={eval_root}\n"
          f"       query={len(q[0])}, gallery={len(g[0])}, "
          f"cams={sorted(set(g[2].tolist()))}")
    return q, g


# =====================================================================
# 2. 모델 + 특징추출  (네 92번 코드 그대로)
# =====================================================================
def build_model(weight_path, num_class):
    """CLIP-ReID ViT-B-16, stride16, SIE off. feature 추출용 래퍼 반환."""
    import sys, torch
    import torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, CLIP_REID)
    from config import cfg
    from model.make_model_clipreid import make_model

    cfg.MODEL.NAME = "ViT-B-16"; cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_CAMERA = False; cfg.MODEL.SIE_COE = 0.0
    cfg.MODEL.ID_LOSS_TYPE = "softmax"
    cfg.INPUT.SIZE_TRAIN = [256, 128]; cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]; cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = "market1501"
    cfg.TEST.WEIGHT = weight_path; cfg.TEST.NECK_FEAT = "before"

    try:
        b = make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        b = make_model(cfg, num_class=num_class, camera_num=6, view_num=1)
    b.load_param(weight_path)
    b = b.to(DEVICE).eval()

    class Backbone(nn.Module):
        def __init__(self, bb): super().__init__(); self.bb = bb
        @torch.no_grad()
        def forward(self, x):
            f = self.bb(x, cam_label=None)
            if isinstance(f, (list, tuple)):
                f = f[0] if isinstance(f[0], torch.Tensor) else f[-1]
            if f.dim() > 2:
                f = f.view(f.size(0), -1)
            return f

    preprocess = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return Backbone(b).eval(), preprocess


def extract_features(model, preprocess, paths, cache_path=None):
    """배치 추출 + L2 정규화 + 캐싱. 반환 np.float32 [N, D] (= 92번 feat()와 동일 결과)"""
    if cache_path and os.path.exists(cache_path):
        print(f"[cache] load {os.path.basename(cache_path)}")
        return np.load(cache_path)
    import torch
    from PIL import Image
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            bp = paths[i:i + BATCH]
            imgs = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in bp]).to(DEVICE)
            f = torch.nn.functional.normalize(model(imgs).float(), dim=1)
            feats.append(f.cpu().numpy())
            print(f"\r[extract] {min(i + BATCH, len(paths))}/{len(paths)}", end="")
    print()
    feats = np.concatenate(feats, 0).astype(np.float32)
    if cache_path:
        np.save(cache_path, feats)
    return feats


# =====================================================================
# 3. 카메라 프로토타입 보정 (CP)  ―  주 기여
#    leakage-safe: proto = 카메라별 평균(수백 ID 평균) → 개별 query id 누출 없음
# =====================================================================
def build_prototypes(g_feats, g_cams, mode, n_proto=0, seed=0):
    rng = np.random.RandomState(seed)
    if mode == "global":
        if n_proto > 0:
            idx = rng.choice(len(g_feats), min(n_proto, len(g_feats)), replace=False)
            return {"__global__": g_feats[idx].mean(0)}
        return {"__global__": g_feats.mean(0)}
    protos = {}
    for c in np.unique(g_cams):
        sel = np.where(g_cams == c)[0]
        if n_proto > 0 and len(sel) > n_proto:
            sel = rng.choice(sel, n_proto, replace=False)
        protos[int(c)] = g_feats[sel].mean(0)
    return protos


def apply_cp(feats, cams, protos, mode):
    out = feats.copy()
    if mode == "global":
        out = out - protos["__global__"][None, :]
    else:
        gmean = np.mean(list(protos.values()), axis=0)   # 못 본 cam fallback
        for i, c in enumerate(cams):
            out[i] = out[i] - protos.get(int(c), gmean)
    n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1e-12
    return (out / n).astype(np.float32)


# =====================================================================
# 4. 거리 + 표준 메트릭 (torchreid eval_market1501 규칙)
# =====================================================================
def cosine_distmat(qf, gf):
    return (1.0 - qf @ gf.T).astype(np.float32)

def euclidean_distmat(a, b):
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (a @ b.T))).astype(np.float32)

def eval_market(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    all_cmc, all_AP, valid = [], [], 0
    for qi in range(num_q):
        order = indices[qi]
        remove = (g_pids[order] == q_pids[qi]) & (g_cams[order] == q_cams[qi])
        keep = ~remove
        raw = matches[qi][keep]
        if not raw.any():
            continue
        cmc = raw.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank]); valid += 1
        num_rel = raw.sum()
        tmp = raw.cumsum() / (np.arange(len(raw)) + 1.0)
        all_AP.append((tmp * raw).sum() / num_rel)
    cmc = np.asarray(all_cmc).sum(0) / valid
    return cmc, float(np.mean(all_AP))


# =====================================================================
# 5. k-reciprocal re-ranking (Zhong et al. CVPR'17)
# =====================================================================
def re_ranking(q_g, q_q, g_g, k1=20, k2=6, lambda_value=0.3):
    original_dist = np.concatenate(
        [np.concatenate([q_q, q_g], axis=1),
         np.concatenate([q_g.T, g_g], axis=1)], axis=0).astype(np.float32)
    original_dist = np.power(original_dist, 2)
    original_dist /= np.max(original_dist, axis=0)
    original_dist = original_dist.T
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    query_num, all_num = q_g.shape[0], original_dist.shape[0]
    for i in range(all_num):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        krip = fwd[np.where(bwd == i)[0]]
        krip_exp = krip
        for c in krip:
            cf = initial_rank[c, :int(round(k1 / 2.)) + 1]
            cb = initial_rank[cf, :int(round(k1 / 2.)) + 1]
            ckr = cf[np.where(cb == c)[0]]
            if len(np.intersect1d(ckr, krip)) > 2. / 3 * len(ckr):
                krip_exp = np.append(krip_exp, ckr)
        krip_exp = np.unique(krip_exp)
        w = np.exp(-original_dist[i, krip_exp])
        V[i, krip_exp] = w / w.sum()
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        Vq = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            Vq[i] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = Vq
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jacc = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        tmp = np.zeros((1, all_num), dtype=np.float32)
        nz = np.where(V[i, :] != 0)[0]
        imgs = [invIndex[ind] for ind in nz]
        for j in range(len(nz)):
            tmp[0, imgs[j]] += np.minimum(V[i, nz[j]], V[imgs[j], nz[j]])
        jacc[i] = 1 - tmp / (2. - tmp)
    final = jacc * (1 - lambda_value) + original_dist * lambda_value
    return final[:query_num, query_num:]


# =====================================================================
# 6. 실행
# =====================================================================
def report(tag, cmc, mAP, base=None):
    r1, r5, r10 = cmc[0] * 100, cmc[4] * 100, cmc[9] * 100
    s = f"{tag:<20} R1={r1:5.1f}  R5={r5:5.1f}  R10={r10:5.1f}  mAP={mAP*100:5.1f}"
    if base is not None:
        s += f"   (R1 {r1-base[0]:+.1f}, mAP {mAP*100-base[1]:+.1f})"
    logline(s)
    return (r1, mAP * 100)


def run_direction(direction):
    cfg = DIR_CFG[direction]
    logline("\n" + "=" * 78)
    logline(f"## [{datetime.date.today()}] script93 표준평가  방향={direction}")
    logline(f"   weight={os.path.basename(cfg['weight'])}  eval={cfg['eval_root'].split('/')[-1]}")
    logline("=" * 78)

    (q_paths, q_pids, q_cams), (g_paths, g_pids, g_cams) = load_eval_data(cfg["eval_root"])
    model, preprocess = build_model(cfg["weight"], cfg["num_class"])
    qf = extract_features(model, preprocess, q_paths, f"{CACHE_DIR}/{direction}_query.npy")
    gf = extract_features(model, preprocess, g_paths, f"{CACHE_DIR}/{direction}_gallery.npy")

    # --- baseline ---
    cmc, mAP = eval_market(cosine_distmat(qf, gf), q_pids, g_pids, q_cams, g_cams)
    base = report("baseline", cmc, mAP)

    # --- CP global / percam ---
    for mode in ["global", "percam"]:
        protos = build_prototypes(gf, g_cams, mode, n_proto=0)
        qf_c = apply_cp(qf, q_cams, protos, mode)
        gf_c = apply_cp(gf, g_cams, protos, mode)
        cmc, mAP = eval_market(cosine_distmat(qf_c, gf_c), q_pids, g_pids, q_cams, g_cams)
        report(f"CP-{mode}", cmc, mAP, base)
        if RUN_RERANK:
            d = re_ranking(euclidean_distmat(qf_c, gf_c),
                           euclidean_distmat(qf_c, qf_c),
                           euclidean_distmat(gf_c, gf_c), **RERANK_PARAM)
            cmc, mAP = eval_market(d, q_pids, g_pids, q_cams, g_cams)
            report(f"CP-{mode}+rerank", cmc, mAP, base)

    # --- N_PROTO x seed 안정성 (percam) ---
    logline("\n  [N_PROTO x seed 안정성 / CP-percam]")
    logline(f"  {'N_PROTO':<8}{'R1(mean±std)':<18}{'mAP(mean±std)':<18}")
    for n in NPROTO_SWEEP:
        r1s, maps = [], []
        for sd in SEEDS:
            protos = build_prototypes(gf, g_cams, "percam", n_proto=n, seed=sd)
            qf_c = apply_cp(qf, q_cams, protos, "percam")
            gf_c = apply_cp(gf, g_cams, protos, "percam")
            cmc, mAP = eval_market(cosine_distmat(qf_c, gf_c), q_pids, g_pids, q_cams, g_cams)
            r1s.append(cmc[0] * 100); maps.append(mAP * 100)
            if n == 0:
                break
        lbl = "all" if n == 0 else str(n)
        logline(f"  {lbl:<8}{np.mean(r1s):5.1f}±{np.std(r1s):3.1f}      "
                f"{np.mean(maps):5.1f}±{np.std(maps):3.1f}")


def main():
    for d in DIRECTIONS:
        run_direction(d)
        import torch; torch.cuda.empty_cache()
    logline("\n해석: CP-percam>baseline → 학습0 cross-domain 향상 / "
            "CP-percam>CP-global → 카메라별 보정이 핵심 / "
            "N_PROTO 10~20 안정 → 소량사진 주장 OK")


if __name__ == "__main__":
    main()