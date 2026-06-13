"""
76_proto_ablation.py

검증 ① 기여 검증: 효과가 "카메라별 프로토타입" 때문인가, 그냥 "전체 평균 차감" 때문인가?
   - none      : 보정 없음 (baseline)
   - global    : 전체(모든 카메라 합친) 평균 1개 차감  ← 도메인 centering
   - percam    : 카메라별 프로토타입 차감              ← 네 아이디어
   percam > global 이어야 "카메라별로 하는 게 의미있다" 입증

검증 ② 안정성: N_PROTO ∈ {5,10,20,50} × seed 5개 → 평균±표준편차
   - 무작위 샘플 흔들림에 강건한지
   - 몇 장이면 충분한지

SIE 없는 weight. leakage 방지 (프로토타입 샘플은 평가 query에서 제외).

주의: 시드/N_PROTO 조합이 많아 c1 feature는 캐싱해 재사용. 페어당 query/gen feature도 1회만 추출.
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import random

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
DUKE_GALLERY = f"{DUKE_DIR}/bounding_box_test"
DUKE_QUERY = f"{DUKE_DIR}/query"
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"
DUKE_GEN = f"{PROJECT_DIR}/outputs/duke_c1base_gen"

device = "cuda"
NUM_IDS = 100
TOP_K = 10
ALPHA = 0.7
N_PROTO_LIST = [5, 10, 20, 50]
SEEDS = [0, 1, 2, 3, 4]

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

def setup(gallery_dir, query_dir, target_cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        qby[pid][cam].append(f)
    cam_query_pool = defaultdict(list)
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        cam_query_pool[cam].append(f)
    cvi = {}
    for tc in target_cams:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]:
                ids.append(pid)
            if len(ids) >= NUM_IDS: break
        cvi[tc] = ids
    return gby, qby, cvi, cam_query_pool

def load_nosie(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0
    cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=dataset_name
    cfg.TEST.WEIGHT=weight_path; cfg.TEST.NECK_FEAT='before'
    try:
        m=make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        m=make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([T.Resize([256,128]), T.ToTensor(),
                       T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

@torch.no_grad()
def feat_raw(model, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    return f.cpu().numpy().flatten()

def l2n(x): return x/(np.linalg.norm(x)+1e-9)

def r1_baseline(qf, gf, N):
    sims = qf @ gf.T
    return sum(1 for i in range(N) if sims[i].argmax()==i)/N

def r1_rerank(qf, c1f, genf, N):
    base = qf @ c1f.T; c=0
    for i in range(N):
        s1=base[i]; topk=np.argsort(-s1)[:TOP_K]
        s2=qf[i]@genf[topk].T
        final=ALPHA*s1[topk]+(1-ALPHA)*s2
        if topk[final.argmax()]==i: c+=1
    return c/N

def run(name, weight, dataset, num_class, camnum, eg, eq, cams, gen_dir):
    print("\n"+"="*92)
    print(f"방향: {name}")
    print("="*92)
    if not os.path.exists(weight):
        print(f"⚠️ weight 없음: {weight}"); return
    gby, qby, cvi, cqp = setup(eg, eq, cams)
    model = load_nosie(weight, dataset, num_class, camnum)

    # === raw feature 1회 추출 후 캐싱 ===
    print("feature 추출 중 (1회)...")
    # 페어별 c1/gen/query raw
    pair_data = {}
    for tc in cams:
        c1r, genr, qr, kept = [], [], [], []
        for pid in cvi[tc]:
            gp=f"{gen_dir}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            c1r.append(feat_raw(model, sorted(gby[pid]["c1"])[0]))
            genr.append(feat_raw(model, gp))
            qr.append(feat_raw(model, sorted(qby[pid][tc])[0]))
            kept.append(pid)
        pair_data[tc] = (np.array(c1r), np.array(genr), np.array(qr),
                         set(sorted(qby[pid][tc])[0] for pid in kept if tc in qby[pid]))

    # 카메라별 전체 query raw (프로토타입 샘플 풀) — 미리 다 뽑아두면 무거우니
    # 필요한 카메라만, 최대 N(=max(N_PROTO_LIST)+여유) 만큼만 캐싱
    proto_pool_feats = {}
    POOL_CAP = max(N_PROTO_LIST) + 80
    for cam in set(cams) | {"c1"}:
        files = [f for f in cqp.get(cam, [])]
        random.Random(999).shuffle(files)
        files = files[:POOL_CAP]
        proto_pool_feats[cam] = [(f, feat_raw(model, f)) for f in files]

    # === 검증 ① 기여 검증 (N_PROTO=10, seed=0 고정) ===
    print("\n" + "-"*92)
    print(f"[검증①] 보정 방식 비교 (none / global / percam), N_PROTO=10")
    print(f"{'Pair':<8}{'none':<10}{'global차감':<14}{'percam차감':<14}")
    print("-"*92)
    # global mean = 모든 카메라 query 합친 평균
    all_feats = []
    for cam in cams:
        all_feats += [v for (_,v) in proto_pool_feats[cam][:10]]
    global_mean = np.mean(all_feats, axis=0)
    # c1 proto
    c1_proto10 = np.mean([v for (_,v) in proto_pool_feats["c1"][:10]], axis=0)

    s_none=s_glob=s_cam=0; cnt=0
    for tc in cams:
        c1r, genr, qr, eval_q = pair_data[tc]
        N=len(qr)
        if N==0: continue
        # cam proto (eval query 제외)
        cand=[v for (f,v) in proto_pool_feats[tc] if f not in eval_q][:10]
        cam_proto=np.mean(cand, axis=0)
        # none
        qf=np.array([l2n(x) for x in qr]); c1f=np.array([l2n(x) for x in c1r])
        none=r1_baseline(qf,c1f,N)*100
        # global
        qf=np.array([l2n(x-global_mean) for x in qr])
        c1f=np.array([l2n(x-global_mean) for x in c1r])
        glob=r1_baseline(qf,c1f,N)*100
        # percam
        qf=np.array([l2n(x-cam_proto) for x in qr])
        c1f=np.array([l2n(x-c1_proto10) for x in c1r])
        cam=r1_baseline(qf,c1f,N)*100
        print(f"c1→{tc:<5}{none:<10.2f}{glob:<6.2f}({glob-none:+.1f})    {cam:<6.2f}({cam-none:+.1f})")
        s_none+=none; s_glob+=glob; s_cam+=cam; cnt+=1
    if cnt:
        print("-"*92)
        print(f"{'평균':<8}{s_none/cnt:<10.2f}{s_glob/cnt:<6.2f}({(s_glob-s_none)/cnt:+.1f})    "
              f"{s_cam/cnt:<6.2f}({(s_cam-s_none)/cnt:+.1f})")
        print(f"\n→ percam({s_cam/cnt:.1f}) > global({s_glob/cnt:.1f}) 이면 카메라별 보정의 고유 기여 입증")

    # === 검증 ② N_PROTO × seed 안정성 (percam, baseline 매칭) ===
    print("\n" + "-"*92)
    print(f"[검증②] N_PROTO × seed 안정성 (percam 차감, 전 페어 평균 R1)")
    print(f"{'N_PROTO':<10}{'평균R1':<12}{'표준편차':<12}{'min~max':<16}")
    print("-"*92)
    for npro in N_PROTO_LIST:
        seed_means=[]
        for sd in SEEDS:
            rng=random.Random(sd)
            # c1 proto
            c1files=[f for (f,_) in proto_pool_feats["c1"]]
            c1idx=list(range(len(proto_pool_feats["c1"]))); rng.shuffle(c1idx)
            c1_proto=np.mean([proto_pool_feats["c1"][i][1] for i in c1idx[:npro]], axis=0)
            pair_r1=[]
            for tc in cams:
                c1r, genr, qr, eval_q = pair_data[tc]
                N=len(qr)
                if N==0: continue
                cand=[(f,v) for (f,v) in proto_pool_feats[tc] if f not in eval_q]
                idx=list(range(len(cand))); rng.shuffle(idx)
                if len(idx)<npro: continue
                cam_proto=np.mean([cand[i][1] for i in idx[:npro]], axis=0)
                qf=np.array([l2n(x-cam_proto) for x in qr])
                c1f=np.array([l2n(x-c1_proto) for x in c1r])
                pair_r1.append(r1_baseline(qf,c1f,N)*100)
            if pair_r1: seed_means.append(np.mean(pair_r1))
        if seed_means:
            arr=np.array(seed_means)
            print(f"{npro:<10}{arr.mean():<12.2f}{arr.std():<12.2f}"
                  f"{arr.min():.1f}~{arr.max():.1f}")
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가",
        f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8,
        MARKET_GALLERY, MARKET_QUERY, ["c2","c3","c4","c5","c6"], MARKET_GEN)
    run("Market학습 → Duke평가",
        f"{CKPT}/clipreid_market_nosie.pth", "market1501", 751, 6,
        DUKE_GALLERY, DUKE_QUERY, ["c2","c3","c4","c5","c6","c8"], DUKE_GEN)
    print("\n"+"="*92)
    print("""
해석:
[검증①] percam > global → "카메라별로 하는 것"의 고유 가치 (네 기여 핵심)
        percam ≈ global → 효과는 단순 도메인 centering 때문 → 기여 재프레이밍 필요
[검증②] 표준편차 작고(<1.5) N_PROTO 작아도 안정 → "소량 샘플로 충분" 주장 가능
        N_PROTO 클수록 좋아지면 → trade-off 명시 필요
""")