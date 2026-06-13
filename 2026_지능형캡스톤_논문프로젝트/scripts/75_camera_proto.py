"""
75_camera_proto.py

핵심 아이디어 (사용자 제안):
새 환경 배포 시, 타겟 카메라에서 사진 몇 장(10장)만 얻어
"카메라 프로토타입"(feature 평균)을 계산하고,
매칭 시 query/gallery feature에서 이 프로토타입을 차감 → 카메라 편향 제거.
= 학습 없는 SIE 대체 (Training-Free, Cross-Domain 정당)

leakage 방지: 프로토타입 계산에 쓴 N장은 평가에서 제외.

비교:
  Baseline        : 보정 없음, c1 갤러리
  +CamProto       : 카메라 프로토타입 차감 후 매칭 (생성 X)
  +CamProto +Gen  : 프로토타입 차감 + 생성 갤러리 확장

SIE 없는 weight 사용.

N_PROTO = 10  (타겟 카메라 샘플 수)
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import random

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

random.seed(42); np.random.seed(42)

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
N_PROTO = 10        # 타겟 카메라 프로토타입용 샘플 수
TOP_K = 10
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

def load_clip_features_setup(gallery_dir, query_dir, target_cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        qby[pid][cam].append(f)
    # 카메라별 전체 query 풀 (프로토타입 샘플링용)
    cam_query_pool = defaultdict(list)
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        cam_query_pool[cam].append((pid, f))
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
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_COE = 0.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]; cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD = [0.5,0.5,0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path; cfg.TEST.NECK_FEAT = 'before'
    try:
        m = make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256,128]), T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

@torch.no_grad()
def feat_raw(model, path):
    """보정 전 raw feature (정규화 전)"""
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    try:
        f = model(t)
    except TypeError:
        f = model(t, cam_label=None)
    return f.cpu().numpy().flatten()

def l2n(x):
    return x / (np.linalg.norm(x) + 1e-9)

def build_camera_proto(model, cam_query_pool, target_cam, exclude_files):
    """
    타겟 카메라에서 N_PROTO장 무작위 추출 (평가 제외분),
    raw feature 평균 = 카메라 프로토타입.
    leakage 방지: exclude_files(평가에 쓸 query)는 제외.
    반환: (proto_vector, 사용한 파일 set)
    """
    pool = [(pid, f) for (pid, f) in cam_query_pool[target_cam]
            if f not in exclude_files]
    random.shuffle(pool)
    chosen = pool[:N_PROTO]
    feats = [feat_raw(model, f) for (_, f) in chosen]
    proto = np.mean(feats, axis=0)
    used = set(f for (_, f) in chosen)
    return proto, used

def evaluate(model, gby, qby, valid_ids, target_cam, gen_dir,
             cam_proto, c1_proto):
    """
    각 ID: c1 갤러리, 생성, query feature 추출.
    보정 = raw_feat - 해당 카메라 proto, 그 후 L2 정규화.
    """
    c1_raw, gen_raw, q_raw = [], [], []
    kept = []
    for pid in valid_ids:
        gen_path = f"{gen_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        if not os.path.exists(gen_path): continue
        c1_raw.append(feat_raw(model, sorted(gby[pid]["c1"])[0]))
        gen_raw.append(feat_raw(model, gen_path))
        q_raw.append(feat_raw(model, sorted(qby[pid][target_cam])[0]))
        kept.append(pid)
    if not kept: return None
    c1_raw = np.array(c1_raw); gen_raw = np.array(gen_raw); q_raw = np.array(q_raw)
    N = len(kept)

    def r1_baseline(qf, gf):
        sims = qf @ gf.T
        return sum(1 for i in range(N) if sims[i].argmax()==i)/N
    def r1_rerank(qf, c1f, genf):
        base = qf @ c1f.T; c=0
        for i in range(N):
            s1=base[i]; topk=np.argsort(-s1)[:TOP_K]
            s2=qf[i]@genf[topk].T
            final=ALPHA*s1[topk]+(1-ALPHA)*s2
            if topk[final.argmax()]==i: c+=1
        return c/N

    # --- 보정 없음 (baseline) ---
    qf0 = np.array([l2n(x) for x in q_raw])
    c1f0 = np.array([l2n(x) for x in c1_raw])
    genf0 = np.array([l2n(x) for x in gen_raw])
    base_no = r1_baseline(qf0, c1f0)
    rer_no = r1_rerank(qf0, c1f0, genf0)

    # --- 카메라 프로토타입 차감 후 ---
    # query는 target_cam proto, c1갤러리/생성은 c1_proto 차감
    # (생성은 c1 기반이지만 target 시점이므로 cam_proto가 맞을 수도 → 둘 다 시도 가능. 우선 c1_proto)
    qf1 = np.array([l2n(x - cam_proto) for x in q_raw])
    c1f1 = np.array([l2n(x - c1_proto) for x in c1_raw])
    genf1 = np.array([l2n(x - cam_proto) for x in gen_raw])  # 생성=target시점 → cam_proto
    base_cp = r1_baseline(qf1, c1f1)
    rer_cp = r1_rerank(qf1, c1f1, genf1)

    return {"N":N, "base_no":base_no, "rer_no":rer_no,
            "base_cp":base_cp, "rer_cp":rer_cp}

def run(name, weight, dataset, num_class, camnum, eg, eq, cams, gen_dir):
    print("\n"+"="*92)
    print(f"방향: {name}  (카메라 프로토타입 보정, N_PROTO={N_PROTO})")
    print("="*92)
    if not os.path.exists(weight):
        print(f"⚠️ weight 없음: {weight}"); return
    gby, qby, cvi, cqp = load_clip_features_setup(eg, eq, cams)
    model = load_nosie(weight, dataset, num_class, camnum)

    # c1 프로토타입 (소스 카메라, query 풀의 c1에서)
    print("\nc1 프로토타입 계산...")
    if len(cqp["c1"]) >= N_PROTO:
        c1_files = [f for (_,f) in cqp["c1"]]
        random.shuffle(c1_files)
        c1_proto = np.mean([feat_raw(model, f) for f in c1_files[:N_PROTO]], axis=0)
    else:
        # c1 query 부족 시 gallery c1에서
        c1g = []
        for pid in list(gby.keys())[:50]:
            if "c1" in gby[pid]: c1g.append(sorted(gby[pid]["c1"])[0])
        c1_proto = np.mean([feat_raw(model, f) for f in c1g[:N_PROTO]], axis=0)

    print("평가 중...")
    results = {}
    for tc in cams:
        # 평가에 쓸 query 파일 (leakage 제외용)
        eval_q_files = set(sorted(qby[pid][tc])[0] for pid in cvi[tc]
                           if tc in qby[pid])
        cam_proto, used = build_camera_proto(model, cqp, tc, eval_q_files)
        r = evaluate(model, gby, qby, cvi[tc], tc, gen_dir, cam_proto, c1_proto)
        if r: results[tc] = r

    print("\n"+"-"*92)
    print(f"[{name}]")
    print(f"{'Pair':<8}{'N':<6}{'Base(보정X)':<14}{'Base+CP':<14}"
          f"{'Rerank(보정X)':<16}{'Rerank+CP':<14}")
    print("-"*92)
    s = defaultdict(float); cnt=0
    for tc, r in results.items():
        bn,bc = r['base_no']*100, r['base_cp']*100
        rn,rc = r['rer_no']*100, r['rer_cp']*100
        print(f"c1→{tc:<5}{r['N']:<6}{bn:<14.2f}{bc:<6.2f}({bc-bn:+.1f})    "
              f"{rn:<16.2f}{rc:<6.2f}({rc-rn:+.1f})")
        s['bn']+=bn; s['bc']+=bc; s['rn']+=rn; s['rc']+=rc; cnt+=1
    if cnt:
        print("-"*92)
        print(f"{'평균':<8}{'':<6}{s['bn']/cnt:<14.2f}"
              f"{s['bc']/cnt:<6.2f}({(s['bc']-s['bn'])/cnt:+.1f})    "
              f"{s['rn']/cnt:<16.2f}{s['rc']/cnt:<6.2f}({(s['rc']-s['rn'])/cnt:+.1f})")
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
핵심 확인:
- Base+CP > Base       → 카메라 프로토타입 차감(=학습없는 SIE)이 효과!
                          → 네 아이디어 입증. Training-Free 카메라 보정 성립
- Rerank+CP > Rerank   → 보정 + 생성 결합 시너지
- 변화 없음            → 단순 평균차감으로는 부족 (whitening 등 더 정교한 보정 필요)

N_PROTO=10 = 새 환경에서 타겟 카메라 사진 10장만 (라벨 불필요, 학습 불필요)
""")