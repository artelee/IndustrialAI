"""
74_hardcase_nosie.py

핵심 질문: "baseline이 틀린 query만" 골라서 생성 보강하면 회복되나?
- 전체 적용은 맞힌 것까지 건드려 효과 희석됨 (73 결과)
- baseline 실패 케이스에서만 생성이 도움 되는지 = 진짜 효과

SIE 없는 weight 사용 (Training-Free 일관)

측정:
  1. baseline 틀린 query 집합 H 추출
  2. H에서 Expand/Rerank가 몇 개 회복하나
  3. baseline 맞은 query 집합 C에서 몇 개 망가지나 (부작용)
  4. 순효과 = 회복 - 손상

추가: gen_sim, c1_sim 분포도 hardcase 따로 확인
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T

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

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

def load_split(gallery_dir, query_dir, target_cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'):
            continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        qby[pid][cam].append(f)
    cvi = {}
    for tc in target_cams:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]:
                ids.append(pid)
            if len(ids) >= NUM_IDS:
                break
        cvi[tc] = ids
    return gby, qby, cvi

def load_nosie(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_COE = 0.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path
    cfg.TEST.NECK_FEAT = 'before'
    try:
        m = make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

@torch.no_grad()
def feat(model, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    try:
        f = model(t)
    except TypeError:
        f = model(t, cam_label=None)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def eval_hardcase(model, gby, qby, valid_ids, target_cam, gen_dir):
    c1f, gf, qf = [], [], []
    kept = []
    for pid in valid_ids:
        gen_path = f"{gen_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        if not os.path.exists(gen_path):
            continue
        c1f.append(feat(model, sorted(gby[pid]["c1"])[0]))
        gf.append(feat(model, gen_path))
        qf.append(feat(model, sorted(qby[pid][target_cam])[0]))
        kept.append(pid)
    if not kept:
        return None
    c1f = np.array(c1f); gf = np.array(gf); qf = np.array(qf)
    N = len(kept)

    base_sims = qf @ c1f.T
    base_pred = base_sims.argmax(axis=1)
    base_correct_mask = (base_pred == np.arange(N))   # True=baseline 맞음

    hard_idx = np.where(~base_correct_mask)[0]   # baseline 틀린 것
    easy_idx = np.where(base_correct_mask)[0]    # baseline 맞은 것

    def predict_expand(i):
        s = np.maximum(qf[i] @ c1f.T, qf[i] @ gf.T)
        return s.argmax() == i
    def predict_rerank(i):
        s1 = base_sims[i]
        topk = np.argsort(-s1)[:TOP_K]
        s2 = qf[i] @ gf[topk].T
        final = ALPHA * s1[topk] + (1-ALPHA) * s2
        return topk[final.argmax()] == i

    # hardcase에서 회복 (틀린 것 → 맞음)
    exp_recover = sum(1 for i in hard_idx if predict_expand(i))
    rer_recover = sum(1 for i in hard_idx if predict_rerank(i))
    # easycase에서 손상 (맞은 것 → 틀림)
    exp_break = sum(1 for i in easy_idx if not predict_expand(i))
    rer_break = sum(1 for i in easy_idx if not predict_rerank(i))

    # hardcase의 생성 품질
    if len(hard_idx) > 0:
        hard_gen_sim = np.mean([qf[i] @ gf[i] for i in hard_idx])
        hard_c1_sim = np.mean([qf[i] @ c1f[i] for i in hard_idx])
    else:
        hard_gen_sim = hard_c1_sim = 0.0

    return {
        "N": N, "n_hard": len(hard_idx), "n_easy": len(easy_idx),
        "exp_recover": exp_recover, "rer_recover": rer_recover,
        "exp_break": exp_break, "rer_break": rer_break,
        "hard_c1_sim": hard_c1_sim, "hard_gen_sim": hard_gen_sim,
    }

def run(name, weight, dataset, num_class, camnum, eg, eq, cams, gen_dir):
    print("\n" + "=" * 90)
    print(f"방향: {name}  (SIE 없는 weight, hardcase 분석)")
    print("=" * 90)
    if not os.path.exists(weight):
        print(f"⚠️ weight 없음: {weight}"); return
    gby, qby, cvi = load_split(eg, eq, cams)
    model = load_nosie(weight, dataset, num_class, camnum)
    print("\n평가 중...")
    results = {}
    for tc in cams:
        r = eval_hardcase(model, gby, qby, cvi[tc], tc, gen_dir)
        if r:
            results[tc] = r

    print("\n" + "-" * 90)
    print(f"[{name}]  Expand/Rerank: 회복(틀린→맞음) vs 손상(맞은→틀림)")
    print(f"{'Pair':<7}{'N':<5}{'hard':<6}{'Exp:회복/손상':<16}{'Rer:회복/손상':<16}"
          f"{'hardC1':<9}{'hardGen':<9}")
    print("-" * 90)
    tot = defaultdict(int)
    for tc, r in results.items():
        print(f"c1→{tc:<4}{r['N']:<5}{r['n_hard']:<6}"
              f"{r['exp_recover']:>3}/{r['exp_break']:<11}"
              f"{r['rer_recover']:>3}/{r['rer_break']:<11}"
              f"{r['hard_c1_sim']:<9.3f}{r['hard_gen_sim']:<9.3f}")
        for k in ['n_hard','exp_recover','exp_break','rer_recover','rer_break']:
            tot[k] += r[k]
    print("-" * 90)
    print(f"합계   hard={tot['n_hard']}  "
          f"Expand 회복 {tot['exp_recover']} / 손상 {tot['exp_break']} "
          f"= 순 {tot['exp_recover']-tot['exp_break']:+d}  |  "
          f"Rerank 회복 {tot['rer_recover']} / 손상 {tot['rer_break']} "
          f"= 순 {tot['rer_recover']-tot['rer_break']:+d}")
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가",
        f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8,
        MARKET_GALLERY, MARKET_QUERY, ["c2","c3","c4","c5","c6"], MARKET_GEN)
    run("Market학습 → Duke평가",
        f"{CKPT}/clipreid_market_nosie.pth", "market1501", 751, 6,
        DUKE_GALLERY, DUKE_QUERY, ["c2","c3","c4","c5","c6","c8"], DUKE_GEN)
    print("\n" + "=" * 90)
    print("""
해석:
- 회복 > 손상 (순 +)  → 생성 보강이 진짜 효과. Selective 적용하면 향상 가능
- 회복 ≈ 손상         → 생성이 무작위. 도움도 해도 안 됨
- hardGen < hardC1 크게 → hardcase에서 생성도 query와 멀다 (애초에 어려운 케이스)
- hardGen ≈ hardC1     → 생성 품질 좋은데도 회복 안 되면 = 정보 중복
""")