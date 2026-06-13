"""
71_rerank_sieoff.py

수정 사항 (70 대비):
1. 단순 갤러리 확장(max) → Re-ranking 방식으로 변경 (이전 효과 본 방식)
   Step1: c1 갤러리로 Top-K 후보
   Step2: 후보의 생성 시점 이미지로 재비교 (s2)
   Step3: s = α·s1 + (1-α)·s2
2. SIE off 유지 (cam_label=None) — Training-Free 일관
3. 자세(시점) 중심 — 조명/색감 보정 제외 (데이터에 색감차 거의 없음 확인됨)

비교:
  Baseline   : c1 갤러리 직접 매칭 (Top-1)
  Ours-rerank: Top-K 좁힌 뒤 생성 시점으로 재정렬

양방향: Duke학습→Market, Market학습→Duke
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

MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"

DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
DUKE_GALLERY = f"{DUKE_DIR}/bounding_box_test"
DUKE_QUERY = f"{DUKE_DIR}/query"

MARKET_GEN_POSE = f"{PROJECT_DIR}/outputs/c1base_gen_all"
DUKE_GEN_POSE = f"{PROJECT_DIR}/outputs/duke_c1base_gen"

device = "cuda"
NUM_IDS = 100
TOP_K = 10        # 후보 수 (이전엔 5, 여기선 10도 시도해볼 가치)
ALPHAS = [0.5, 0.7]   # s1/s2 가중치 여러 개 비교

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

def load_clipreid(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [12, 12]
    cfg.MODEL.SIE_CAMERA = True
    cfg.MODEL.SIE_COE = 1.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path
    cfg.TEST.NECK_FEAT = 'before'
    m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.no_grad()
def feat(model, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    f = model(t, cam_label=None)   # SIE off
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def eval_rerank(model, gby, qby, valid_ids, target_cam, gen_pose_dir):
    """
    Re-ranking (이전 효과 본 방식):
      Step1: query vs c1 갤러리 → s1, Top-K 후보
      Step2: query vs 후보의 생성 시점 → s2
      Step3: s = α·s1 + (1-α)·s2 → 재정렬
    """
    c1_feats, gen_feats, query_feats = [], [], []
    kept = []
    for pid in valid_ids:
        c1_path = sorted(gby[pid]["c1"])[0]
        gen_path = f"{gen_pose_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(qby[pid][target_cam])[0]
        if not os.path.exists(gen_path):
            continue
        c1_feats.append(feat(model, c1_path))
        gen_feats.append(feat(model, gen_path))
        query_feats.append(feat(model, q_path))
        kept.append(pid)
    if not kept:
        return None
    c1f = np.array(c1_feats); gf = np.array(gen_feats); qf = np.array(query_feats)
    N = len(kept)

    base_sims = qf @ c1f.T
    baseline_r1 = sum(1 for i in range(N) if base_sims[i].argmax() == i) / N

    res = {"N": N, "baseline": baseline_r1}
    for alpha in ALPHAS:
        correct = 0
        for i in range(N):
            s1 = base_sims[i]
            topk = np.argsort(-s1)[:TOP_K]
            s2 = qf[i] @ gf[topk].T
            final = alpha * s1[topk] + (1 - alpha) * s2
            best = topk[final.argmax()]
            if best == i:
                correct += 1
        res[f"a{alpha}"] = correct / N
    return res

def run(name, train_weight, train_dataset, num_class, train_camnum,
        eval_gallery, eval_query, target_cams, gen_pose_dir):
    print("\n" + "=" * 80)
    print(f"방향: {name}  (SIE OFF, Re-ranking Top-{TOP_K})")
    print("=" * 80)
    gby, qby, cvi = load_split(eval_gallery, eval_query, target_cams)
    model = load_clipreid(train_weight, train_dataset, num_class, train_camnum)
    print("\n평가 중...")
    results = {}
    for tc in target_cams:
        r = eval_rerank(model, gby, qby, cvi[tc], tc, gen_pose_dir)
        if r:
            results[tc] = r
    # 출력
    head = f"{'Pair':<8}{'N':<6}{'Baseline':<12}"
    for a in ALPHAS:
        head += f"{'rerank a='+str(a):<16}"
    print("\n" + "-" * 80)
    print(f"[{name}]")
    print(head)
    print("-" * 80)
    sums = {"baseline": 0}
    for a in ALPHAS:
        sums[f"a{a}"] = 0
    cnt = 0
    for tc, r in results.items():
        line = f"c1→{tc:<5}{r['N']:<6}{r['baseline']*100:<12.2f}"
        for a in ALPHAS:
            v = r[f"a{a}"] * 100
            d = v - r["baseline"] * 100
            line += f"{v:<6.2f}({d:+.1f})    "
            sums[f"a{a}"] += v
        sums["baseline"] += r["baseline"] * 100
        cnt += 1
        print(line)
    if cnt:
        print("-" * 80)
        line = f"{'평균':<8}{'':<6}{sums['baseline']/cnt:<12.2f}"
        for a in ALPHAS:
            v = sums[f"a{a}"] / cnt
            d = v - sums["baseline"] / cnt
            line += f"{v:<6.2f}({d:+.1f})    "
        print(line)
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가",
        f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth",
        "dukemtmcreid", 702, 8,
        MARKET_GALLERY, MARKET_QUERY,
        ["c2", "c3", "c4", "c5", "c6"], MARKET_GEN_POSE)

    run("Market학습 → Duke평가",
        f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth",
        "market1501", 751, 6,
        DUKE_GALLERY, DUKE_QUERY,
        ["c2", "c3", "c4", "c5", "c6", "c8"], DUKE_GEN_POSE)

    print("\n" + "=" * 80)
    print("""
해석:
- rerank > baseline  → Top-K 좁힌 뒤 생성 시점 재정렬이 효과 (자세 보강)
- a=0.7이 보통 안정적 (s1 비중 큼, s2는 보조)
- 이전 세션 SIE on Re-ranking과 비교: SIE off에서도 효과 유지되는지 확인
""")