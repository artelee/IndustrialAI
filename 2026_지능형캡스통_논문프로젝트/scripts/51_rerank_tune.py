"""
Re-ranking 파라미터 탐색
Top-K와 alpha 다양화 → 최적 조합 찾기
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_BASE = f"{PROJECT_DIR}/outputs/c1base_gen_all"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
NUM_IDS = 100
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]

# 탐색
TOP_KS = [3, 5, 10, 20]
ALPHAS = [0.3, 0.5, 0.7]

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1','0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

cam_valid_ids = {}
for cam in TARGET_CAMS:
    ids = []
    for pid in sorted(gallery_by_id.keys()):
        if "c1" in gallery_by_id[pid] and cam in query_by_id[pid]:
            gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
            if os.path.exists(gen_path):
                ids.append(pid)
        if len(ids) >= NUM_IDS: break
    cam_valid_ids[cam] = ids

# Cross-Domain 모델 (효과 있던 것만)
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'dukemtmcreid'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

print("CLIP-ReID Duke 학습 weight 로드 (Cross-Domain)...")
model = make_model(cfg, num_class=702, camera_num=8, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 모든 feature 한 번만 추출
print("\nFeature 추출...")
all_feats = {}
for cam in TARGET_CAMS:
    print(f"  {cam}...")
    c1_f, gen_f, q_f = [], [], []
    for pid in tqdm(cam_valid_ids[cam], leave=False):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
        q_path = sorted(query_by_id[pid][cam])[0]
        c1_f.append(feat(c1_path, cam_to_id["c1"]))
        gen_f.append(feat(gen_path, cam_to_id[cam]))
        q_f.append(feat(q_path, cam_to_id[cam]))
    all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))

def rerank_eval(c1_feats, gen_feats, query_feats, top_k, alpha):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    correct = 0
    for i in range(N):
        topk_idx = np.argsort(-s1[i])[:top_k]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = alpha * s1_topk + (1 - alpha) * s2
        if topk_idx[final.argmax()] == i:
            correct += 1
    return correct / N

baseline_r1 = {}
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    sims = q_f @ c1_f.T
    baseline_r1[cam] = sum(1 for i in range(len(q_f)) if sims[i].argmax() == i) / len(q_f)

# 모든 파라미터 조합
print("\n" + "="*100)
print("파라미터 탐색 (Cross-Domain)")
print("="*100)

best = {"score": 0, "top_k": 0, "alpha": 0, "avg_gain": -100}

for top_k in TOP_KS:
    for alpha in ALPHAS:
        results = {}
        total_gain = 0
        for cam in TARGET_CAMS:
            c1_f, gen_f, q_f = all_feats[cam]
            r1 = rerank_eval(c1_f, gen_f, q_f, top_k, alpha)
            gain = (r1 - baseline_r1[cam]) * 100
            results[cam] = (r1, gain)
            total_gain += gain
        avg_gain = total_gain / len(TARGET_CAMS)
        
        wins = sum(1 for _, g in results.values() if g > 0)
        line = f"Top-K={top_k:<3} α={alpha} | avg gain {avg_gain:+.2f}%p | ✅{wins}/5 | "
        line += " ".join(f"{cam}:{g:+.1f}" for cam, (_, g) in results.items())
        print(line)
        
        if avg_gain > best["avg_gain"]:
            best = {"score": sum(r for r, _ in results.values()),
                    "top_k": top_k, "alpha": alpha, "avg_gain": avg_gain,
                    "results": results}

print("\n" + "="*100)
print(f"최적 조합: Top-K={best['top_k']}, alpha={best['alpha']}")
print(f"평균 향상: {best['avg_gain']:+.2f}%p")
print("="*100)
print(f"\n{'Pair':<10}{'Baseline':<12}{'Ours':<12}{'향상':<10}")
print("-"*50)
for cam, (r1, gain) in best["results"].items():
    mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
    print(f"c1→{cam:<7}{baseline_r1[cam]*100:<12.2f}{r1*100:<12.2f}{gain:+.2f} {mark}")
