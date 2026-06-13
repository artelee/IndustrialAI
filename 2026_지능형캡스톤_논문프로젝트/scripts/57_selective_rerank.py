"""
Selective Re-ranking
Baseline confidence 낮은 케이스에만 Re-rank 적용
Easy case는 건드리지 않음
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
GEN_BASE = f"{PROJECT_DIR}/outputs/duke_c1base_gen"
DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
GALLERY_DIR = f"{DUKE_DIR}/bounding_box_test"
QUERY_DIR = f"{DUKE_DIR}/query"

device = "cuda"
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6", "c8"]
TOP_K = 5
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

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
    cam_valid_ids[cam] = ids

cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'market1501'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

print("CLIP-ReID Market 학습 로드...")
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5,'c7':5,'c8':5}

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 모든 feature 추출
print("\nFeature 추출...")
all_feats = {}
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = [], [], []
    for pid in tqdm(cam_valid_ids[cam], desc=cam, leave=False):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
        q_path = sorted(query_by_id[pid][cam])[0]
        c1_f.append(feat(c1_path, cam_to_id["c1"]))
        gen_f.append(feat(gen_path, cam_to_id[cam]))
        q_f.append(feat(q_path, cam_to_id[cam]))
    all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))

# === 셋업 A: Oracle (Baseline 틀린 것만 Re-rank) ===
# === 셋업 B: Confidence threshold (Top-1 점수 낮은 것만 Re-rank) ===

def eval_oracle(c1_feats, gen_feats, query_feats):
    """Oracle: Baseline 틀린 것만 Re-rank, 맞은 건 그대로"""
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    ours_correct = baseline_correct.copy()  # 맞은 건 그대로
    
    # 틀린 것만 Re-rank
    for i in range(N):
        if baseline_correct[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        ours_top1 = topk_idx[final.argmax()]
        ours_correct[i] = (ours_top1 == i)
    
    return baseline_correct.sum() / N, ours_correct.sum() / N

def eval_confidence(c1_feats, gen_feats, query_feats, threshold):
    """Confidence: Top-1 점수가 threshold 미만일 때만 Re-rank"""
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    top1_scores = s1.max(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    rerank_mask = top1_scores < threshold
    ours_top1 = baseline_top1.copy()
    
    for i in range(N):
        if not rerank_mask[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        ours_top1[i] = topk_idx[final.argmax()]
    
    ours_correct = ours_top1 == np.arange(N)
    return baseline_correct.sum() / N, ours_correct.sum() / N, rerank_mask.sum()

# Oracle 결과
print("\n" + "="*80)
print("Oracle (Baseline 틀린 것만 Re-rank) - 이상적 상한선")
print("="*80)
print(f"{'Pair':<10}{'N':<6}{'Baseline':<15}{'Oracle':<15}{'향상':<10}")
print("-"*80)

total_n, total_b, total_o = 0, 0, 0
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    r_b, r_o = eval_oracle(c1_f, gen_f, q_f)
    n = len(q_f)
    gain = (r_o - r_b) * 100
    mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
    print(f"c1→{cam:<7}{n:<6}{r_b*100:<15.2f}{r_o*100:<15.2f}{gain:+.2f} {mark}")
    total_n += n
    total_b += int(r_b * n)
    total_o += int(r_o * n)

avg_gain = (total_o - total_b) / total_n * 100
print("-"*80)
print(f"{'합계':<10}{total_n:<6}{total_b/total_n*100:<15.2f}{total_o/total_n*100:<15.2f}{avg_gain:+.2f}")

# Confidence threshold 다양화
print("\n" + "="*80)
print("Confidence-aware (실용적, threshold 다양화)")
print("="*80)

for threshold in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    print(f"\n[Threshold {threshold}]")
    print(f"{'Pair':<10}{'N':<6}{'Baseline':<12}{'Ours':<12}{'향상':<10}{'Re-rank수':<10}")
    print("-"*80)
    total_b, total_o = 0, 0
    for cam in TARGET_CAMS:
        c1_f, gen_f, q_f = all_feats[cam]
        r_b, r_o, n_rerank = eval_confidence(c1_f, gen_f, q_f, threshold)
        n = len(q_f)
        gain = (r_o - r_b) * 100
        mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
        print(f"c1→{cam:<7}{n:<6}{r_b*100:<12.2f}{r_o*100:<12.2f}{gain:+.2f} {mark:<5}{n_rerank}")
        total_b += int(r_b * n)
        total_o += int(r_o * n)
    print(f"{'합계':<10}{total_n:<6}{total_b/total_n*100:<12.2f}{total_o/total_n*100:<12.2f}"
          f"{(total_o-total_b)/total_n*100:+.2f}")
