"""
Confidence 측정 방식 3가지 비교
- A: Top-1 점수
- B: Top-1 - Top-2 차이
- C: Top-5 분산

각 측정 × 4 비율 = 12 셋업
Oracle (+2.48%p)과 비교
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
RATIOS = [0.1, 0.2, 0.3, 0.5]

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

# Feature 추출
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

def compute_confidences(s1):
    """3가지 confidence 측정"""
    N = s1.shape[0]
    top1_scores = s1.max(axis=1)
    
    # B: Top-1 - Top-2 차이
    sorted_s = np.sort(s1, axis=1)
    top1_top2_diff = sorted_s[:, -1] - sorted_s[:, -2]
    
    # C: Top-5 표준편차 (작을수록 비슷 = 자신 없음)
    # 그래서 신뢰도 = -std (높을수록 자신 있음)
    top5 = sorted_s[:, -5:]
    top5_std = top5.std(axis=1)
    
    return {
        'A_top1': top1_scores,           # 클수록 자신 있음
        'B_diff': top1_top2_diff,        # 클수록 자신 있음
        'C_std':  top5_std,              # 클수록 자신 있음 (분포 spread)
    }

def eval_selective(c1_feats, gen_feats, query_feats, confidence_scores, ratio):
    """하위 ratio*N개에만 Re-rank 적용"""
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    # Bottom-K 선택
    n_rerank = int(N * ratio)
    if n_rerank == 0: n_rerank = 1
    rerank_idx = np.argsort(confidence_scores)[:n_rerank]
    rerank_mask = np.zeros(N, dtype=bool)
    rerank_mask[rerank_idx] = True
    
    ours_top1 = baseline_top1.copy()
    for i in rerank_idx:
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        ours_top1[i] = topk_idx[final.argmax()]
    
    ours_correct = ours_top1 == np.arange(N)
    
    # 통계
    hard_in_rerank = (~baseline_correct & rerank_mask).sum()
    easy_in_rerank = (baseline_correct & rerank_mask).sum()
    
    return {
        'baseline_r1': baseline_correct.sum() / N,
        'ours_r1': ours_correct.sum() / N,
        'n_rerank': n_rerank,
        'hard_in_rerank': hard_in_rerank,
        'easy_in_rerank': easy_in_rerank,
    }

# 평가
print("\n" + "="*110)
print("Confidence 기반 선별 Re-ranking 비교")
print("Oracle 잠재력: +2.48%p")
print("="*110)

for measure in ['A_top1', 'B_diff', 'C_std']:
    label = {'A_top1':'A: Top-1 점수', 'B_diff':'B: Top-1-Top-2 차이', 'C_std':'C: Top-5 분산'}[measure]
    print(f"\n[{label}]")
    print(f"{'Ratio':<8}{'BL R1':<10}{'Ours R1':<10}{'향상':<10}{'Re-rank':<10}{'Hard in':<10}{'Easy in':<10}")
    print("-"*110)
    
    for ratio in RATIOS:
        total_n, total_b, total_o = 0, 0, 0
        total_rerank, total_hard, total_easy = 0, 0, 0
        for cam in TARGET_CAMS:
            c1_f, gen_f, q_f = all_feats[cam]
            s1 = q_f @ c1_f.T
            confs = compute_confidences(s1)
            r = eval_selective(c1_f, gen_f, q_f, confs[measure], ratio)
            n = len(q_f)
            total_n += n
            total_b += int(r['baseline_r1'] * n)
            total_o += int(r['ours_r1'] * n)
            total_rerank += r['n_rerank']
            total_hard += r['hard_in_rerank']
            total_easy += r['easy_in_rerank']
        
        bl = total_b / total_n * 100
        ours = total_o / total_n * 100
        gain = ours - bl
        mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
        precision = total_hard / max(total_rerank, 1) * 100  # Re-rank한 것 중 hard 비율
        print(f"{ratio*100:.0f}%{'':<6}{bl:<10.2f}{ours:<10.2f}{gain:+.2f} {mark:<5}"
              f"{total_rerank:<10}{total_hard:<10}{total_easy:<10}({precision:.0f}%)")

# Oracle 결과도 같이
print("\n" + "="*110)
print("Oracle (정답 알고): Baseline 60.93%, Oracle 63.41%, 향상 +2.48%p")
print("="*110)
print("→ 각 confidence 방식 vs Oracle 잠재력 비교")
