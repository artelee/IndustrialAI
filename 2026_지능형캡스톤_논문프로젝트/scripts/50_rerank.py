"""
Re-ranking + 생성 (Inference 시 생성 활용)

Step 1: c1 갤러리로 Top-K 후보
Step 2: 후보의 생성 c?로 정밀 비교
Step 3: Re-rank

비교: Baseline vs Ours
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
TOP_K = 5  # 1단계 후보 수
ALPHA = 0.5  # 1단계와 2단계 가중치

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

print("데이터 로드...")
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
    print(f"  c1 → {cam}: {len(ids)}명")

# CLIP-ReID 로드
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
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

@torch.no_grad()
def feat(model, path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def eval_rerank(model, target_cam, valid_ids, top_k=TOP_K, alpha=ALPHA):
    """Re-ranking 평가"""
    # 갤러리: c1 실제 + 생성 c?
    c1_feats = []     # Step 1용
    gen_feats = []    # Step 2용
    query_feats = []
    
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(query_by_id[pid][target_cam])[0]
        c1_feats.append(feat(model, c1_path, cam_to_id["c1"]))
        gen_feats.append(feat(model, gen_path, cam_to_id[target_cam]))
        query_feats.append(feat(model, q_path, cam_to_id[target_cam]))
    
    c1_feats = np.array(c1_feats)
    gen_feats = np.array(gen_feats)
    query_feats = np.array(query_feats)
    
    N = len(valid_ids)
    
    # Baseline: c1 갤러리만
    baseline_sims = query_feats @ c1_feats.T
    baseline_correct = sum(1 for i in range(N) if baseline_sims[i].argmax() == i)
    
    # Ours: Re-ranking
    rerank_correct = 0
    for i in range(N):
        # Step 1: c1 갤러리로 Top-K 후보
        s1_scores = baseline_sims[i]  # 100명 점수
        topk_idx = np.argsort(-s1_scores)[:top_k]  # Top-K 인덱스
        
        # Step 2: Top-K의 생성 c?와 query 비교
        topk_gen_feats = gen_feats[topk_idx]
        s2_scores = query_feats[i] @ topk_gen_feats.T  # Top-K 내 점수
        
        # Re-rank: 두 점수 결합
        s1_topk = s1_scores[topk_idx]
        final_scores = alpha * s1_topk + (1 - alpha) * s2_scores
        
        # 최종 Top-1
        best_in_topk = topk_idx[final_scores.argmax()]
        if best_in_topk == i:
            rerank_correct += 1
    
    return baseline_correct / N, rerank_correct / N

# Same-Domain
print("\n" + "="*70)
print("Re-ranking 평가 (Same-Domain, Market 학습)")
print("="*70)
print(f"Top-K={TOP_K}, alpha={ALPHA}")

model = load_clipreid(
    f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth",
    "market1501", 751, 6
)
same_results = {}
for cam in TARGET_CAMS:
    print(f"\nc1 → {cam}")
    r1_b, r1_o = eval_rerank(model, cam, cam_valid_ids[cam])
    same_results[cam] = (r1_b, r1_o)
    diff = (r1_o - r1_b) * 100
    mark = "✅" if r1_o > r1_b else ("=" if r1_o == r1_b else "❌")
    print(f"  Baseline: {r1_b*100:.2f}%, Ours (rerank): {r1_o*100:.2f}% ({diff:+.2f} {mark})")

# Cross-Domain
print("\n" + "="*70)
print("Re-ranking 평가 (Cross-Domain, Duke 학습)")
print("="*70)
del model
torch.cuda.empty_cache()
model = load_clipreid(
    f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth",
    "dukemtmcreid", 702, 8
)
cross_results = {}
for cam in TARGET_CAMS:
    print(f"\nc1 → {cam}")
    r1_b, r1_o = eval_rerank(model, cam, cam_valid_ids[cam])
    cross_results[cam] = (r1_b, r1_o)
    diff = (r1_o - r1_b) * 100
    mark = "✅" if r1_o > r1_b else ("=" if r1_o == r1_b else "❌")
    print(f"  Baseline: {r1_b*100:.2f}%, Ours (rerank): {r1_o*100:.2f}% ({diff:+.2f} {mark})")

# 종합
print("\n\n" + "="*80)
print(f"Re-ranking 종합 결과 (Top-{TOP_K}, alpha={ALPHA})")
print("="*80)
print("\n[Same-Domain]")
print(f"{'Pair':<10}{'Baseline':<15}{'Ours':<15}{'향상':<10}")
print("-"*80)
for cam, (rb, ro) in same_results.items():
    diff = (ro-rb)*100
    mark = "✅" if ro > rb else ("=" if ro == rb else "❌")
    print(f"c1→{cam:<7}{rb*100:<15.2f}{ro*100:<15.2f}{diff:+.2f} {mark}")

print("\n[Cross-Domain]")
print(f"{'Pair':<10}{'Baseline':<15}{'Ours':<15}{'향상':<10}")
print("-"*80)
for cam, (rb, ro) in cross_results.items():
    diff = (ro-rb)*100
    mark = "✅" if ro > rb else ("=" if ro == rb else "❌")
    print(f"c1→{cam:<7}{rb*100:<15.2f}{ro*100:<15.2f}{diff:+.2f} {mark}")
