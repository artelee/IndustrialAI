
"""
표준 Market-1501 평가 + Stage-wise 매칭
mAP, Rank-1, Rank-5, Rank-10 비교

3가지 방식:
1. Baseline: 표준 갤러리
2. 단순 확장: 갤러리 + 생성
3. Stage-wise: query 카메라별 생성 → Stage 1, 전체 갤러리 → Stage 2
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR_BASE = f"{PROJECT_DIR}/outputs/standard_gen"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_DIR_BASE, exist_ok=True)
device = "cuda"

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 표준 Market-1501 데이터 로드
print("데이터 로드...")
gallery_files = sorted(glob.glob(f"{GALLERY_DIR}/*.jpg"))
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

# 메타데이터
gallery_meta = []  # (path, pid, cam)
for f in gallery_files:
    pid, cam = parse(f)
    gallery_meta.append((f, pid, cam))

query_meta = []
for f in query_files:
    pid, cam = parse(f)
    query_meta.append((f, pid, cam))

print(f"Gallery: {len(gallery_meta)}장")
print(f"Query: {len(query_meta)}장")

# CLIP
print("CLIP 로드...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def extract_features(paths, batch_size=64, desc=""):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inp = clip_proc(images=imgs, return_tensors="pt").to(device)
        f = clip_model.get_image_features(**inp)
        f = torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)

# Gallery, Query feature 추출
print("\nGallery feature 추출...")
gallery_paths = [m[0] for m in gallery_meta]
gallery_feats = extract_features(gallery_paths, desc="Gallery")
gallery_pids = np.array([m[1] for m in gallery_meta])
gallery_cams = np.array([m[2] for m in gallery_meta])

print("\nQuery feature 추출...")
query_paths = [m[0] for m in query_meta]
query_feats = extract_features(query_paths, desc="Query")
query_pids = np.array([m[1] for m in query_meta])
query_cams = np.array([m[2] for m in query_meta])

# ===== 표준 Re-ID 평가 함수 =====
def evaluate_reid(query_feats, query_pids, query_cams,
                   gallery_feats, gallery_pids, gallery_cams):
    """표준 mAP, Rank-K 평가"""
    num_q = len(query_pids)

    # 유사도 행렬
    sims = query_feats @ gallery_feats.T

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]

        # junk: 같은 ID + 같은 카메라, distractor (-1, 0000)
        order = np.argsort(-sims[q_idx])
        keep = np.ones(len(gallery_pids), dtype=bool)
        # 같은 ID + 같은 카메라 제외
        same_id_same_cam = (gallery_pids == q_pid) & (gallery_cams == q_cam)
        keep[same_id_same_cam] = False
        # distractor 제외
        keep[gallery_pids == "0000"] = False
        keep[gallery_pids == "-1"] = False

        order_valid = order[keep[order]]

        # 정답 찾기
        matches = (gallery_pids[order_valid] == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i+1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1

    if num_valid_q == 0:
        return 0.0, 0.0, 0.0, 0.0

    all_cmc = np.array(all_cmc).mean(axis=0)
    mAP = np.mean(all_AP)

    return mAP, all_cmc[0], all_cmc[4], all_cmc[9]  # mAP, R1, R5, R10

# ===== 방식 1: Baseline =====
print("\n" + "="*60)
print("방식 1: Baseline (표준 Re-ID)")
print("="*60)
mAP, r1, r5, r10 = evaluate_reid(
    query_feats, query_pids, query_cams,
    gallery_feats, gallery_pids, gallery_cams
)
print(f"mAP: {mAP*100:.2f}%  Rank-1: {r1*100:.2f}%  Rank-5: {r5*100:.2f}%  Rank-10: {r10*100:.2f}%")

# 단순 확장과 Stage-wise는 생성 이미지 필요
# (생성 6시간 작업 → 별도 진행)
