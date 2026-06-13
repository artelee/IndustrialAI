"""
Stage 1: Diagnostic Analysis
- 갤러리 시점 다양성 분석
- 시점 다양성과 매칭 정확도 상관관계
- 50명 확장 케이스별 효과 분해
"""

import os
import glob
import json
import torch
import numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torchreid
from tqdm import tqdm

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/generated_50"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

device = "cuda"
N_IDS = 50

def parse_market_filename(filepath):
    fname = os.path.basename(filepath)
    parts = fname.split("_")
    return parts[0], parts[1][:2]

# ===== 데이터 구조 =====
print("=" * 60)
print("[STAGE 1] Diagnostic Analysis")
print("=" * 60)

all_gallery_files = sorted(glob.glob(f"{GALLERY_DIR}/*.jpg"))
all_query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

# Gallery: ID별 카메라 분포
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in all_gallery_files:
    pid, cam = parse_market_filename(f)
    if pid in ("-1", "0000"):
        continue
    gallery_by_id[pid][cam].append(f)

# Query: ID별 카메라
query_by_id = defaultdict(lambda: defaultdict(list))
for f in all_query_files:
    pid, cam = parse_market_filename(f)
    query_by_id[pid][cam].append(f)

# ===== Q1: 갤러리 시점 다양성 분포 =====
print("\n" + "=" * 60)
print("[Q1] 갤러리 시점 다양성 분포")
print("=" * 60)

n_cams_per_id = [len(gallery_by_id[pid]) for pid in gallery_by_id]
n_imgs_per_id = [sum(len(v) for v in gallery_by_id[pid].values()) for pid in gallery_by_id]

print(f"총 ID 수: {len(gallery_by_id)}")
print(f"\nID당 카메라 수 분포:")
for n in range(1, 7):
    count = sum(1 for x in n_cams_per_id if x == n)
    pct = 100 * count / len(n_cams_per_id)
    bar = "█" * int(pct / 2)
    print(f"  {n}개 카메라: {count:>3}명 ({pct:>5.1f}%) {bar}")

print(f"\nID당 이미지 수 분포:")
print(f"  평균: {np.mean(n_imgs_per_id):.1f}")
print(f"  중앙값: {np.median(n_imgs_per_id):.0f}")
print(f"  최소: {min(n_imgs_per_id)}, 최대: {max(n_imgs_per_id)}")

# ===== Q2: 시점 다양성 vs 매칭 정확도 =====
print("\n" + "=" * 60)
print("[Q2] 시점 다양성과 매칭 정확도 상관관계")
print("=" * 60)

# OSNet 로드
print("OSNet 로드 중...")
model = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
torchreid.utils.load_pretrained_weights(model, OSNET_WEIGHTS)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_batch(paths, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="특징"):
        batch = paths[i:i+batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = model(imgs)
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

# 전체 특징 추출
print("\nGallery 특징 추출...")
gallery_feats = extract_batch(all_gallery_files)
gallery_pids = np.array([parse_market_filename(f)[0] for f in all_gallery_files])
gallery_cams = np.array([parse_market_filename(f)[1] for f in all_gallery_files])

print("\nQuery 특징 추출...")
query_feats = extract_batch(all_query_files)
query_pids = np.array([parse_market_filename(f)[0] for f in all_query_files])
query_cams = np.array([parse_market_filename(f)[1] for f in all_query_files])

# ID별 AP 계산
print("\nID별 AP 계산...")
distmat = 1 - query_feats @ gallery_feats.T

id_to_aps = defaultdict(list)
for q_idx in range(len(query_pids)):
    q_pid, q_cam = query_pids[q_idx], query_cams[q_idx]
    order = np.argsort(distmat[q_idx])
    remove = (gallery_pids[order] == q_pid) & (gallery_cams[order] == q_cam)
    junk = (gallery_pids[order] == "-1") | (gallery_pids[order] == "0000")
    keep = np.invert(remove) & np.invert(junk)
    
    orig_cmc = (gallery_pids[order] == q_pid).astype(np.int32)[keep]
    if not np.any(orig_cmc):
        continue
    
    num_rel = orig_cmc.sum()
    tmp_cmc = orig_cmc.cumsum()
    tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
    AP = (np.asarray(tmp_cmc) * orig_cmc).sum() / num_rel
    id_to_aps[q_pid].append(AP)

# 시점 다양성별 mAP
print("\n카메라 수 그룹별 mAP:")
print(f"{'카메라 수':<12} {'ID 수':<10} {'mAP':<10} {'평균 이미지':<12}")
print("-" * 50)
for n_cam in range(1, 7):
    ids = [pid for pid in id_to_aps if len(gallery_by_id[pid]) == n_cam]
    if not ids:
        continue
    aps = []
    n_imgs = []
    for pid in ids:
        aps.extend(id_to_aps[pid])
        n_imgs.append(sum(len(v) for v in gallery_by_id[pid].values()))
    mean_ap = 100 * np.mean(aps)
    print(f"  {n_cam}개         {len(ids):<10} {mean_ap:<10.2f} {np.mean(n_imgs):<12.1f}")

# ===== Q3: 50명 확장 케이스 분석 =====
print("\n" + "=" * 60)
print("[Q3] 50명 확장 효과 케이스별 분해")
print("=" * 60)

# 확장한 50명 ID 다시 식별
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and "c3" in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= N_IDS:
        break

# 생성 이미지 특징
gen_paths = [f"{GEN_DIR}/{pid}_c3_generated.png" for pid in valid_ids]
gen_paths = [p for p in gen_paths if os.path.exists(p)]
print(f"\n생성 이미지: {len(gen_paths)}장")
gen_feats = extract_batch(gen_paths)

# 각 확장 ID에 대해:
# - baseline c1 이미지의 query sim
# - 진짜 c3 이미지의 query sim  
# - 가짜 c3 이미지(생성)의 query sim
print(f"\n{'ID':<8} {'cams':<6} {'real_c3_sim':<14} {'fake_c3_sim':<14} {'diff':<10}")
print("-" * 60)

per_id_stats = []
for idx, pid in enumerate(valid_ids):
    # 이 ID의 c3 query (첫 번째만)
    c3_q = [(c, f) for c, qs in query_by_id[pid].items() if c == "c3" for f in qs]
    if not c3_q:
        continue
    q_path = c3_q[0][1]
    q_feat = extract_batch([q_path])[0]
    
    # 진짜 c3 이미지 (갤러리)
    real_c3_imgs = gallery_by_id[pid].get("c3", [])
    if real_c3_imgs:
        real_feats = extract_batch(real_c3_imgs)
        real_sims = real_feats @ q_feat
        real_sim_mean = real_sims.mean()
    else:
        real_sim_mean = None
    
    # 가짜 c3 (생성)
    fake_feat = gen_feats[idx]
    fake_sim = float(fake_feat @ q_feat)
    
    n_cams = len(gallery_by_id[pid])
    
    if real_sim_mean is not None:
        diff = fake_sim - real_sim_mean
        per_id_stats.append({
            "pid": pid, "n_cams": n_cams,
            "real_sim": float(real_sim_mean), "fake_sim": fake_sim, "diff": float(diff)
        })
        if idx < 10:  # 처음 10개만 출력
            print(f"{pid:<8} {n_cams:<6} {real_sim_mean:<14.4f} {fake_sim:<14.4f} {diff:+.4f}")

# 통계
if per_id_stats:
    diffs = [s["diff"] for s in per_id_stats]
    print(f"\n📊 통계:")
    print(f"  분석 ID 수: {len(per_id_stats)}")
    print(f"  fake_sim - real_sim 평균: {np.mean(diffs):+.4f}")
    print(f"  fake가 real보다 큰 케이스: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    print(f"  → fake_sim 평균: {np.mean([s['fake_sim'] for s in per_id_stats]):.4f}")
    print(f"  → real_sim 평균: {np.mean([s['real_sim'] for s in per_id_stats]):.4f}")

# JSON 저장
out = {
    "n_cams_distribution": {n: sum(1 for x in n_cams_per_id if x == n) for n in range(1, 7)},
    "per_id_stats": per_id_stats,
}
with open(f"{PROJECT_DIR}/outputs/diagnostic.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n✅ 진단 결과 저장: outputs/diagnostic.json")