"""
표준 Market-1501 평가 (Re-ID 학습 가중치 적용)
- OSNet x1.0 (Market-1501 fine-tuned, mAP 82.6% 베이스라인)
- 이전 10번 스크립트와 동일하지만 가중치만 변경
"""

import os
import glob
import json
import time
import torch
import numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torchreid
from tqdm import tqdm

# ========== 경로 ==========
HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/generated_50"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

# 🔧 Re-ID 학습된 가중치 경로
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

device = "cuda"
N_IDS = 50

# ========== 파일명 파서 ==========
def parse_market_filename(filepath):
    fname = os.path.basename(filepath)
    parts = fname.split("_")
    pid = parts[0]
    cam = parts[1][:2]
    return pid, cam

# ========== 1. ID 선택 (이전과 동일) ==========
print("=" * 60)
print("[STEP 1] ID 선택")
print("=" * 60)

all_gallery_files = sorted(glob.glob(f"{GALLERY_DIR}/*.jpg"))
all_query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in all_gallery_files:
    pid, cam = parse_market_filename(f)
    if pid in ("-1", "0000"):
        continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in all_query_files:
    pid, cam = parse_market_filename(f)
    query_by_id[pid][cam].append(f)

valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and "c3" in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= N_IDS:
        break

print(f"선택된 50명: {valid_ids[:5]} ... {valid_ids[-5:]}")

# ========== 2. 기존 생성 이미지 사용 ==========
generated_paths = {}
for pid in valid_ids:
    p = f"{GEN_DIR}/{pid}_c3_generated.png"
    if os.path.exists(p):
        generated_paths[pid] = p

print(f"기존 생성 이미지: {len(generated_paths)}/{N_IDS}")
if len(generated_paths) < N_IDS:
    print("❌ 생성 이미지 부족. 먼저 scripts/10_standard_eval.py 의 STEP 4를 돌려야 함")
    exit(1)

# ========== 3. OSNet 로드 (🔧 Re-ID 학습 가중치) ==========
print("\n" + "=" * 60)
print("[STEP 3] OSNet 로드 (Re-ID 학습 가중치)")
print("=" * 60)

# num_classes는 학습 시 ID 수 (Market train: 751)
# 평가에서는 분류 layer 안 쓰니까 상관 없음
model = torchreid.models.build_model(
    name='osnet_x1_0', 
    num_classes=751,
    pretrained=False,  # 🔧 ImageNet pretrained 안 받음 (덮어쓸 거니까)
)

# 🔧 Re-ID 학습 가중치 로드
print(f"가중치 로드: {OSNET_WEIGHTS}")
torchreid.utils.load_pretrained_weights(model, OSNET_WEIGHTS)

model = model.eval().to(device)
print("✅ OSNet 로드 완료 (Market-1501 fine-tuned)")

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_batch(paths, batch_size=64):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="특징 추출"):
        batch = paths[i:i+batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = model(imgs)
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

# ========== 4. 특징 추출 ==========
print("\n" + "=" * 60)
print("[STEP 4] 특징 추출")
print("=" * 60)

gallery_paths = all_gallery_files
gallery_pids = [parse_market_filename(f)[0] for f in gallery_paths]
gallery_cams = [parse_market_filename(f)[1] for f in gallery_paths]

print(f"\n원본 Gallery: {len(gallery_paths)}장")
gallery_feats = extract_batch(gallery_paths)

print(f"\nQuery: {len(all_query_files)}장")
query_paths = all_query_files
query_pids = [parse_market_filename(f)[0] for f in query_paths]
query_cams = [parse_market_filename(f)[1] for f in query_paths]
query_feats = extract_batch(query_paths)

print(f"\n생성 이미지: {len(generated_paths)}장")
gen_paths = [generated_paths[pid] for pid in valid_ids]
gen_pids = list(valid_ids)
gen_cams = ["c3"] * len(valid_ids)
gen_feats = extract_batch(gen_paths)

# ========== 5. 표준 평가 ==========
print("\n" + "=" * 60)
print("[STEP 5] 표준 Market-1501 평가")
print("=" * 60)

def evaluate_market1501(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams, max_rank=50):
    q_pids = np.array(q_pids)
    q_cams = np.array(q_cams)
    g_pids = np.array(g_pids)
    g_cams = np.array(g_cams)
    
    distmat = 1 - q_feats @ g_feats.T
    num_q = q_feats.shape[0]
    
    all_cmc = []
    all_aps = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_cams[q_idx]
        
        order = np.argsort(distmat[q_idx])
        remove = (g_pids[order] == q_pid) & (g_cams[order] == q_cam)
        keep = np.invert(remove)
        junk = (g_pids[order] == "-1") | (g_pids[order] == "0000")
        keep = keep & np.invert(junk)
        
        orig_cmc = (g_pids[order] == q_pid).astype(np.int32)[keep]
        if not np.any(orig_cmc):
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_aps.append(AP)
    
    if num_valid_q == 0:
        return None
    
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return {
        "mAP": 100 * np.mean(all_aps),
        "Rank-1": 100 * all_cmc[0],
        "Rank-5": 100 * all_cmc[4],
        "Rank-10": 100 * all_cmc[9],
        "num_valid_q": num_valid_q,
    }

# Baseline
print("\n--- Baseline ---")
result_base = evaluate_market1501(
    query_feats, query_pids, query_cams,
    gallery_feats, gallery_pids, gallery_cams,
)
print(f"mAP={result_base['mAP']:.2f}%  R-1={result_base['Rank-1']:.2f}%  "
      f"R-5={result_base['Rank-5']:.2f}%  R-10={result_base['Rank-10']:.2f}%")

# Expanded
print("\n--- Expanded ---")
ext_g_feats = np.concatenate([gallery_feats, gen_feats], axis=0)
ext_g_pids = list(gallery_pids) + list(gen_pids)
ext_g_cams = list(gallery_cams) + list(gen_cams)

result_exp = evaluate_market1501(
    query_feats, query_pids, query_cams,
    ext_g_feats, ext_g_pids, ext_g_cams,
)
print(f"mAP={result_exp['mAP']:.2f}%  R-1={result_exp['Rank-1']:.2f}%  "
      f"R-5={result_exp['Rank-5']:.2f}%  R-10={result_exp['Rank-10']:.2f}%")

# 50명 한정
print("\n--- 50명 한정 (확장 효과 직접 측정) ---")
mask = np.isin(query_pids, valid_ids)
sub_q_feats = query_feats[mask]
sub_q_pids = np.array(query_pids)[mask]
sub_q_cams = np.array(query_cams)[mask]

result_base_50 = evaluate_market1501(
    sub_q_feats, sub_q_pids, sub_q_cams,
    gallery_feats, gallery_pids, gallery_cams,
)
result_exp_50 = evaluate_market1501(
    sub_q_feats, sub_q_pids, sub_q_cams,
    ext_g_feats, ext_g_pids, ext_g_cams,
)
print(f"50명 query 수: {mask.sum()}")
print(f"Baseline: mAP={result_base_50['mAP']:.2f}%  R-1={result_base_50['Rank-1']:.2f}%")
print(f"Expanded: mAP={result_exp_50['mAP']:.2f}%  R-1={result_exp_50['Rank-1']:.2f}%")

# ========== 최종 ==========
print("\n" + "=" * 60)
print("📊 최종 결과")
print("=" * 60)
print(f"\n{'조건':<25} {'mAP':>8} {'Rank-1':>10} {'Rank-5':>10} {'Rank-10':>10}")
print("-" * 65)
print(f"{'Baseline (전체)':<25} {result_base['mAP']:>7.2f}% {result_base['Rank-1']:>9.2f}% "
      f"{result_base['Rank-5']:>9.2f}% {result_base['Rank-10']:>9.2f}%")
print(f"{'Expanded (전체)':<25} {result_exp['mAP']:>7.2f}% {result_exp['Rank-1']:>9.2f}% "
      f"{result_exp['Rank-5']:>9.2f}% {result_exp['Rank-10']:>9.2f}%")
diff_full = {k: result_exp[k] - result_base[k] for k in ["mAP", "Rank-1", "Rank-5", "Rank-10"]}
print(f"{'변화 (전체)':<25} {diff_full['mAP']:>+7.2f}p {diff_full['Rank-1']:>+9.2f}p "
      f"{diff_full['Rank-5']:>+9.2f}p {diff_full['Rank-10']:>+9.2f}p")
print()
print(f"{'Baseline (50명)':<25} {result_base_50['mAP']:>7.2f}% {result_base_50['Rank-1']:>9.2f}% "
      f"{result_base_50['Rank-5']:>9.2f}% {result_base_50['Rank-10']:>9.2f}%")
print(f"{'Expanded (50명)':<25} {result_exp_50['mAP']:>7.2f}% {result_exp_50['Rank-1']:>9.2f}% "
      f"{result_exp_50['Rank-5']:>9.2f}% {result_exp_50['Rank-10']:>9.2f}%")
diff_50 = {k: result_exp_50[k] - result_base_50[k] for k in ["mAP", "Rank-1", "Rank-5", "Rank-10"]}
print(f"{'변화 (50명)':<25} {diff_50['mAP']:>+7.2f}p {diff_50['Rank-1']:>+9.2f}p "
      f"{diff_50['Rank-5']:>+9.2f}p {diff_50['Rank-10']:>+9.2f}p")

# JSON 저장
out = {
    "weight_path": OSNET_WEIGHTS,
    "n_expanded_ids": len(valid_ids),
    "baseline_full": result_base,
    "expanded_full": result_exp,
    "baseline_50only": result_base_50,
    "expanded_50only": result_exp_50,
}
with open(f"{PROJECT_DIR}/outputs/result_standard_v2.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n결과: outputs/result_standard_v2.json")