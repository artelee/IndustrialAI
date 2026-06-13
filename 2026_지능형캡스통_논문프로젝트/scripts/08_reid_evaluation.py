"""
소규모 Re-ID 평가
- 갤러리 확장 전/후 매칭 정확도 비교
- 5명 ID에 대해 측정
- OSNet (torchreid) 사용
"""

import os
import glob
import torch
import numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torchreid

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
DEBUG_DIR = f"{PROJECT_DIR}/debug"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

TEST_IDS = ["0001", "0003", "0004", "0005", "0006"]
device = "cuda"

# ===== 1. OSNet 모델 로드 =====
print("[1/5] OSNet 로드...")
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True,
)
model = model.eval().to(device)

# Re-ID 표준 전처리 (torchreid 기본값)
transform = T.Compose([
    T.Resize((256, 128)),  # OSNet 표준 입력
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_feature(img_path_or_pil):
    """이미지에서 OSNet 특징 벡터 추출"""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    # L2 정규화
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy().flatten()

print("✅ OSNet 로드 완료\n")

# ===== 2. ID별 이미지 매핑 =====
print("[2/5] 이미지 매핑...")
id_to_files = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{GALLERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    parts = fname.split("_")
    pid = parts[0]
    if pid in ("-1", "0000"):
        continue
    cam = parts[1][:2]
    id_to_files[pid][cam].append(f)

# 같은 ID의 query 이미지
query_to_files = defaultdict(list)
for f in glob.glob(f"{QUERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    pid = fname.split("_")[0]
    query_to_files[pid].append(f)

# ===== 3. 갤러리 구성 =====
print("[3/5] 갤러리 구성...")

# Baseline 갤러리: 5명 ID의 c1 이미지 1장씩
# (시점 변환 없이 다른 카메라로 매칭 가능한지 보기 위함)
baseline_gallery = []  # list of (path/PIL, id, cam)

for pid in TEST_IDS:
    if "c1" in id_to_files[pid]:
        c1_path = sorted(id_to_files[pid]["c1"])[0]
        baseline_gallery.append((c1_path, pid, "c1"))

print(f"Baseline 갤러리: {len(baseline_gallery)}장 (각 ID 1장씩, c1 카메라)")

# Expanded 갤러리: Baseline + 생성 이미지 (c3 시점으로 변환된 것)
expanded_gallery = list(baseline_gallery)

for pid in TEST_IDS:
    gen_path = f"{DEBUG_DIR}/plus_{pid}_generated.png"
    if os.path.exists(gen_path):
        expanded_gallery.append((gen_path, pid, "c3_generated"))

print(f"Expanded 갤러리: {len(expanded_gallery)}장 (Baseline + 생성 이미지)\n")

# ===== 4. 쿼리 준비 =====
# 각 ID마다 c3 카메라의 query 이미지 사용
# (생성 이미지가 c3 시점이므로 매칭 잘 되어야 함)
print("[4/5] Query 준비...")
queries = []  # list of (path, true_id)

for pid in TEST_IDS:
    if pid not in query_to_files:
        print(f"  ⚠️ ID {pid} query 없음")
        continue
    # c3 카메라 query 우선
    c3_queries = [q for q in query_to_files[pid] if "_c3" in os.path.basename(q)]
    if c3_queries:
        queries.append((sorted(c3_queries)[0], pid))
        print(f"  ID {pid}: {os.path.basename(c3_queries[0])} (c3)")
    else:
        # c3 없으면 아무거나
        queries.append((sorted(query_to_files[pid])[0], pid))
        print(f"  ID {pid}: {os.path.basename(query_to_files[pid][0])} (c3 없음)")

print(f"\nQuery 수: {len(queries)}\n")

# ===== 5. 특징 추출 및 매칭 =====
print("[5/5] 매칭 평가...")

def evaluate(gallery, queries, name):
    print(f"\n--- {name} ---")
    print(f"갤러리 크기: {len(gallery)}, Query 수: {len(queries)}")
    
    # 갤러리 특징 추출
    gallery_feats = np.array([extract_feature(g[0]) for g in gallery])
    gallery_ids = [g[1] for g in gallery]
    gallery_cams = [g[2] for g in gallery]
    
    # Query 매칭
    correct = 0
    details = []
    
    for q_path, q_id in queries:
        q_feat = extract_feature(q_path)
        
        # Cosine similarity (정규화 했으므로 dot product)
        sims = gallery_feats @ q_feat
        ranking = np.argsort(-sims)  # 내림차순 (높은 유사도가 앞)
        
        # Top-1
        top1_idx = ranking[0]
        top1_id = gallery_ids[top1_idx]
        top1_cam = gallery_cams[top1_idx]
        top1_sim = sims[top1_idx]
        
        is_correct = (top1_id == q_id)
        if is_correct:
            correct += 1
        
        # 정답과의 거리도 확인
        gt_indices = [i for i, gid in enumerate(gallery_ids) if gid == q_id]
        gt_sims = [(gallery_cams[i], sims[i]) for i in gt_indices]
        
        details.append({
            "query_id": q_id,
            "query_file": os.path.basename(q_path),
            "top1_id": top1_id,
            "top1_cam": top1_cam,
            "top1_sim": top1_sim,
            "correct": is_correct,
            "gt_sims": gt_sims,
        })
    
    # 결과 출력
    print(f"\nRank-1 정확도: {correct}/{len(queries)} = {100*correct/len(queries):.1f}%\n")
    
    for d in details:
        status = "✅" if d["correct"] else "❌"
        print(f"{status} Query ID={d['query_id']} ({d['query_file']})")
        print(f"   Top-1: ID={d['top1_id']} cam={d['top1_cam']} sim={d['top1_sim']:.4f}")
        print(f"   정답들과의 유사도:")
        for cam, sim in d["gt_sims"]:
            mark = " ⭐" if cam == "c3_generated" else ""
            print(f"     {cam}: {sim:.4f}{mark}")
    
    return correct / len(queries)


# 두 조건 평가
acc_baseline = evaluate(baseline_gallery, queries, "Baseline (원본만)")
acc_expanded = evaluate(expanded_gallery, queries, "Expanded (원본 + 생성)")

print("\n" + "=" * 60)
print(f"📊 최종 결과")
print("=" * 60)
print(f"Baseline:  {100*acc_baseline:.1f}% (갤러리 {len(baseline_gallery)}장)")
print(f"Expanded:  {100*acc_expanded:.1f}% (갤러리 {len(expanded_gallery)}장)")
print(f"변화량:    {100*(acc_expanded - acc_baseline):+.1f}%p")
print()
print("⭐ = 생성 이미지가 query와 얼마나 가까운지 (sim 값 클수록 좋음)")
print("    원본 c1과의 sim 보다 크면 생성 이미지가 매칭에 기여한 것")