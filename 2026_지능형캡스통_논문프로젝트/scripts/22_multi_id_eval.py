
"""
50명 multi-ID Re-ID 평가
진짜 시나리오:
- 갤러리에 50명 등록 (각 ID 1~N장)
- Query: 실제 c6 이미지
- 매칭: 50명 중 정답 ID 찾기

조건:
A. 갤러리 = 각 ID의 c1만 (50장)
B. 갤러리 = 각 ID의 c1 + 생성 c6 (100장)
C. 갤러리 = 각 ID의 c1 + 실제 c6 (100장) - upper bound
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchreid
import torchvision.transforms as T

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/clip_gen_c6"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

device = "cuda"

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ("-1","0000"): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# 50명 선택
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    cams = set(gallery_by_id[pid].keys())
    if cams >= {"c1","c2","c3","c4","c5","c6"} and "c6" in query_by_id[pid]:
        gen_path = f"{GEN_DIR}/{pid}_gen_c6.png"
        if os.path.exists(gen_path):
            valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

print(f"평가 대상: {len(valid_ids)}명\n")

# ===== Feature extractor 두 개 =====
print("CLIP 로드...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("OSNet 로드...")
osnet = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
torchreid.utils.load_pretrained_weights(osnet, OSNET_WEIGHTS)
osnet = osnet.eval().to(device)
osnet_tf = T.Compose([
    T.Resize((256,128)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@torch.no_grad()
def clip_feat(path_or_pil):
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inp)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def osnet_feat(path_or_pil):
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    t = osnet_tf(img).unsqueeze(0).to(device)
    f = osnet(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# ===== 평가 함수 =====
def evaluate(feat_fn, name):
    """3가지 조건으로 평가"""

    # Query: 각 ID의 실제 c6 이미지
    query_feats = []
    query_ids = []
    for pid in valid_ids:
        q_path = sorted(query_by_id[pid]["c6"])[0]
        query_feats.append(feat_fn(q_path))
        query_ids.append(pid)
    query_feats = np.array(query_feats)

    # ===== 조건 A: Baseline (c1만) =====
    gallery_feats_A = []
    gallery_ids_A = []
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gallery_feats_A.append(feat_fn(c1_path))
        gallery_ids_A.append(pid)
    gallery_feats_A = np.array(gallery_feats_A)

    # ===== 조건 B: Ours (c1 + 생성 c6) =====
    gallery_feats_B = list(gallery_feats_A)
    gallery_ids_B = list(gallery_ids_A)
    for pid in valid_ids:
        gen_path = f"{GEN_DIR}/{pid}_gen_c6.png"
        gallery_feats_B.append(feat_fn(gen_path))
        gallery_ids_B.append(pid)
    gallery_feats_B = np.array(gallery_feats_B)

    # ===== 조건 C: Upper bound (c1 + 실제 c6) =====
    gallery_feats_C = list(gallery_feats_A)
    gallery_ids_C = list(gallery_ids_A)
    for pid in valid_ids:
        # 실제 c6 갤러리 이미지 (query랑 다른 거)
        c6_gallery = gallery_by_id[pid].get("c6", [])
        if c6_gallery:
            gallery_feats_C.append(feat_fn(c6_gallery[0]))
            gallery_ids_C.append(pid)
    gallery_feats_C = np.array(gallery_feats_C)

    # 매칭 평가
    def compute_rank1(query_feats, gallery_feats, gallery_ids):
        sims = query_feats @ gallery_feats.T  # (N_query, N_gallery)
        top1_idx = sims.argmax(axis=1)
        correct = sum(1 for i, idx in enumerate(top1_idx) 
                     if gallery_ids[idx] == query_ids[i])
        return correct / len(query_ids)

    r1_A = compute_rank1(query_feats, gallery_feats_A, gallery_ids_A)
    r1_B = compute_rank1(query_feats, gallery_feats_B, gallery_ids_B)
    r1_C = compute_rank1(query_feats, gallery_feats_C, gallery_ids_C)

    print(f"\n[{name} feature]")
    print(f"  조건 A (c1만):              Rank-1 = {r1_A:.4f} ({r1_A*100:.1f}%)")
    print(f"  조건 B (c1 + 생성c6):       Rank-1 = {r1_B:.4f} ({r1_B*100:.1f}%)")
    print(f"  조건 C (c1 + 실제c6) [upper]: Rank-1 = {r1_C:.4f} ({r1_C*100:.1f}%)")
    print(f"  → B - A 향상: {(r1_B-r1_A)*100:+.1f}%p")
    print(f"  → C - A 향상 (이상): {(r1_C-r1_A)*100:+.1f}%p")
    print(f"  → 생성이 이상 대비 회복률: {(r1_B-r1_A)/(r1_C-r1_A+1e-6)*100:.1f}%")
    return r1_A, r1_B, r1_C

# ===== 실행 =====
print("="*70)
print("Multi-ID Re-ID 평가 (50명, c1→c6)")
print("="*70)

evaluate(clip_feat, "CLIP")
evaluate(osnet_feat, "OSNet")
