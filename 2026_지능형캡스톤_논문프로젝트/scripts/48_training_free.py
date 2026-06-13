"""
Training-Free 평가
ID 학습 안 된 일반 비전 모델로 매칭

모델:
1. CLIP (일반)
2. DINOv2
3. ViT-ImageNet

셋업: 100명 갤러리, 모든 cross-camera 페어
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_BASE = f"{PROJECT_DIR}/outputs/c1base_gen_all"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
NUM_IDS = 100
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터 로드
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
for target_cam in TARGET_CAMS:
    ids = []
    for pid in sorted(gallery_by_id.keys()):
        if "c1" in gallery_by_id[pid] and target_cam in query_by_id[pid]:
            gen_path = f"{GEN_BASE}/{target_cam}/{pid}_gen_{target_cam}.png"
            if os.path.exists(gen_path):
                ids.append(pid)
        if len(ids) >= NUM_IDS:
            break
    cam_valid_ids[target_cam] = ids
    print(f"  c1 → {target_cam}: {len(ids)}명")

# === Feature Extractors ===
print("\n모델 로드...")

# 1. CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP 로드")

# 2. DINOv2
try:
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
    dino_transform = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    has_dino = True
    print("✅ DINOv2 로드")
except Exception as e:
    print(f"⚠️ DINOv2 실패: {e}")
    has_dino = False

# 3. ViT-ImageNet
try:
    import timm
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).to(device).eval()
    vit_transform = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    has_vit = True
    print("✅ ViT-ImageNet 로드")
except Exception as e:
    print(f"⚠️ ViT 실패: {e}")
    has_vit = False

@torch.no_grad()
def clip_feat(path):
    img = Image.open(path).convert("RGB")
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inp)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def dino_feat(path):
    img = Image.open(path).convert("RGB")
    t = dino_transform(img).unsqueeze(0).to(device)
    f = dinov2(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def vit_feat(path):
    img = Image.open(path).convert("RGB")
    t = vit_transform(img).unsqueeze(0).to(device)
    f = vit_model(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 평가
def eval_pair(feat_fn, target_cam, valid_ids):
    baseline_gf, ours_gf, query_f = [], [], []
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(query_by_id[pid][target_cam])[0]
        baseline_gf.append(feat_fn(c1_path))
        ours_gf.append(feat_fn(gen_path))
        query_f.append(feat_fn(q_path))
    bf, of, qf = np.array(baseline_gf), np.array(ours_gf), np.array(query_f)
    def r1(gf):
        sims = qf @ gf.T
        return sum(1 for i in range(len(qf)) if sims[i].argmax() == i) / len(qf)
    def self_sim(gf):
        return np.mean([qf[i] @ gf[i] for i in range(len(qf))])
    return r1(bf), r1(of), self_sim(bf), self_sim(of)

# 실행
all_results = {}

for model_name, feat_fn in [
    ("CLIP", clip_feat),
    ("DINOv2", dino_feat if has_dino else None),
    ("ViT-ImageNet", vit_feat if has_vit else None),
]:
    if feat_fn is None: continue
    print(f"\n" + "="*70)
    print(f"[{model_name}]")
    print("="*70)
    results = {}
    for cam in TARGET_CAMS:
        print(f"\nc1 → {cam}")
        r1_b, r1_o, sim_b, sim_o = eval_pair(feat_fn, cam, cam_valid_ids[cam])
        results[cam] = (r1_b, r1_o, sim_b, sim_o)
        print(f"  Baseline R1: {r1_b*100:.2f}% (sim {sim_b:.4f})")
        print(f"  Ours     R1: {r1_o*100:.2f}% (sim {sim_o:.4f})")
    all_results[model_name] = results

# 종합표
print("\n\n" + "="*100)
print("Training-Free 종합 결과 (모든 페어, 100명 갤러리)")
print("="*100)

for model_name, results in all_results.items():
    print(f"\n[{model_name}]")
    print(f"{'Pair':<10}{'Baseline R1':<15}{'Ours R1':<15}{'향상':<12}{'B sim':<10}{'O sim':<10}")
    print("-"*100)
    for cam, (rb, ro, sb, so) in results.items():
        diff = (ro-rb)*100
        mark = "✅" if ro > rb else ("=" if ro == rb else "❌")
        print(f"c1→{cam:<7}{rb*100:<15.2f}{ro*100:<15.2f}{diff:+.2f} {mark:<5}{sb:<10.4f}{so:<10.4f}")
