"""
CLIP feature로 sim 측정
IP-Adapter와 동일한 feature 공간
→ 생성 이미지 sim이 올라갈 가능성
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/generated_50"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

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

valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and "c3" in query_by_id[pid]:
        gen_path = f"{GEN_DIR}/{pid}_c3_generated.png"
        if os.path.exists(gen_path):
            valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

print(f"분석 ID: {len(valid_ids)}명")

# CLIP 로드
print("CLIP 로드...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("✅ CLIP 로드 완료\n")

@torch.no_grad()
def clip_feat(path_or_pil):
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    inputs = processor(images=img, return_tensors="pt").to(device)
    feat = model.get_image_features(**inputs)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy().flatten()

print("="*65)
print(f"{'ID':<8} {'sim(c3, 생성c3)':>18} {'sim(c3, 원본c1)':>18} {'생성>원본?':>10}")
print("-"*65)

results = []
better_count = 0

for pid in valid_ids:
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c3_query = sorted(query_by_id[pid]["c3"])[0]
    gen_path = f"{GEN_DIR}/{pid}_c3_generated.png"

    f_c3   = clip_feat(c3_query)
    f_gen  = clip_feat(gen_path)
    f_c1   = clip_feat(c1_path)

    sim_gen = float(f_c3 @ f_gen)
    sim_c1  = float(f_c3 @ f_c1)
    better  = "✅" if sim_gen > sim_c1 else "❌"
    if sim_gen > sim_c1:
        better_count += 1

    results.append({"pid": pid, "sim_gen": sim_gen, "sim_c1": sim_c1})
    print(f"{pid:<8} {sim_gen:>18.4f} {sim_c1:>18.4f} {better:>10}")

print("="*65)
avg_gen = np.mean([r["sim_gen"] for r in results])
avg_c1  = np.mean([r["sim_c1"]  for r in results])
n = len(results)

print(f"\n[CLIP feature 결과]")
print(f"sim(진짜c3, 생성c3) 평균:  {avg_gen:.4f}")
print(f"sim(진짜c3, 원본c1) 평균:  {avg_c1:.4f}")
print(f"생성이 원본보다 가까운 케이스: {better_count}/{n} ({100*better_count/n:.1f}%)")
print(f"\n[이전 OSNet 결과]")
print(f"sim(진짜c3, 생성c3) 평균:  0.5714")
print(f"sim(진짜c3, 원본c1) 평균:  0.7560")
print(f"생성이 원본보다 가까운 케이스: 2/50 (4.0%)")

if avg_gen > 0.5714:
    print(f"\n✅ CLIP feature가 OSNet보다 생성 이미지 sim 향상")
    print(f"   IP-Adapter와 같은 feature 공간이라 예상된 결과")
else:
    print(f"\n❌ CLIP feature도 OSNet과 비슷")
