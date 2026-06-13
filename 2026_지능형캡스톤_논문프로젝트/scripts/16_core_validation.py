
"""
핵심 검증:
생성 c2 이미지가 진짜 c2 이미지와 얼마나 가까운가?
vs c1 원본과 진짜 c2의 거리

이게 논문의 진짜 검증이야.
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torchreid

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/generated_50"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

device = "cuda"

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터 로드
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ("-1","0000"): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# OSNet 로드
model = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
torchreid.utils.load_pretrained_weights(model, OSNET_WEIGHTS)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize((256,128)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@torch.no_grad()
def feat(path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    f = model(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 핵심 비교
print("="*60)
print("핵심 검증: 생성 c2 vs 진짜 c2 거리")
print("="*60)
print(f"{'ID':<8} {'sim(진짜c2, 생성c2)':>22} {'sim(진짜c2, 원본c1)':>22} {'생성이 더 가까운가':>16}")
print("-"*72)

results = []
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and "c3" in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

better_count = 0
for pid in valid_ids:
    gen_path = f"{GEN_DIR}/{pid}_c3_generated.png"
    if not os.path.exists(gen_path):
        continue

    # 진짜 c3 query
    real_c3 = sorted(query_by_id[pid]["c3"])[0]
    # c1 원본
    real_c1 = sorted(gallery_by_id[pid]["c1"])[0]

    f_real_c3 = feat(real_c3)
    f_gen_c3  = feat(gen_path)
    f_real_c1 = feat(real_c1)

    sim_gen  = float(f_real_c3 @ f_gen_c3)   # 진짜c3 vs 생성c3
    sim_c1   = float(f_real_c3 @ f_real_c1)  # 진짜c3 vs 원본c1

    better = "✅ YES" if sim_gen > sim_c1 else "❌ NO"
    if sim_gen > sim_c1:
        better_count += 1

    results.append({"pid": pid, "sim_gen": sim_gen, "sim_c1": sim_c1})
    print(f"{pid:<8} {sim_gen:>22.4f} {sim_c1:>22.4f} {better:>16}")

n = len(results)
avg_gen = np.mean([r["sim_gen"] for r in results])
avg_c1  = np.mean([r["sim_c1"]  for r in results])

print("="*72)
print(f"\n{'분석 ID 수:':<30} {n}")
print(f"{'sim(진짜c2, 생성c2) 평균:':<30} {avg_gen:.4f}")
print(f"{'sim(진짜c2, 원본c1) 평균:':<30} {avg_c1:.4f}")
print(f"{'생성이 원본보다 가까운 케이스:':<30} {better_count}/{n} ({100*better_count/n:.1f}%)")

print("\n" + "="*60)
if avg_gen > avg_c1:
    print("✅ 결론: 생성 c2가 원본 c1보다 진짜 c2에 더 가까움")
    print("   → 너 방법 유효. 갤러리 확장이 의미 있음.")
else:
    print("❌ 결론: 생성 c2가 원본 c1보다 진짜 c2에 더 멀음")
    print("   → 생성 품질 부족. 방법 개선 필요.")
print("="*60)
