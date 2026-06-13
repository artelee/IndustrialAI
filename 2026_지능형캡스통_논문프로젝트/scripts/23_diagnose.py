import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/clip_gen_c6"
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
    cams = set(gallery_by_id[pid].keys())
    if cams >= {"c1","c2","c3","c4","c5","c6"} and "c6" in query_by_id[pid]:
        if os.path.exists(f"{GEN_DIR}/{pid}_gen_c6.png"):
            valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def feat(p):
    img = Image.open(p).convert("RGB") if isinstance(p, str) else p
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inp)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 갤러리 구성: 각 ID당 c1과 생성 c6 따로 저장
gallery_c1 = []   # 50개 (각 ID의 c1)
gallery_gen = []  # 50개 (각 ID의 생성 c6)
for pid in valid_ids:
    c1 = sorted(gallery_by_id[pid]["c1"])[0]
    gen = f"{GEN_DIR}/{pid}_gen_c6.png"
    gallery_c1.append(feat(c1))
    gallery_gen.append(feat(gen))
gallery_c1 = np.array(gallery_c1)
gallery_gen = np.array(gallery_gen)

# 각 query에 대해 진단
print(f"{'ID':<8} {'A(c1만) Top-1':<15} {'B(c1+gen) Top-1':<18} {'gen이 c1 이김?':<15} {'gen 자기ID Top-1?':<15}")
print("-"*90)

correct_A = 0
correct_B = 0
gen_wins_own = 0
gen_top1_for_own = 0

for i, pid in enumerate(valid_ids):
    q_path = sorted(query_by_id[pid]["c6"])[0]
    q_feat = feat(q_path)

    # 조건 A: c1만
    sims_A = q_feat @ gallery_c1.T
    top1_A = sims_A.argmax()
    pred_A = valid_ids[top1_A]
    ok_A = "✅" if pred_A == pid else "❌"
    if pred_A == pid: correct_A += 1

    # 조건 B: c1 + 생성
    gallery_B = np.concatenate([gallery_c1, gallery_gen])
    gallery_B_ids = valid_ids + valid_ids
    sims_B = q_feat @ gallery_B.T
    top1_B = sims_B.argmax()
    pred_B = gallery_B_ids[top1_B]
    src_B = "gen" if top1_B >= len(valid_ids) else "c1"
    ok_B = "✅" if pred_B == pid else "❌"
    if pred_B == pid: correct_B += 1

    # 자기 ID 내에서: gen이 c1보다 가까운가?
    sim_own_c1 = q_feat @ gallery_c1[i]
    sim_own_gen = q_feat @ gallery_gen[i]
    gen_better = sim_own_gen > sim_own_c1
    if gen_better: gen_wins_own += 1

    # gen이 자기 ID에 대해 Top-1?
    rank_of_gen = (sims_B > sims_B[len(valid_ids)+i]).sum()  # 자기 gen보다 가까운 게 몇 개
    if rank_of_gen == 0: gen_top1_for_own += 1

    print(f"{pid:<8} {pred_A:<15} {pred_B+'('+src_B+')':<18} {'✅' if gen_better else '❌':<15} {'✅' if rank_of_gen==0 else '❌':<15}")

print("="*90)
print(f"\n[종합]")
print(f"조건 A 정확도: {correct_A}/{len(valid_ids)} = {100*correct_A/len(valid_ids):.1f}%")
print(f"조건 B 정확도: {correct_B}/{len(valid_ids)} = {100*correct_B/len(valid_ids):.1f}%")
print(f"\n자기 ID 내에서 gen이 c1보다 가까운 케이스: {gen_wins_own}/{len(valid_ids)}")
print(f"gen이 전체 갤러리에서 자기 ID Top-1: {gen_top1_for_own}/{len(valid_ids)}")
print(f"\n→ {gen_wins_own}명은 gen이 자기 c1보다 가까운데")
print(f"  {gen_top1_for_own}명만 전체에서 Top-1")
print(f"  나머지 {gen_wins_own - gen_top1_for_own}명은 다른 사람의 c1/gen이 더 가까운 경우")