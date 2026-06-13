"""
교수님 설계 실험

Baseline: 갤러리 = c1 실제만, Query = c2~c6 실제
Ours:     갤러리 = c1 실제 + c2~c6 생성, Query = c2~c6 실제
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
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
GEN_DIR = f"{PROJECT_DIR}/outputs/gen_for_prof"
os.makedirs(GEN_DIR, exist_ok=True)

device = "cuda"

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터 정리
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

# c1에 갤러리 있고 c2~c6 중 어디든 query 있는 ID
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" not in gallery_by_id[pid]:
        continue
    has_query = any(c in query_by_id[pid] for c in ["c2","c3","c4","c5","c6"])
    if has_query:
        valid_ids.append(pid)

print(f"평가 가능 ID: {len(valid_ids)}명")

# CLIP-ReID (Market 학습)
print("\nCLIP-ReID 로드...")
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'market1501'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

@torch.no_grad()
def feat(img_or_path, cam_id):
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# === 생성 이미지 준비 (없으면 생성) ===
# 우리는 이미 일부 생성 이미지 있음. 일단 그거 활용 + 부족하면 메모
EXISTING_GEN_DIRS = [
    f"{PROJECT_DIR}/outputs/clip_gen_c6",
    f"{PROJECT_DIR}/outputs/clip_gen_c6_large",
    f"{PROJECT_DIR}/outputs/standard_gen",
]

def find_gen_image(pid, target_cam):
    """기존 생성 이미지 찾기"""
    candidates = [
        f"{pid}_gen_{target_cam}.png",
        f"{pid}_{target_cam}.png",
        f"{pid}_gen_c6.png" if target_cam == "c6" else None,
    ]
    for d in EXISTING_GEN_DIRS:
        for c in candidates:
            if c is None: continue
            p = f"{d}/{c}"
            if os.path.exists(p):
                return p
    return None

# === 갤러리/Query 구성 ===
print("\n갤러리 feature 추출...")

# Baseline 갤러리: 각 ID의 c1 실제 1장
gallery_baseline_feats = []
gallery_baseline_ids = []
for pid in tqdm(valid_ids, desc="Baseline gallery"):
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    gallery_baseline_feats.append(feat(c1_path, cam_to_id["c1"]))
    gallery_baseline_ids.append(pid)
gallery_baseline_feats = np.array(gallery_baseline_feats)

# Ours 갤러리: c1 실제 + c2~c6 생성 (있는 거만)
print("\nOurs 갤러리 (c1 + 생성) 구성...")
gallery_ours_feats = []
gallery_ours_ids = []
gen_found = {"c2":0, "c3":0, "c4":0, "c5":0, "c6":0}

for pid in tqdm(valid_ids, desc="Ours gallery"):
    # c1 실제
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    gallery_ours_feats.append(feat(c1_path, cam_to_id["c1"]))
    gallery_ours_ids.append(pid)
    
    # c2~c6 생성 (있는 것만 추가)
    for cam in ["c2","c3","c4","c5","c6"]:
        gen_path = find_gen_image(pid, cam)
        if gen_path:
            gallery_ours_feats.append(feat(gen_path, cam_to_id[cam]))
            gallery_ours_ids.append(pid)
            gen_found[cam] += 1

gallery_ours_feats = np.array(gallery_ours_feats)
print(f"\n생성 이미지 발견 통계:")
for cam, n in gen_found.items():
    print(f"  {cam}: {n}장")
print(f"Baseline 갤러리: {len(gallery_baseline_feats)}장")
print(f"Ours 갤러리:     {len(gallery_ours_feats)}장")

# === Query: c2~c6 실제 ===
print("\nQuery feature 추출...")
query_feats = []
query_ids = []
query_cams = []
for pid in tqdm(valid_ids):
    for cam in ["c2","c3","c4","c5","c6"]:
        if cam not in query_by_id[pid]:
            continue
        for q_path in query_by_id[pid][cam]:
            query_feats.append(feat(q_path, cam_to_id[cam]))
            query_ids.append(pid)
            query_cams.append(cam)
query_feats = np.array(query_feats)
print(f"Query: {len(query_feats)}장")

# === 평가 ===
def evaluate(gf, gids, qf, qids):
    sims = qf @ gf.T
    correct = 0
    for i, q_pid in enumerate(qids):
        top1 = sims[i].argmax()
        if gids[top1] == q_pid:
            correct += 1
    return correct / len(qids)

print("\n매칭 평가 중...")
r1_base = evaluate(gallery_baseline_feats, gallery_baseline_ids,
                   query_feats, query_ids)
r1_ours = evaluate(gallery_ours_feats, gallery_ours_ids,
                   query_feats, query_ids)

# mAP도 계산
def evaluate_map(gf, gids, qf, qids):
    sims = qf @ gf.T
    gids = np.array(gids)
    APs = []
    for i, q_pid in enumerate(qids):
        order = np.argsort(-sims[i])
        matches = (gids[order] == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue
        cmc = matches.cumsum()
        tmp = [x/(j+1.0) for j, x in enumerate(cmc)]
        tmp = np.asarray(tmp) * matches
        APs.append(tmp.sum() / matches.sum())
    return np.mean(APs)

mAP_base = evaluate_map(gallery_baseline_feats, gallery_baseline_ids,
                         query_feats, query_ids)
mAP_ours = evaluate_map(gallery_ours_feats, gallery_ours_ids,
                         query_feats, query_ids)

print("\n" + "="*65)
print("교수님 설계 실험 결과")
print("="*65)
print(f"평가 ID: {len(valid_ids)}명, Query: {len(query_feats)}장")
print(f"Baseline 갤러리: {len(gallery_baseline_feats)}장 (c1 실제만)")
print(f"Ours 갤러리:     {len(gallery_ours_feats)}장 (c1 + 생성)")
print()
print(f"{'':<25}{'Rank-1':<15}{'mAP':<15}")
print(f"{'Baseline':<25}{r1_base*100:<15.2f}{mAP_base*100:<15.2f}")
print(f"{'Ours':<25}{r1_ours*100:<15.2f}{mAP_ours*100:<15.2f}")
print(f"{'향상':<25}{(r1_ours-r1_base)*100:<+15.2f}{(mAP_ours-mAP_base)*100:<+15.2f}")