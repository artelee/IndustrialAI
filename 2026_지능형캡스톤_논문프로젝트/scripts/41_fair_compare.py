
"""
공정 비교 실험
Baseline: 갤러리 = c1 실제만
Ours:     갤러리 = 생성 c6만 (c1 기반)
Query:    실제 c6

같은 갤러리 크기 (ID당 1장)
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

device = "cuda"

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 생성 이미지 찾기
GEN_DIRS = [
    f"{PROJECT_DIR}/outputs/clip_gen_c6",
    f"{PROJECT_DIR}/outputs/clip_gen_c6_large",
    f"{PROJECT_DIR}/outputs/realistic_gen",
    f"{PROJECT_DIR}/outputs/gen_for_prof",
]

def find_gen_c6(pid):
    """c6 생성 이미지 찾기"""
    candidates = [
        f"{pid}_gen_c6.png",
        f"{pid}_c6.png",
    ]
    for d in GEN_DIRS:
        for c in candidates:
            p = f"{d}/{c}"
            if os.path.exists(p):
                return p
    return None

# 데이터
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

# c1, 생성 c6, c6 query 다 있는 ID
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" not in gallery_by_id[pid]:
        continue
    if "c6" not in query_by_id[pid]:
        continue
    if find_gen_c6(pid) is None:
        continue
    valid_ids.append(pid)

print(f"평가 ID: {len(valid_ids)}명")
print(f"  - c1 실제 있음")
print(f"  - 생성 c6 있음")
print(f"  - c6 query 있음")

if len(valid_ids) < 30:
    print(f"\n⚠️  유효 ID 너무 적음. 생성 c6 더 필요.")
    print("생성 디렉토리 상태:")
    for d in GEN_DIRS:
        if os.path.exists(d):
            n = len(glob.glob(f"{d}/*"))
            print(f"  {d}: {n}장")

# CLIP-ReID
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
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 갤러리 구성
print("\n갤러리 feature 추출...")

# Baseline: c1 실제
baseline_feats = []
ids = []
for pid in tqdm(valid_ids, desc="Baseline (c1)"):
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    baseline_feats.append(feat(c1_path, cam_to_id["c1"]))
    ids.append(pid)
baseline_feats = np.array(baseline_feats)

# Ours: 생성 c6
ours_feats = []
for pid in tqdm(valid_ids, desc="Ours (생성 c6)"):
    gen_path = find_gen_c6(pid)
    ours_feats.append(feat(gen_path, cam_to_id["c6"]))
ours_feats = np.array(ours_feats)

# Query: 실제 c6
print("\nQuery feature 추출...")
query_feats = []
query_ids = []
for pid in tqdm(valid_ids):
    q_path = sorted(query_by_id[pid]["c6"])[0]
    query_feats.append(feat(q_path, cam_to_id["c6"]))
    query_ids.append(pid)
query_feats = np.array(query_feats)

# 평가
def eval_rank1(gf, gids, qf, qids):
    sims = qf @ gf.T
    correct = 0
    for i, q_pid in enumerate(qids):
        top1 = sims[i].argmax()
        if gids[top1] == q_pid:
            correct += 1
    return correct / len(qids)

r1_base = eval_rank1(baseline_feats, ids, query_feats, query_ids)
r1_ours = eval_rank1(ours_feats, ids, query_feats, query_ids)

# 평균 sim 분석
def avg_self_sim(gf, qf):
    """각 query와 자기 ID 갤러리의 평균 sim"""
    return np.mean([qf[i] @ gf[i] for i in range(len(qf))])

avg_base = avg_self_sim(baseline_feats, query_feats)
avg_ours = avg_self_sim(ours_feats, query_feats)

print("\n" + "="*65)
print("공정 비교 실험 결과")
print("="*65)
print(f"평가 ID: {len(valid_ids)}명")
print(f"갤러리 크기: 각 ID 1장씩 ({len(valid_ids)}장)")
print()
print(f"{'':<35}{'Rank-1':<15}{'self-sim 평균':<15}")
print(f"{'Baseline (c1 실제)':<35}{r1_base*100:<15.2f}{avg_base:<15.4f}")
print(f"{'Ours (생성 c6)':<35}{r1_ours*100:<15.2f}{avg_ours:<15.4f}")
print(f"{'향상':<35}{(r1_ours-r1_base)*100:<+15.2f}{avg_ours-avg_base:<+15.4f}")

print(f"\n[해석]")
if r1_ours > r1_base:
    print(f"✅ 생성 c6이 c1보다 query c6와 더 잘 매칭")
    print(f"   시점 일치 효과 입증")
else:
    print(f"❌ c1 실제가 생성 c6보다 매칭 더 잘 됨")
    print(f"   시점 일치보다 실제 데이터의 가치가 큼")
