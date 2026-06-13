"""
Cross-Domain 평가
Source: DukeMTMC 학습 CLIP-ReID
Target: Market-1501 평가

Baseline: c1 실제만 갤러리
Ours:     c1 base + c6 자세 생성 (strength 0.4)
Query:    실제 c6
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
GEN_DIR = f"{PROJECT_DIR}/outputs/c1base_gen_c6/s40"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
NUM_IDS = 100

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

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

valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" not in gallery_by_id[pid]: continue
    if "c6" not in query_by_id[pid]: continue
    gen_path = f"{GEN_DIR}/{pid}_gen_c6.png"
    if not os.path.exists(gen_path): continue
    valid_ids.append(pid)
    if len(valid_ids) >= NUM_IDS: break

print(f"평가 ID: {len(valid_ids)}명\n")

# CLIP-ReID Duke 학습 weight
print("CLIP-ReID 로드 (Duke 학습 weight)...")
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'dukemtmcreid'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

# Duke: 702 IDs, 8 cameras
model = make_model(cfg, num_class=702, camera_num=8, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)
print("✅ 로드 완료\n")

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

# Feature 추출
print("Feature 추출...")
baseline_feats, ours_feats, query_feats = [], [], []
for pid in tqdm(valid_ids):
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    gen_path = f"{GEN_DIR}/{pid}_gen_c6.png"
    q_path = sorted(query_by_id[pid]["c6"])[0]
    
    baseline_feats.append(feat(c1_path, cam_to_id["c1"]))
    ours_feats.append(feat(gen_path, cam_to_id["c6"]))
    query_feats.append(feat(q_path, cam_to_id["c6"]))

baseline_feats = np.array(baseline_feats)
ours_feats = np.array(ours_feats)
query_feats = np.array(query_feats)

# 평가
def eval_rank1(gf, qf):
    sims = qf @ gf.T
    return sum(1 for i in range(len(qf)) if sims[i].argmax() == i) / len(qf)

def avg_self_sim(gf, qf):
    return np.mean([qf[i] @ gf[i] for i in range(len(qf))])

r1_b = eval_rank1(baseline_feats, query_feats)
r1_o = eval_rank1(ours_feats, query_feats)
sim_b = avg_self_sim(baseline_feats, query_feats)
sim_o = avg_self_sim(ours_feats, query_feats)

print("\n" + "="*70)
print("Cross-Domain 평가 결과")
print("="*70)
print(f"Source: DukeMTMC 학습 (ViT-CLIP-ReID-SIE-OLP)")
print(f"Target: Market-1501 평가")
print(f"평가 ID: {len(valid_ids)}명, c1 → c6")
print()
print(f"{'설정':<35}{'Rank-1':<15}{'self-sim':<15}")
print(f"{'Baseline (c1 실제)':<35}{r1_b*100:<15.2f}{sim_b:<15.4f}")
print(f"{'Ours (생성 c6, strength 0.4)':<35}{r1_o*100:<15.2f}{sim_o:<15.4f}")
diff = (r1_o - r1_b) * 100
mark = "✅" if r1_o > r1_b else ("=" if r1_o == r1_b else "❌")
print(f"{'향상':<35}{diff:+.2f} {mark}     {sim_o-sim_b:+.4f}")

print(f"\n[비교: Same-Domain]")
print(f"Market 학습 모델로 Market 평가: Baseline 94%, Ours 90% (-4%p)")
print(f"\n[Cross-Domain 기대]")
print(f"Baseline 낮음 (학습 못 본 도메인)")
print(f"Ours 향상 가능성 ↑ (생성 갤러리가 도메인 갭 보완)")
