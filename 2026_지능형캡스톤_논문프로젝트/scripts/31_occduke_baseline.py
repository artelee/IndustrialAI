"""
Occluded-DukeMTMC CLIP-ReID Baseline (Occ-Duke 학습된 weight 사용)
표준 mAP/Rank-K 평가
"""

import os, sys, glob, torch, numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
GALLERY_DIR = f"{OCC_DIR}/bounding_box_test"
QUERY_DIR = f"{OCC_DIR}/query"
device = "cuda"

def parse(f):
    name = os.path.basename(f).split(".")[0]
    parts = name.split("_")
    return parts[0], parts[1]  # pid, cam

# 데이터
print("데이터 로드...")
gallery_files = sorted(glob.glob(f"{GALLERY_DIR}/*.jpg"))
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

# 카메라 ID 매핑 (c1~c8)
all_cams = set()
for f in gallery_files + query_files:
    _, cam = parse(f)
    all_cams.add(cam)
cam_list = sorted(all_cams)
cam_to_id = {c: i for i, c in enumerate(cam_list)}
print(f"  카메라: {cam_list}")
print(f"  Gallery: {len(gallery_files)}장")
print(f"  Query: {len(query_files)}장")

# Train ID 수 확인 (num_class용)
train_ids = set()
for f in glob.glob(f"{OCC_DIR}/bounding_box_train/*.jpg"):
    pid, _ = parse(f)
    if pid not in ('-1', '0000'):
        train_ids.add(pid)
NUM_CLASSES = len(train_ids)
print(f"  Train ID 수: {NUM_CLASSES}")

# CLIP-ReID 로드
print("\nCLIP-ReID (Occ-Duke 학습) 로드...")
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'occ_duke'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid_occduke/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

CAMERA_NUM = len(cam_list)
model = make_model(cfg, num_class=NUM_CLASSES, camera_num=CAMERA_NUM, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)
print("✅ 로드 완료")

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

@torch.no_grad()
def extract(files, desc):
    feats, pids, cams = [], [], []
    for f in tqdm(files, desc=desc):
        pid, cam = parse(f)
        img = Image.open(f).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        c = torch.tensor([cam_to_id[cam]]).to(device)
        feat = model(t, cam_label=c)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        feats.append(feat.cpu().numpy().flatten())
        pids.append(pid)
        cams.append(cam)
    return np.array(feats), np.array(pids), np.array(cams)

print("\n갤러리 feature 추출...")
gf, gp, gc = extract(gallery_files, "Gallery")
print("쿼리 feature 추출...")
qf, qp, qc = extract(query_files, "Query")

# 표준 mAP/Rank 평가
print("\n매칭 평가...")
sims = qf @ gf.T
all_cmc, all_AP = [], []

for q_idx in range(len(qp)):
    q_pid, q_cam = qp[q_idx], qc[q_idx]
    
    order = np.argsort(-sims[q_idx])
    keep = ~((gp == q_pid) & (gc == q_cam))
    keep[gp == "0000"] = False
    keep[gp == "-1"] = False
    order_valid = order[keep[order]]
    
    matches = (gp[order_valid] == q_pid).astype(np.int32)
    if matches.sum() == 0:
        continue
    
    cmc = matches.cumsum()
    cmc[cmc > 1] = 1
    all_cmc.append(cmc[:50])
    
    num_rel = matches.sum()
    tmp = matches.cumsum()
    tmp = [x/(i+1.0) for i, x in enumerate(tmp)]
    tmp = np.asarray(tmp) * matches
    all_AP.append(tmp.sum() / num_rel)

cmc = np.array(all_cmc).mean(axis=0)
mAP = np.mean(all_AP)

print("\n" + "="*60)
print("Occluded-DukeMTMC CLIP-ReID Baseline")
print("="*60)
print(f"mAP:     {mAP*100:.2f}%")
print(f"Rank-1:  {cmc[0]*100:.2f}%")
print(f"Rank-5:  {cmc[4]*100:.2f}%")
print(f"Rank-10: {cmc[9]*100:.2f}%")
print("="*60)
print(f"\n[참고 - 학계 SOTA 기준]")
print(f"CLIP-ReID 공식: 약 mAP 60% 수준")
print(f"ProFD (CLIP 기반 SOTA): mAP 62.8%, Rank-1 70.8%")