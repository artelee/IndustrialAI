"""
CLIP-ReID feature로 Stage-wise 매칭 효과 검증
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

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

print(f"평가 ID: {len(valid_ids)}명")

# CLIP-ReID config
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
cfg.TEST.WEIGHT = "/home/ubuntu/reid-gallery-expansion/checkpoints/clipreid/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

print("CLIP-ReID 로드...")
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

@torch.no_grad()
def feat(path_or_pil, cam_id=0):
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    t = transform(img).unsqueeze(0).to(device)
    cam = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=cam)
    f = torch.nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

print("\n갤러리 feature 추출...")
c1_feats = []
gen_feats = []
for i, pid in enumerate(valid_ids):
    c1_feats.append(feat(sorted(gallery_by_id[pid]["c1"])[0], cam_id=0))
    gen_feats.append(feat(f"{GEN_DIR}/{pid}_gen_c6.png", cam_id=5))
c1_feats = np.array(c1_feats)
gen_feats = np.array(gen_feats)

print("매칭 평가...")
TOP_K = 5
correct_baseline = 0
correct_expanded = 0
correct_stage = 0
correct_stage1 = 0
stage1_hit_topk = 0

for i, pid in enumerate(valid_ids):
    q_feat = feat(sorted(query_by_id[pid]["c6"])[0], cam_id=5)
    sims_c1 = q_feat @ c1_feats.T
    sims_gen = q_feat @ gen_feats.T

    if valid_ids[sims_c1.argmax()] == pid:
        correct_baseline += 1

    all_sims = np.concatenate([sims_c1, sims_gen])
    all_ids = valid_ids + valid_ids
    if all_ids[all_sims.argmax()] == pid:
        correct_expanded += 1

    topk = np.argsort(-sims_gen)[:TOP_K]
    topk_ids = [valid_ids[j] for j in topk]
    if topk_ids[0] == pid:
        correct_stage1 += 1
    if pid in topk_ids:
        stage1_hit_topk += 1

    c1_topk = [sims_c1[valid_ids.index(tid)] for tid in topk_ids]
    if topk_ids[np.argmax(c1_topk)] == pid:
        correct_stage += 1

n = len(valid_ids)
print("\n" + "="*65)
print(f"CLIP-ReID feature 평가 (50명, c1->c6)")
print("="*65)
print(f"방식 1 (Baseline, c1만):       {correct_baseline}/{n} = {100*correct_baseline/n:.1f}%")
print(f"방식 2 (단순 확장):            {correct_expanded}/{n} = {100*correct_expanded/n:.1f}%")
print(f"방식 3 (Stage-wise, K={TOP_K}):    {correct_stage}/{n} = {100*correct_stage/n:.1f}%")
print(f"  Stage 1 only:               {correct_stage1}/{n} = {100*correct_stage1/n:.1f}%")
print(f"  Stage 1 Top-{TOP_K} 포함율:     {stage1_hit_topk}/{n} = {100*stage1_hit_topk/n:.1f}%")
print(f"\n→ Stage 효과: {(correct_stage-correct_baseline)/n*100:+.1f}%p")
