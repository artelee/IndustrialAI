"""
Cross-Domain SIE 효과 분석
Duke 학습 → Market 평가

A: SIE 사용 (cam_label = 실제)
B: SIE 안 씀 (cam_label = 0 모두)
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
GEN_BASE = f"{PROJECT_DIR}/outputs/c1base_gen_all"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
TOP_K = 5
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

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
for cam in TARGET_CAMS:
    ids = []
    for pid in sorted(gallery_by_id.keys()):
        if "c1" in gallery_by_id[pid] and cam in query_by_id[pid]:
            gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
            if os.path.exists(gen_path):
                ids.append(pid)
    cam_valid_ids[cam] = ids

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

print("CLIP-ReID Duke 학습 로드...")
model = make_model(cfg, num_class=702, camera_num=8, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def extract_feats(use_sie):
    """use_sie=True: 카메라 정보 활용, False: 모두 cam_label=0"""
    cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}
    all_feats = {}
    for cam in TARGET_CAMS:
        c1_f, gen_f, q_f = [], [], []
        for pid in tqdm(cam_valid_ids[cam], desc=f"{cam} (SIE={use_sie})", leave=False):
            c1_path = sorted(gallery_by_id[pid]["c1"])[0]
            gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
            q_path = sorted(query_by_id[pid][cam])[0]
            if use_sie:
                c1_f.append(feat(c1_path, cam_to_id["c1"]))
                gen_f.append(feat(gen_path, cam_to_id[cam]))
                q_f.append(feat(q_path, cam_to_id[cam]))
            else:
                c1_f.append(feat(c1_path, 0))
                gen_f.append(feat(gen_path, 0))
                q_f.append(feat(q_path, 0))
        all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))
    return all_feats

def eval_oracle(c1_feats, gen_feats, query_feats):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    ours_correct = baseline_correct.copy()
    for i in range(N):
        if baseline_correct[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        if topk_idx[final.argmax()] == i:
            ours_correct[i] = True
    return baseline_correct.sum() / N, ours_correct.sum() / N

print("\n[A] SIE 사용 (cam_label=실제)")
feats_A = extract_feats(use_sie=True)
print("\n[B] SIE 안 씀 (cam_label=0)")
feats_B = extract_feats(use_sie=False)

print("\n" + "="*90)
print("SIE Ablation (Cross-Domain Duke → Market)")
print("="*90)
print(f"{'Pair':<10}{'N':<8}{'A-BL':<10}{'A-Oracle':<10}{'B-BL':<10}{'B-Oracle':<10}")
print("-"*90)

total = defaultdict(int)
for cam in TARGET_CAMS:
    n = len(feats_A[cam][2])
    a_bl, a_or = eval_oracle(*feats_A[cam])
    b_bl, b_or = eval_oracle(*feats_B[cam])
    total['N'] += n
    total['A_BL'] += a_bl * n
    total['A_OR'] += a_or * n
    total['B_BL'] += b_bl * n
    total['B_OR'] += b_or * n
    print(f"c1→{cam:<7}{n:<8}{a_bl*100:<10.2f}{a_or*100:<10.2f}{b_bl*100:<10.2f}{b_or*100:<10.2f}")

a_bl_avg = total['A_BL'] / total['N'] * 100
a_or_avg = total['A_OR'] / total['N'] * 100
b_bl_avg = total['B_BL'] / total['N'] * 100
b_or_avg = total['B_OR'] / total['N'] * 100
print("-"*90)
print(f"{'합계':<10}{total['N']:<8}{a_bl_avg:<10.2f}{a_or_avg:<10.2f}{b_bl_avg:<10.2f}{b_or_avg:<10.2f}")

print(f"\n[해석]")
print(f"A (SIE 활용): Baseline {a_bl_avg:.2f}% → Oracle {a_or_avg:.2f}% (+{a_or_avg-a_bl_avg:.2f}%p)")
print(f"B (SIE 무시): Baseline {b_bl_avg:.2f}% → Oracle {b_or_avg:.2f}% (+{b_or_avg-b_bl_avg:.2f}%p)")
print(f"")
print(f"SIE 효과 (Baseline): {a_bl_avg - b_bl_avg:+.2f}%p")
print(f"SIE 효과 (Oracle):   {a_or_avg - b_or_avg:+.2f}%p")
