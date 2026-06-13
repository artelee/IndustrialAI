"""
가설 A (Re-ranking) vs 가설 B (Gallery Expansion) 비교
Hard case에서 어느 게 더 회복?

Duke 학습 → Market 평가 (Cross-Domain)
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
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

print("\nFeature 추출...")
all_feats = {}
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = [], [], []
    for pid in tqdm(cam_valid_ids[cam], desc=cam, leave=False):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
        q_path = sorted(query_by_id[pid][cam])[0]
        c1_f.append(feat(c1_path, cam_to_id["c1"]))
        gen_f.append(feat(gen_path, cam_to_id[cam]))
        q_f.append(feat(q_path, cam_to_id[cam]))
    all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))

# 가설 A: Re-ranking with Top-K
def eval_A(c1_feats, gen_feats, query_feats):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_correct = s1.argmax(axis=1) == np.arange(N)
    
    ours_correct = baseline_correct.copy()
    for i in range(N):
        if baseline_correct[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        if topk_idx[final.argmax()] == i:
            ours_correct[i] = True
    return baseline_correct, ours_correct

# 가설 B: Gallery Expansion
def eval_B(c1_feats, gen_feats, query_feats):
    N = len(query_feats)
    
    # 갤러리 = c1 실제 + 생성 c? (총 2N장)
    # 인덱스 0~N-1: c1 실제 (ID i = 인덱스 i)
    # 인덱스 N~2N-1: 생성 c? (ID i = 인덱스 N+i)
    extended_gallery = np.concatenate([c1_feats, gen_feats], axis=0)  # (2N, dim)
    
    s = query_feats @ extended_gallery.T  # (N, 2N)
    baseline_correct = (query_feats @ c1_feats.T).argmax(axis=1) == np.arange(N)
    
    # Top-1 찾고 ID로 매핑
    ours_correct = np.zeros(N, dtype=bool)
    for i in range(N):
        top1_in_extended = s[i].argmax()
        # 인덱스 → ID
        if top1_in_extended < N:
            ours_id = top1_in_extended
        else:
            ours_id = top1_in_extended - N
        ours_correct[i] = (ours_id == i)
    
    return baseline_correct, ours_correct

# 평가
print("\n" + "="*110)
print(f"가설 A (Re-ranking) vs 가설 B (Gallery Expansion)")
print(f"Cross-Domain: Duke 학습 → Market 평가")
print("="*110)
print(f"\n{'Pair':<10}{'N':<8}{'BL %':<10}{'A: Oracle %':<15}{'A 향상':<10}{'B: Expand %':<15}{'B 향상':<10}{'A vs B':<10}")
print("-"*110)

total = defaultdict(int)
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    bl_c, a_c = eval_A(c1_f, gen_f, q_f)
    _, b_c = eval_B(c1_f, gen_f, q_f)
    
    n = len(q_f)
    bl = bl_c.sum() / n * 100
    a = a_c.sum() / n * 100
    b = b_c.sum() / n * 100
    
    total['N'] += n
    total['BL'] += bl_c.sum()
    total['A'] += a_c.sum()
    total['B'] += b_c.sum()
    
    a_gain = a - bl
    b_gain = b - bl
    diff = b - a
    winner = "B>A" if b > a else ("A>B" if a > b else "=")
    print(f"c1→{cam:<7}{n:<8}{bl:<10.2f}{a:<15.2f}{a_gain:+.2f}    {b:<15.2f}{b_gain:+.2f}    {winner}")

bl_avg = total['BL'] / total['N'] * 100
a_avg = total['A'] / total['N'] * 100
b_avg = total['B'] / total['N'] * 100
print("-"*110)
print(f"{'합계':<10}{total['N']:<8}{bl_avg:<10.2f}{a_avg:<15.2f}{a_avg-bl_avg:+.2f}    {b_avg:<15.2f}{b_avg-bl_avg:+.2f}")

print(f"\n[종합]")
print(f"Baseline:       {bl_avg:.2f}%")
print(f"가설 A (Oracle): {a_avg:.2f}% (+{a_avg-bl_avg:.2f}%p)")
print(f"가설 B (Expand): {b_avg:.2f}% (+{b_avg-bl_avg:.2f}%p)")
print(f"\nA vs B 차이:    {b_avg-a_avg:+.2f}%p")

if b_avg > a_avg:
    print(f"→ 가설 B (Gallery Expansion)가 더 효과적")
elif a_avg > b_avg:
    print(f"→ 가설 A (Re-ranking)가 더 효과적")
else:
    print(f"→ 비슷한 효과")
