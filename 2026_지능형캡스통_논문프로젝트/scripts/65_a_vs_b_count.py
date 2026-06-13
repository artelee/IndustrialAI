
"""
가설 A vs B 직접 비교 (개수로)
- Baseline 못 맞춘 개수
- A가 회복한 개수
- B가 회복한 개수
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

def eval_both(c1_feats, gen_feats, query_feats):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_correct = s1.argmax(axis=1) == np.arange(N)
    hard_mask = ~baseline_correct
    
    # 가설 A: Hard case에 Re-ranking (Top-K 내)
    a_recovered = np.zeros(N, dtype=bool)
    for i in range(N):
        if not hard_mask[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        if topk_idx[final.argmax()] == i:
            a_recovered[i] = True
    
    # 가설 B: Hard case에 Gallery Expansion (c1 + 생성 c?, 2N장)
    b_recovered = np.zeros(N, dtype=bool)
    extended_gallery = np.concatenate([c1_feats, gen_feats], axis=0)  # (2N, dim)
    s_ext = query_feats @ extended_gallery.T
    
    for i in range(N):
        if not hard_mask[i]: continue
        top1_ext = s_ext[i].argmax()
        ours_id = top1_ext if top1_ext < N else top1_ext - N
        if ours_id == i:
            b_recovered[i] = True
    
    return {
        'N': N,
        'hard': hard_mask.sum(),
        'A_recovered': a_recovered.sum(),
        'B_recovered': b_recovered.sum(),
        'A_only': (a_recovered & ~b_recovered).sum(),
        'B_only': (b_recovered & ~a_recovered).sum(),
        'both': (a_recovered & b_recovered).sum(),
    }

print("\n" + "="*110)
print(f"가설 A vs B 회복 개수 비교 (Hard case 대상)")
print(f"Cross-Domain: Duke 학습 → Market 평가")
print("="*110)
print(f"\n{'Pair':<10}{'전체':<8}{'못맞춤':<10}{'A 회복':<10}{'B 회복':<10}{'A만':<8}{'B만':<8}{'둘다':<8}")
print("-"*110)

total = defaultdict(int)
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    r = eval_both(c1_f, gen_f, q_f)
    print(f"c1→{cam:<7}{r['N']:<8}{r['hard']:<10}{r['A_recovered']:<10}"
          f"{r['B_recovered']:<10}{r['A_only']:<8}{r['B_only']:<8}{r['both']:<8}")
    for k in r:
        total[k] += r[k]

print("-"*110)
print(f"{'합계':<10}{total['N']:<8}{total['hard']:<10}{total['A_recovered']:<10}"
      f"{total['B_recovered']:<10}{total['A_only']:<8}{total['B_only']:<8}{total['both']:<8}")

print(f"\n[요약]")
print(f"전체 query:              {total['N']}개")
print(f"Baseline 못 맞춘 것:      {total['hard']}개")
print(f"")
print(f"가설 A (Re-ranking) 회복: {total['A_recovered']}개")
print(f"가설 B (Expansion) 회복:  {total['B_recovered']}개")
print(f"")
print(f"A만 회복:                 {total['A_only']}개")
print(f"B만 회복:                 {total['B_only']}개")
print(f"둘 다 회복:               {total['both']}개")

# 차이
diff = total['B_recovered'] - total['A_recovered']
if diff > 0:
    print(f"\n→ B가 A보다 {diff}개 더 회복")
elif diff < 0:
    print(f"\n→ A가 B보다 {-diff}개 더 회복")
else:
    print(f"\n→ A와 B 동일")
