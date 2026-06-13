
"""
Hard case 분석
- Baseline Top-1 틀린 케이스만 모음
- Ours가 그 중 몇 개 회복하는지

Cross-Domain: Market 학습 → Duke 평가
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
GEN_BASE = f"{PROJECT_DIR}/outputs/duke_c1base_gen"
DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
GALLERY_DIR = f"{DUKE_DIR}/bounding_box_test"
QUERY_DIR = f"{DUKE_DIR}/query"

device = "cuda"
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6", "c8"]
TOP_K = 5
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터
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

# Market 학습 모델
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

print("CLIP-ReID Market 학습 로드...")
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5,'c7':5,'c8':5}

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def analyze(target_cam, valid_ids):
    """Hard case 분석"""
    c1_feats, gen_feats, query_feats = [], [], []
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(query_by_id[pid][target_cam])[0]
        c1_feats.append(feat(c1_path, cam_to_id["c1"]))
        gen_feats.append(feat(gen_path, cam_to_id[target_cam]))
        query_feats.append(feat(q_path, cam_to_id[target_cam]))
    
    c1_feats = np.array(c1_feats)
    gen_feats = np.array(gen_feats)
    query_feats = np.array(query_feats)
    N = len(query_feats)
    
    s1 = query_feats @ c1_feats.T
    
    # Baseline Top-1
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    # Ours (Re-rank)
    ours_correct = np.zeros(N, dtype=bool)
    in_topk = np.zeros(N, dtype=bool)  # 정답이 Top-K 안에 있나
    
    for i in range(N):
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        in_topk[i] = i in topk_idx
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        ours_top1 = topk_idx[final.argmax()]
        ours_correct[i] = (ours_top1 == i)
    
    # 분석
    easy_mask = baseline_correct          # Baseline 맞춤
    hard_mask = ~baseline_correct          # Baseline 틀림
    hard_in_topk = hard_mask & in_topk     # 틀렸지만 Top-K에 정답 있음
    hard_not_topk = hard_mask & ~in_topk   # 틀렸고 Top-K에도 없음
    
    # 회복/잃음
    recovered = hard_mask & ours_correct           # Hard → Ours가 회복
    lost = easy_mask & ~ours_correct                # Easy → Ours가 잃음
    
    return {
        'N': N,
        'baseline_correct': baseline_correct.sum(),
        'baseline_wrong': hard_mask.sum(),
        'hard_in_topk': hard_in_topk.sum(),
        'hard_not_topk': hard_not_topk.sum(),
        'recovered': recovered.sum(),
        'lost': lost.sum(),
        'ours_correct': ours_correct.sum(),
    }

# 카메라별 분석
print("\n" + "="*100)
print("Hard Case 분석 (Cross-Domain Market → Duke)")
print("="*100)
print(f"\n{'Pair':<10}{'N':<6}{'BL맞':<8}{'BL틀':<8}{'TopK내':<10}{'회복':<8}{'잃음':<8}{'Ours맞':<10}{'회복률':<10}")
print("-"*100)

total = defaultdict(int)
for cam in TARGET_CAMS:
    r = analyze(cam, cam_valid_ids[cam])
    recover_rate = r['recovered'] / max(r['hard_in_topk'], 1) * 100  # Top-K 내 회복률
    overall_recover = r['recovered'] / max(r['baseline_wrong'], 1) * 100  # 전체 회복률
    
    print(f"c1→{cam:<7}{r['N']:<6}{r['baseline_correct']:<8}{r['baseline_wrong']:<8}"
          f"{r['hard_in_topk']:<10}{r['recovered']:<8}{r['lost']:<8}{r['ours_correct']:<10}"
          f"{recover_rate:.1f}%/{overall_recover:.1f}%")
    
    for k in r:
        total[k] += r[k]

print("-"*100)
recover_rate = total['recovered'] / max(total['hard_in_topk'], 1) * 100
overall_recover = total['recovered'] / max(total['baseline_wrong'], 1) * 100
print(f"{'합계':<10}{total['N']:<6}{total['baseline_correct']:<8}{total['baseline_wrong']:<8}"
      f"{total['hard_in_topk']:<10}{total['recovered']:<8}{total['lost']:<8}{total['ours_correct']:<10}"
      f"{recover_rate:.1f}%/{overall_recover:.1f}%")

print(f"\n[해석]")
print(f"- TopK내 회복률 = Top-K에 정답 있던 hard case 중 회복 비율")
print(f"- 전체 회복률 = 전체 hard case 중 회복 비율")
print(f"- 잃음 = Baseline 맞췄는데 Ours가 틀린 케이스")
print(f"")
print(f"순효과: 회복 {total['recovered']} - 잃음 {total['lost']} = {total['recovered'] - total['lost']:+d}")
