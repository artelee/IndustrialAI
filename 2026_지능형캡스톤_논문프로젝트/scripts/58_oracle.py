"""
Oracle 분석
Baseline 틀린 케이스만 정확히 골라서 Re-rank
"이 방법의 잠재력은?" 측정
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

# Feature 추출
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

# Oracle 평가
def eval_oracle(c1_feats, gen_feats, query_feats):
    """Oracle: Baseline 틀린 것만 Re-rank, 맞은 건 그대로"""
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    # 통계
    n_hard = (~baseline_correct).sum()
    n_easy = baseline_correct.sum()
    
    # Hard case에 Re-rank 적용
    ours_correct = baseline_correct.copy()
    recovered = 0
    in_topk = 0
    for i in range(N):
        if baseline_correct[i]: continue  # Easy 그대로
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        if i in topk_idx:
            in_topk += 1
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        ours_top1 = topk_idx[final.argmax()]
        if ours_top1 == i:
            ours_correct[i] = True
            recovered += 1
    
    return {
        'N': N,
        'baseline_correct': n_easy,
        'baseline_wrong': n_hard,
        'in_topk': in_topk,
        'recovered': recovered,
        'ours_correct': ours_correct.sum(),
        'r1_baseline': n_easy / N * 100,
        'r1_oracle': ours_correct.sum() / N * 100,
    }

# 평가
print("\n" + "="*100)
print(f"Oracle 분석 (Hard case에만 Re-rank, Easy는 그대로)")
print(f"Top-K={TOP_K}, alpha={ALPHA}")
print("="*100)
print(f"\n{'Pair':<10}{'N':<6}{'BL맞':<8}{'BL틀':<8}{'TopK내':<10}{'회복':<8}{'Oracle맞':<12}{'Baseline':<12}{'Oracle':<12}{'향상':<10}")
print("-"*100)

total = defaultdict(int)
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    r = eval_oracle(c1_f, gen_f, q_f)
    gain = r['r1_oracle'] - r['r1_baseline']
    mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
    print(f"c1→{cam:<7}{r['N']:<6}{r['baseline_correct']:<8}{r['baseline_wrong']:<8}"
          f"{r['in_topk']:<10}{r['recovered']:<8}{r['ours_correct']:<12}"
          f"{r['r1_baseline']:<12.2f}{r['r1_oracle']:<12.2f}{gain:+.2f} {mark}")
    for k in ['N','baseline_correct','baseline_wrong','in_topk','recovered','ours_correct']:
        total[k] += r[k]

avg_baseline = total['baseline_correct'] / total['N'] * 100
avg_oracle = total['ours_correct'] / total['N'] * 100
print("-"*100)
print(f"{'합계':<10}{total['N']:<6}{total['baseline_correct']:<8}{total['baseline_wrong']:<8}"
      f"{total['in_topk']:<10}{total['recovered']:<8}{total['ours_correct']:<12}"
      f"{avg_baseline:<12.2f}{avg_oracle:<12.2f}{avg_oracle-avg_baseline:+.2f}")

# 회복률
recover_rate_topk = total['recovered'] / max(total['in_topk'], 1) * 100
recover_rate_overall = total['recovered'] / max(total['baseline_wrong'], 1) * 100
print(f"\n[회복률]")
print(f"Top-K 내 회복률: {total['recovered']}/{total['in_topk']} = {recover_rate_topk:.1f}%")
print(f"전체 회복률: {total['recovered']}/{total['baseline_wrong']} = {recover_rate_overall:.1f}%")

print(f"\n[Oracle 의미]")
print(f"Baseline Rank-1: {avg_baseline:.2f}%")
print(f"Oracle Rank-1:   {avg_oracle:.2f}%")
print(f"잠재 향상:        {avg_oracle - avg_baseline:+.2f}%p")
print(f"→ 'Baseline 틀린 것만 정확히 골라 Re-rank했다면 +{avg_oracle - avg_baseline:.2f}%p 향상 가능'")
