"""
Stage-wise Cascade Matching 효과 검증

3가지 매칭 방식 비교:
1. Baseline: c1 갤러리만 사용
2. 단순 확장: c1 + 생성 c6 모두 비교
3. Stage-wise: Stage 1(생성 c6 매칭) → Stage 2(c1 검증)
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchreid
import torchvision.transforms as T

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
GEN_DIR = f"{PROJECT_DIR}/outputs/clip_gen_c6"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

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

# 50명 선택
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    cams = set(gallery_by_id[pid].keys())
    if cams >= {"c1","c2","c3","c4","c5","c6"} and "c6" in query_by_id[pid]:
        if os.path.exists(f"{GEN_DIR}/{pid}_gen_c6.png"):
            valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

print(f"평가 ID: {len(valid_ids)}명\n")

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# OSNet
osnet = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
torchreid.utils.load_pretrained_weights(osnet, OSNET_WEIGHTS)
osnet = osnet.eval().to(device)
osnet_tf = T.Compose([
    T.Resize((256,128)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@torch.no_grad()
def clip_feat(p):
    img = Image.open(p).convert("RGB") if isinstance(p, str) else p
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inp)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

@torch.no_grad()
def osnet_feat(p):
    img = Image.open(p).convert("RGB") if isinstance(p, str) else p
    t = osnet_tf(img).unsqueeze(0).to(device)
    f = osnet(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def evaluate(feat_fn, name, top_k=5):
    print(f"\n{'='*60}")
    print(f"[{name} feature]")
    print(f"{'='*60}")

    # 갤러리 구성
    gen_gallery = []  # 50명의 생성 c6
    c1_gallery = []   # 50명의 c1 원본
    for pid in valid_ids:
        gen_gallery.append(feat_fn(f"{GEN_DIR}/{pid}_gen_c6.png"))
        c1_gallery.append(feat_fn(sorted(gallery_by_id[pid]["c1"])[0]))
    gen_gallery = np.array(gen_gallery)
    c1_gallery = np.array(c1_gallery)

    # 평가
    correct_baseline = 0   # c1만
    correct_expanded = 0   # c1 + 생성 (단순 확장)
    correct_stage    = 0   # Stage-wise (Stage1+Stage2)
    correct_stage1   = 0   # Stage 1만 (생성 c6 매칭)

    stage1_hit_in_topk = 0  # Stage 1에서 정답이 Top-K에 포함되는 비율

    for i, pid in enumerate(valid_ids):
        q_feat = feat_fn(sorted(query_by_id[pid]["c6"])[0])

        # 방식 1: Baseline (c1만)
        sims_c1 = q_feat @ c1_gallery.T
        if valid_ids[sims_c1.argmax()] == pid:
            correct_baseline += 1

        # 방식 2: 단순 확장 (c1 + 생성)
        all_sims = np.concatenate([sims_c1, q_feat @ gen_gallery.T])
        all_ids = valid_ids + valid_ids
        if all_ids[all_sims.argmax()] == pid:
            correct_expanded += 1

        # 방식 3: Stage-wise
        # Stage 1: 생성 c6 갤러리 매칭
        sims_gen = q_feat @ gen_gallery.T
        topk_gen_idx = np.argsort(-sims_gen)[:top_k]
        topk_ids = [valid_ids[j] for j in topk_gen_idx]

        # Stage 1 정확도
        if topk_ids[0] == pid:
            correct_stage1 += 1
        if pid in topk_ids:
            stage1_hit_in_topk += 1

        # Stage 2: Top-K 중 c1과 가장 가까운 것
        if topk_ids:
            c1_sims_topk = [sims_c1[valid_ids.index(tid)] for tid in topk_ids]
            best = topk_ids[np.argmax(c1_sims_topk)]
            if best == pid:
                correct_stage += 1

    n = len(valid_ids)
    print(f"  방식 1 (Baseline, c1만):           {correct_baseline}/{n} = {100*correct_baseline/n:5.1f}%")
    print(f"  방식 2 (단순 확장, c1+생성):       {correct_expanded}/{n} = {100*correct_expanded/n:5.1f}%")
    print(f"  방식 3 (Stage-wise, K={top_k}):       {correct_stage}/{n} = {100*correct_stage/n:5.1f}%")
    print(f"    └ Stage 1 only (생성 매칭 Top-1): {correct_stage1}/{n} = {100*correct_stage1/n:5.1f}%")
    print(f"    └ Stage 1 정답 Top-{top_k} 포함율:    {stage1_hit_in_topk}/{n} = {100*stage1_hit_in_topk/n:5.1f}%")

evaluate(clip_feat, "CLIP")
evaluate(osnet_feat, "OSNet")
