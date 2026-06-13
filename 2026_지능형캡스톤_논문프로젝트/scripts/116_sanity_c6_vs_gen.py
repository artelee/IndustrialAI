#!/usr/bin/env python
"""C6 query ↔ {C1 real, 생성본} cosine 직접 측정 — 진짜 매칭 가능성 확인"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_DIR=f"{PROJECT_DIR}/outputs/c1_multipose_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
device="cuda"; W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
ids=[p for p in sorted(qby) if "c6" in qby[p] and "c1" in gby[p]][:5]

sys.path.insert(0,"/home/ubuntu/CLIP-ReID")
from config import cfg as rc
from model.make_model_clipreid import make_model
rc.MODEL.NAME="ViT-B-16"; rc.MODEL.STRIDE_SIZE=[16,16]; rc.MODEL.SIE_CAMERA=False
rc.MODEL.SIE_COE=0.0; rc.MODEL.ID_LOSS_TYPE="softmax"
rc.INPUT.SIZE_TRAIN=[256,128]; rc.INPUT.SIZE_TEST=[256,128]
rc.INPUT.PIXEL_MEAN=[0.5]*3; rc.INPUT.PIXEL_STD=[0.5]*3
rc.DATASETS.NAMES="market1501"; rc.TEST.WEIGHT=W; rc.TEST.NECK_FEAT="before"
try: _r=make_model(rc,num_class=702,camera_num=0,view_num=1)
except Exception: _r=make_model(rc,num_class=702,camera_num=6,view_num=1)
_r.load_param(W); _r=_r.to(device).eval()
tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat(p):
    t=tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
    f=_r(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()

print(f"{'PID':<8}{'C6q↔C1r':<12}{'C6q↔gen0':<12}{'C6q↔gen1':<12}{'C6q↔gen2':<12}{'C6q↔임의C1':<12}")
print("-"*68)
all_c6=[]; all_other_c1=[]
for pid in ids:
    c6q=feat(qby[pid]["c6"][0])
    c1r=feat(gby[pid]["c1"][0])
    gens=[feat(f"{GEN_DIR}/raw/pose{k}/{pid}_gen.png") for k in range(3)
          if os.path.exists(f"{GEN_DIR}/raw/pose{k}/{pid}_gen.png")]
    # 임의 다른 사람 c1 (distractor 참고)
    other_pid=[p for p in sorted(gby) if p!=pid and "c1" in gby[p]][0]
    other=feat(gby[other_pid]["c1"][0])
    s_real=float(c6q@c1r); s_gens=[float(c6q@g) for g in gens]; s_other=float(c6q@other)
    print(f"{pid:<8}{s_real:<12.3f}"+"".join(f"{s:<12.3f}" for s in s_gens)+
          " "*(12*(3-len(s_gens))) + f"{s_other:<12.3f}")
print("\n해석:")
print("  C6q↔C1r (정답): cross-camera 정답이라 0.5~0.8 정도 기대")
print("  C6q↔gen   (생성): 정답보다 높으면 → 매칭에 도움. 낮으면 → 도움 안 됨")
print("  C6q↔임의C1 (오답): distractor 기준선. 정답보다 낮아야")