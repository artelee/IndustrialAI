#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
121_cctv_degrade_test.py ─ 생성본 CCTV 열화 효과 (5명 테스트)

가설: 생성본이 너무 깨끗해서 C6 query와 도메인 갭. CCTV 화질로 열화하면 query에 가까워질까?

5명에 대해 cosine 비교:
  C6 query ↔ C1 real (정답, 목표)
  C6 query ↔ 생성본 (깨끗, 현재)
  C6 query ↔ 생성본+CCTV열화 (제안)
→ 열화본이 깨끗한 생성본보다 query에 가까우면 전체 실험 가치 있음
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import cv2
from skimage.exposure import match_histograms
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"; GEN_ROOT=f"{PROJECT_DIR}/outputs/allcam_to_c6"
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
OUT=f"{PROJECT_DIR}/outputs/degrade_test"; os.makedirs(OUT,exist_ok=True)
device="cuda"; W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
SRC_CAM="c1"  # 소스 카메라 (생성본 있는 곳)
N=5

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]

def make_it_look_like_cctv(gen_pil, ref_pil):
    # 과격 버전 (히스토그램매칭 + 64x128 + JPEG35)
    img=cv2.cvtColor(np.array(gen_pil),cv2.COLOR_RGB2BGR)
    ref=cv2.cvtColor(np.array(ref_pil),cv2.COLOR_RGB2BGR)
    h,w=img.shape[:2]
    ref_r=cv2.resize(ref,(w,h))
    matched=match_histograms(img,ref_r,channel_axis=-1).astype('uint8')
    small=cv2.resize(matched,(64,128),interpolation=cv2.INTER_AREA)
    _,enc=cv2.imencode('.jpg',small,[int(cv2.IMWRITE_JPEG_QUALITY),35])
    noisy=cv2.imdecode(enc,1)
    deg=cv2.resize(noisy,(w,h),interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(deg,cv2.COLOR_BGR2RGB))

def make_it_look_like_cctv_mild(gen_pil, ref_pil=None):
    # 부드러운 버전 (128x256 + 약블러 + JPEG75, 색감왜곡 없음)
    img=cv2.cvtColor(np.array(gen_pil),cv2.COLOR_RGB2BGR)
    h,w=img.shape[:2]
    small=cv2.resize(img,(128,256),interpolation=cv2.INTER_AREA)
    blurred=cv2.GaussianBlur(small,(3,3),0)
    _,enc=cv2.imencode('.jpg',blurred,[int(cv2.IMWRITE_JPEG_QUALITY),75])
    noisy=cv2.imdecode(enc,1)
    deg=cv2.resize(noisy,(w,h),interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(deg,cv2.COLOR_BGR2RGB))

print("데이터 로드...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
# 5명: C6 query + C1 gallery + 생성본 있는 사람
ids=[p for p in sorted(qby) if "c6" in qby[p] and SRC_CAM in gby[p]
     and os.path.exists(f"{GEN_ROOT}/{SRC_CAM}/pose0/{p}_gen.png")][:N]
print(f"  테스트 {len(ids)}명: {ids}")

print("CLIP-ReID 로드...")
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
def feat_img(img):
    t=tf(img.convert("RGB")).unsqueeze(0).to(device)
    f=_r(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()
def feat_path(p): return feat_img(Image.open(p))

def l2(v): return v/(np.linalg.norm(v)+1e-12)

# 먼저 도메인 갭 벡터 계산용: 전체 5명의 real/gen 평균
all_real=[]; all_gen=[]
for pid in ids:
    all_real.append(feat_path(gby[pid][SRC_CAM][0]))
    for k in range(3):
        gp=f"{GEN_ROOT}/{SRC_CAM}/pose{k}/{pid}_gen.png"
        if os.path.exists(gp): all_gen.append(feat_img(Image.open(gp)))
mean_real=np.mean(all_real,axis=0)
mean_gen=np.mean(all_gen,axis=0)
domain_gap=mean_real-mean_gen   # AI→CCTV 방향 벡터

print(f"\n{'PID':<8}{'정답(C1r)':<11}{'생성(깨끗)':<12}{'열화(부드)':<12}{'특징보정':<12}")
print("-"*55)
clean_all=[]; mild_all=[]; calib_all=[]; real_all=[]
for pid in ids:
    c6q=feat_path(qby[pid]["c6"][0])
    c1r=feat_path(gby[pid][SRC_CAM][0])
    cleans=[]; milds=[]; calibs=[]
    for k in range(3):
        gp=f"{GEN_ROOT}/{SRC_CAM}/pose{k}/{pid}_gen.png"
        if not os.path.exists(gp): continue
        gen=Image.open(gp)
        gf=feat_img(gen)
        cleans.append(gf)
        milds.append(feat_img(make_it_look_like_cctv_mild(gen)))
        calibs.append(l2(gf+domain_gap))    # 피드백 공식: gen + (μreal−μgen)
    s_real=float(c6q@c1r)
    s_clean=float(c6q@l2(np.mean(cleans,axis=0)))
    s_mild=float(c6q@l2(np.mean(milds,axis=0)))
    s_calib=float(c6q@l2(np.mean(calibs,axis=0)))
    real_all.append(s_real); clean_all.append(s_clean); mild_all.append(s_mild); calib_all.append(s_calib)
    print(f"{pid:<8}{s_real:<11.3f}{s_clean:<12.3f}{s_mild:<12.3f}{s_calib:<12.3f}")
print("-"*55)
print(f"{'평균':<8}{np.mean(real_all):<11.3f}{np.mean(clean_all):<12.3f}{np.mean(mild_all):<12.3f}{np.mean(calib_all):<12.3f}")
print("\n해석:")
print(f"  정답(C1r)={np.mean(real_all):.3f} ← 생성본이 이걸 넘어야 융합에 도움")
print(f"  특징보정(domain gap)={np.mean(calib_all):.3f} > 깨끗({np.mean(clean_all):.3f}) 이면 → feature 보정 효과")
print(f"  단, 정답({np.mean(real_all):.3f})을 넘는지가 핵심")