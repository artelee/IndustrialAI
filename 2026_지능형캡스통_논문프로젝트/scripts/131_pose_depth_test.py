#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
131_pose_depth_test.py ─ skeleton-only vs skeleton+depth 생성 비교 (5명)

가설: OpenPose skeleton만으로는 자세 정보가 부족해 외형이 깨짐.
     depth map을 ControlNet에 추가하면 자세/체형이 더 정확히 잡혀 외형 보존 ↑ (Pose-dIVE 방식).

비교: 같은 5명, 같은 IPA(0.8)에서
  A) skeleton only (현재)
  B) skeleton + depth (멀티 ControlNet)
→ 생성본↔정답 / 생성본↔쿼리 cosine 측정. B가 높으면 depth 추가 효과.
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector, MidasDetector
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"
OUT=f"{PROJECT_DIR}/outputs/pose_depth_test"; os.makedirs(OUT,exist_ok=True)
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
device,dtype="cuda",torch.float16
SRC_CAM="c1"; N=5
IPA_SCALE=0.8
POSE_SCALE=1.0
DEPTH_SCALE=0.6   # depth 보조 (너무 강하면 자세 무시)
PROMPT="a photo of a person, full body shot, standing, surveillance CCTV camera, top-down angle, photorealistic, correct anatomy"
NEG="blurry, low quality, deformed, multiple people, extra limbs, missing limbs, bad anatomy, bad proportions, disfigured, mutated"

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]

print("1. 데이터 로드 + 5명 선정...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
ids=[p for p in sorted(qby) if "c6" in qby[p] and SRC_CAM in gby[p]][:N]
print(f"   선정: {ids}")

print("2. OpenPose + Midas(depth) 추출기 로드...")
op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
midas=MidasDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
def to_img(o):
    if isinstance(o,tuple): o=o[0]
    if o is None: return None
    if not isinstance(o,Image.Image): o=Image.fromarray(o)
    return o
poses={}; depths={}
for pid in ids:
    src=Image.open(qby[pid]["c6"][0]).convert("RGB").resize((512,768),Image.LANCZOS)
    poses[pid]=to_img(op(src))
    if poses[pid]: poses[pid]=poses[pid].resize((512,768),Image.LANCZOS)
    depths[pid]=to_img(midas(src))
    if depths[pid]: depths[pid]=depths[pid].resize((512,768),Image.LANCZOS)

print("3. CLIP-ReID 로드...")
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
rtf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
@torch.no_grad()
def feat_img(img):
    t=rtf(img.convert("RGB")).unsqueeze(0).to(device)
    f=_r(t,cam_label=None)
    if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f=f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()
def feat_path(p): return feat_img(Image.open(p))
c1_real={pid:feat_path(gby[pid][SRC_CAM][0]) for pid in ids}
c6_query={pid:feat_path(qby[pid]["c6"][0]) for pid in ids}

# ===== 파이프라인 A: skeleton only =====
def build_pipe(controlnets):
    pipe=StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=controlnets,
        cache_dir=CACHE_DIR,torch_dtype=dtype,safety_checker=None,requires_safety_checker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IPA_SCALE)
    return pipe.to(device)

print("4. ControlNet 모델 로드...")
cn_pose=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
cn_depth=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",cache_dir=CACHE_DIR,torch_dtype=dtype)

resA=[]; resB=[]
# A) skeleton only
print("5-A. skeleton only 생성...")
pipeA=build_pipe(cn_pose)
for pid in ids:
    if poses[pid] is None: continue
    c1=Image.open(gby[pid][SRC_CAM][0]).convert("RGB").resize((512,1024),Image.LANCZOS)
    g=torch.Generator(device).manual_seed(42)
    gen=pipeA(prompt=PROMPT,negative_prompt=NEG,image=poses[pid],ip_adapter_image=c1,
              controlnet_conditioning_scale=POSE_SCALE,num_inference_steps=30,guidance_scale=7.5,
              width=512,height=768,generator=g).images[0]
    gen.save(f"{OUT}/{pid}_A_skeleton.png")
    gf=feat_img(gen); resA.append((float(c1_real[pid]@gf),float(c6_query[pid]@gf)))
del pipeA; torch.cuda.empty_cache()

# B) skeleton + depth
print("5-B. skeleton + depth 생성...")
pipeB=build_pipe([cn_pose,cn_depth])
for pid in ids:
    if poses[pid] is None or depths[pid] is None: continue
    c1=Image.open(gby[pid][SRC_CAM][0]).convert("RGB").resize((512,1024),Image.LANCZOS)
    g=torch.Generator(device).manual_seed(42)
    gen=pipeB(prompt=PROMPT,negative_prompt=NEG,image=[poses[pid],depths[pid]],ip_adapter_image=c1,
              controlnet_conditioning_scale=[POSE_SCALE,DEPTH_SCALE],num_inference_steps=30,guidance_scale=7.5,
              width=512,height=768,generator=g).images[0]
    gen.save(f"{OUT}/{pid}_B_skeleton_depth.png")
    gf=feat_img(gen); resB.append((float(c1_real[pid]@gf),float(c6_query[pid]@gf)))
del pipeB; torch.cuda.empty_cache()

# ===== 결과 =====
base_q=np.mean([c1_real[p]@c6_query[p] for p in ids])
print("\n"+"="*56)
print("생성 방식별 cosine (5명 평균)")
print("="*56)
print(f"{'방식':<26}{'생성↔정답':<12}{'생성↔쿼리':<12}")
print("-"*56)
print(f"{'[기준] 정답↔쿼리':<26}{'-':<12}{base_q:<12.3f}")
print("-"*56)
print(f"{'A) skeleton only':<26}{np.mean([v[0] for v in resA]):<12.3f}{np.mean([v[1] for v in resA]):<12.3f}")
print(f"{'B) skeleton + depth':<26}{np.mean([v[0] for v in resB]):<12.3f}{np.mean([v[1] for v in resB]):<12.3f}")
print("-"*56)
print(f"B > A 이면 depth 추가가 외형 보존 개선. 목표선={base_q:.3f}")
print(f"이미지: {OUT}/ (A/B 자세·외형 눈으로 비교)")