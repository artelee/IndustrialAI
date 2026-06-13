#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
132_realvis_test.py ─ 실사모델(Realistic Vision) + 4법칙 생성 테스트 (5명)

4가지 법칙 적용:
  1. 실사 특화 베이스 모델 (Realistic Vision) vs 기존 SD1.5 비교
  2. 해상도 512×768 고정
  3. IPA scale 여러개 (0.5 / 0.8) — 실사모델에선 다를 수 있어 비교
  4. 강한 negative prompt (해부학 오류 방지)

기존(SD1.5, IPA0.8) vs 새방식(RealVis, IPA0.5/0.8) cosine + 눈 비교.
IP-Adapter가 RealVis와 호환 안되면 자동 건너뜀.
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector
import torchvision.transforms as T
import torch.nn as nn

HOME=os.path.expanduser("~"); PROJECT_DIR=f"{HOME}/reid-gallery-expansion"
CACHE_DIR=f"{PROJECT_DIR}/checkpoints"
OUT=f"{PROJECT_DIR}/outputs/realvis_test"; os.makedirs(OUT,exist_ok=True)
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
device,dtype="cuda",torch.float16
SRC_CAM="c1"; N=5

# 테스트 조합: (이름, 베이스모델, IPA scale)
CONFIGS=[
    ("SD15_IPA0.8",   "stable-diffusion-v1-5/stable-diffusion-v1-5", 0.8),  # 기존(대조군)
    ("RealVis_IPA0.5","SG161222/Realistic_Vision_V5.1_noVAE",        0.5),  # 조언대로
    ("RealVis_IPA0.8","SG161222/Realistic_Vision_V5.1_noVAE",        0.8),  # 실사+높은IPA
]
PROMPT="RAW photo of a person, full body shot, standing, CCTV surveillance footage style, top-down angle, highly detailed, photorealistic, correct anatomy"
NEG="blurry, low quality, deformed, multiple people, extra limbs, missing limbs, bad anatomy, bad proportions, disfigured, mutated, face portrait, cropped"

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]

print("1. 데이터 + 5명 선정...")
gby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{GAL_DIR}/*.jpg")):
    pid,cam=parse(f)
    if pid in ('-1','0000'): continue
    gby[pid][cam].append(f)
qby=defaultdict(lambda:defaultdict(list))
for f in sorted(glob.glob(f"{QRY_DIR}/*.jpg")):
    pid,cam=parse(f); qby[pid][cam].append(f)
ids=[p for p in sorted(qby) if "c6" in qby[p] and SRC_CAM in gby[p]][:N]
print(f"   {ids}")

print("2. OpenPose + C6 자세...")
op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
def to_img(o):
    if isinstance(o,tuple): o=o[0]
    if o is None: return None
    if not isinstance(o,Image.Image): o=Image.fromarray(o)
    return o.resize((512,768),Image.LANCZOS)
poses={pid:to_img(op(Image.open(qby[pid]["c6"][0]).convert("RGB").resize((512,768),Image.LANCZOS))) for pid in ids}

print("3. CLIP-ReID...")
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
base_q=np.mean([c1_real[p]@c6_query[p] for p in ids])

cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
results={}
for name,mpath,ipa in CONFIGS:
    print(f"\n4. [{name}] 로드+생성 (model={mpath.split('/')[-1]}, IPA={ipa})")
    try:
        pipe=StableDiffusionControlNetPipeline.from_pretrained(
            mpath,controlnet=cn,cache_dir=CACHE_DIR,torch_dtype=dtype,
            safety_checker=None,requires_safety_checker=False)
        pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
        pipe.set_ip_adapter_scale(ipa); pipe=pipe.to(device)
    except Exception as e:
        print(f"   [건너뜀] 로드 실패: {e}"); continue
    vals=[]
    for pid in ids:
        if poses[pid] is None: continue
        c1=Image.open(gby[pid][SRC_CAM][0]).convert("RGB").resize((512,1024),Image.LANCZOS)
        g=torch.Generator(device).manual_seed(42)
        gen=pipe(prompt=PROMPT,negative_prompt=NEG,image=poses[pid],ip_adapter_image=c1,
                 controlnet_conditioning_scale=1.0,num_inference_steps=30,guidance_scale=7.5,
                 width=512,height=768,generator=g).images[0]
        gen.save(f"{OUT}/{name}_{pid}.png")
        gf=feat_img(gen); vals.append((float(c1_real[pid]@gf),float(c6_query[pid]@gf)))
    results[name]=vals
    del pipe; torch.cuda.empty_cache()

print("\n"+"="*60)
print("생성 방식별 cosine (5명 평균)")
print("="*60)
print(f"{'방식':<20}{'생성↔정답':<12}{'생성↔쿼리':<12}")
print("-"*60)
print(f"{'[기준] 정답↔쿼리':<20}{'-':<12}{base_q:<12.3f}")
print("-"*60)
for name,vals in results.items():
    if not vals: continue
    cr=np.mean([v[0] for v in vals]); cq=np.mean([v[1] for v in vals])
    print(f"{name:<20}{cr:<12.3f}{cq:<12.3f}")
print("-"*60)
print(f"목표선={base_q:.3f}. 생성↔쿼리가 높을수록 좋음")
print(f"★ 이미지를 꼭 눈으로 비교: {OUT}/  (실사모델이 '진짜 사람'같은지)")