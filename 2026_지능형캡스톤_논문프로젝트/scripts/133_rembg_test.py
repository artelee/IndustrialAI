#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
133_rembg_test.py ─ 배경 누끼(rembg) + 바닥 프롬프트 효과 테스트 (5명)

가설: IP-Adapter가 C1 배경(나무/차/노이즈)까지 외형으로 착각 → 배경 오염 + 외형 흐려짐.
     rembg로 사람만 남기고 회색 배경 처리 후 IPA에 넣으면 외형에 집중 → cosine ↑.

비교 (RealVis + IPA0.8, 132 최고설정 고정):
  A) 원본 그대로 (배경 포함)
  B) rembg 누끼 + 회색배경 + 바닥 프롬프트
→ B가 A보다 cosine 높으면 배경 오염이 원인이었음.
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
OUT=f"{PROJECT_DIR}/outputs/rembg_test"; os.makedirs(OUT,exist_ok=True)
MARKET=f"/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GAL_DIR=f"{MARKET}/bounding_box_test"; QRY_DIR=f"{MARKET}/query"
W=f"{CACHE_DIR}/clipreid_duke_nosie.pth"
device,dtype="cuda",torch.float16
SRC_CAM="c1"; N=5
BASE_MODEL="SG161222/Realistic_Vision_V5.1_noVAE"  # 132 최고
IPA_SCALE=0.8

# 바닥 공사 프롬프트
PROMPT_BASE="RAW photo of a person, full body shot, standing, CCTV surveillance footage style, top-down angle, photorealistic, correct anatomy"
PROMPT_FLOOR=", plain concrete floor, simple gray background, clean street"
NEG_BASE="blurry, low quality, deformed, multiple people, extra limbs, missing limbs, bad anatomy, bad proportions, disfigured, mutated, face portrait, cropped"
NEG_FLOOR=", trees, cars, buildings, complex background, messy background, floating objects"

def parse(f): p=os.path.basename(f).split("_"); return p[0],p[1][:2]

print("1. 데이터 + 5명...")
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

# rembg 준비
print("2. rembg 로드...")
try:
    from rembg import remove, new_session
    sess=new_session("u2net")
    def cutout(pil):
        rgba=remove(pil, session=sess)
        bg=Image.new("RGB",rgba.size,(128,128,128))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    REMBG_OK=True
except Exception as e:
    print(f"   rembg 실패({e}) → 누끼 없이 진행"); REMBG_OK=False

print("3. OpenPose + C6 자세...")
op=OpenposeDetector.from_pretrained("lllyasviel/Annotators",cache_dir=CACHE_DIR)
def to_img(o):
    if isinstance(o,tuple): o=o[0]
    if o is None: return None
    if not isinstance(o,Image.Image): o=Image.fromarray(o)
    return o.resize((512,768),Image.LANCZOS)
poses={pid:to_img(op(Image.open(qby[pid]["c6"][0]).convert("RGB").resize((512,768),Image.LANCZOS))) for pid in ids}

print("4. CLIP-ReID...")
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

print("5. 파이프라인 로드...")
cn=ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",cache_dir=CACHE_DIR,torch_dtype=dtype)
pipe=StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,controlnet=cn,cache_dir=CACHE_DIR,torch_dtype=dtype,
    safety_checker=None,requires_safety_checker=False)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter",subfolder="models",weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IPA_SCALE); pipe=pipe.to(device)

def gen_one(c1_img, pid, prompt, neg, tag):
    g=torch.Generator(device).manual_seed(42)
    img=pipe(prompt=prompt,negative_prompt=neg,image=poses[pid],ip_adapter_image=c1_img,
             controlnet_conditioning_scale=1.0,num_inference_steps=30,guidance_scale=7.5,
             width=512,height=768,generator=g).images[0]
    img.save(f"{OUT}/{pid}_{tag}.png")
    return feat_img(img)

resA=[]; resB=[]
print("6. 생성 (A: 원본배경 / B: 누끼+바닥프롬프트)...")
for pid in ids:
    if poses[pid] is None: continue
    raw=Image.open(gby[pid][SRC_CAM][0]).convert("RGB")
    # A) 원본 배경
    cA=raw.resize((512,1024),Image.LANCZOS)
    fA=gen_one(cA,pid,PROMPT_BASE,NEG_BASE,"A_orig")
    resA.append((float(c1_real[pid]@fA),float(c6_query[pid]@fA)))
    # B) 누끼 + 바닥 프롬프트
    if REMBG_OK:
        cleaned=cutout(raw)
        cleaned.resize((256,512)).save(f"{OUT}/{pid}_cutout.png")  # 누끼 확인용
        cB=cleaned.resize((512,1024),Image.LANCZOS)
    else:
        cB=cA
    fB=gen_one(cB,pid,PROMPT_BASE+PROMPT_FLOOR,NEG_BASE+NEG_FLOOR,"B_rembg")
    resB.append((float(c1_real[pid]@fB),float(c6_query[pid]@fB)))

print("\n"+"="*56)
print("배경 누끼 효과 (5명 평균, RealVis IPA0.8)")
print("="*56)
print(f"{'방식':<28}{'생성↔정답':<12}{'생성↔쿼리':<12}")
print("-"*56)
print(f"{'[기준] 정답↔쿼리':<28}{'-':<12}{base_q:<12.3f}")
print("-"*56)
print(f"{'A) 원본 배경':<28}{np.mean([v[0] for v in resA]):<12.3f}{np.mean([v[1] for v in resA]):<12.3f}")
print(f"{'B) 누끼+바닥프롬프트':<28}{np.mean([v[0] for v in resB]):<12.3f}{np.mean([v[1] for v in resB]):<12.3f}")
print("-"*56)
print(f"B > A 이면 배경 오염이 외형 흐림의 원인이었음. 목표선={base_q:.3f}")
print(f"★ 이미지: {OUT}/  (A=배경포함 / B=누끼 / *_cutout=누끼확인)")