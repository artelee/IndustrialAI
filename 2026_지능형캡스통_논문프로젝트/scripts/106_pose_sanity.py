#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
106_pose_sanity.py ─ 한 ID 빠른 sanity check
  img2img 방식 (105와 동일) vs txt2img+IPA 방식 비교
  목적: "인물이 안 만들어지는" 원인이 strength(img2img) 때문인지 확인
"""
import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel, DDIMScheduler,
)
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OUT_DIR = f"{PROJECT_DIR}/outputs/pose_sanity"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device, dtype = "cuda", torch.float16
SOURCE_CAM = "c6"
TARGET_CAM = "c1"        # 자세 가져올 타겟 카메라 1개만
SEED = 42
SIZE = (384, 768)
os.makedirs(OUT_DIR, exist_ok=True)


def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]


# 1. 데이터 ─ c6 원본 있고 c1 query 있는 ID 1명
print("데이터 로드...")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1', '0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

pid = next(p for p in sorted(gallery_by_id)
           if SOURCE_CAM in gallery_by_id[p] and TARGET_CAM in query_by_id[p])
c6_path = sorted(gallery_by_id[pid][SOURCE_CAM])[0]
base_img = Image.open(c6_path).convert("RGB").resize(SIZE, Image.LANCZOS)
print(f"테스트 PID: {pid}, 원본: {os.path.basename(c6_path)}")

# 자세 참조: 평가 ID 아닌 다른 사람의 c1 이미지
pose_src = next(gallery_by_id[p][TARGET_CAM][0]
                for p in sorted(gallery_by_id)
                if p != pid and TARGET_CAM in gallery_by_id[p])
pose_img = Image.open(pose_src).convert("RGB").resize(SIZE, Image.LANCZOS)
print(f"자세 참조: {os.path.basename(pose_src)}")


# 2. OpenPose skeleton 추출
print("OpenPose 로드 + skeleton 추출...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
skel = openpose(pose_img)
if isinstance(skel, tuple): skel = skel[0]
if not isinstance(skel, Image.Image): skel = Image.fromarray(skel)
skel = skel.resize(SIZE, Image.LANCZOS)
skel.save(f"{OUT_DIR}/_skeleton.png")


# 3. 파이프라인 2종 로드
def load_cn():
    return ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype)

print("img2img 파이프라인 로드...")
pipe_i2i = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=load_cn(), cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe_i2i.scheduler = DDIMScheduler.from_config(pipe_i2i.scheduler.config)
pipe_i2i = pipe_i2i.to(device)

print("txt2img + IP-Adapter 파이프라인 로드...")
pipe_t2i = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=load_cn(), cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe_t2i.scheduler = DDIMScheduler.from_config(pipe_t2i.scheduler.config)
pipe_t2i.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors")
pipe_t2i = pipe_t2i.to(device)

PROMPT = "a photo of a person, full body, surveillance"
NEG = "blurry, low quality, deformed, multiple people, extra limbs"


def gen_i2i(strength, cn_scale):
    g = torch.Generator(device).manual_seed(SEED)
    return pipe_i2i(
        prompt=PROMPT, negative_prompt=NEG,
        image=base_img, control_image=skel,
        strength=strength, controlnet_conditioning_scale=cn_scale,
        num_inference_steps=30, guidance_scale=7.5,
        width=SIZE[0], height=SIZE[1], generator=g,
    ).images[0]


def gen_t2i(ip_scale, cn_scale):
    g = torch.Generator(device).manual_seed(SEED)
    pipe_t2i.set_ip_adapter_scale(ip_scale)
    return pipe_t2i(
        prompt=PROMPT, negative_prompt=NEG,
        image=skel, ip_adapter_image=base_img,
        controlnet_conditioning_scale=cn_scale,
        num_inference_steps=30, guidance_scale=7.5,
        width=SIZE[0], height=SIZE[1], generator=g,
    ).images[0]


# 4. 파라미터 grid
print("생성 중...")
cells = [("c6 source", base_img), ("skeleton", skel)]

# img2img: strength 스윕 (105의 0.4 망한 값 포함)
for s in [0.4, 0.65, 0.8]:
    cells.append((f"i2i s={s}", gen_i2i(s, 1.0)))

# txt2img+IPA: ip_scale 스윕
for ip in [0.6, 0.8, 1.0]:
    cells.append((f"t2i ip={ip}", gen_t2i(ip, 1.0)))


# 5. grid 저장
cw, ch, lh = 200, 400, 26
n = len(cells)
grid = Image.new("RGB", (cw * n + 10, ch + lh + 10), "white")
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    font = ImageFont.load_default()
for i, (lbl, img) in enumerate(cells):
    x = i * cw + 5
    draw.text((x, 4), lbl, fill="navy" if "=" in lbl else "black", font=font)
    grid.paste(img.resize((cw - 5, ch), Image.LANCZOS), (x, lh))
grid.save(f"{OUT_DIR}/grid_pid{pid}.png")
print(f"\n✅ 저장: {OUT_DIR}/grid_pid{pid}.png")
print("   왼쪽부터: 원본 / 스켈레톤 / img2img(s=0.4,0.65,0.8) / txt2img+IPA(ip=0.6,0.8,1.0)")