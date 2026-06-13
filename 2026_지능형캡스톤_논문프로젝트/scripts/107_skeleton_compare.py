#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
107_skeleton_compare.py ─ skeleton 포맷 비교
  같은 자세 참조 → (A) body-only  vs  (B) hand+face 포함
  각각으로 txt2img+IPA 생성, 자세 제어 차이 확인
"""
import os, glob, torch
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler,
)
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OUT_DIR = f"{PROJECT_DIR}/outputs/skeleton_compare"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device, dtype = "cuda", torch.float16
SOURCE_CAM, TARGET_CAM = "c6", "c1"
SEED, SIZE = 42, (384, 768)
CN_SCALE, IP_SCALE = 1.0, 0.8   # 105와 동일 조건 유지
os.makedirs(OUT_DIR, exist_ok=True)


def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]


# 1. 데이터 (105와 동일한 선정 로직)
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

pose_src = next(gallery_by_id[p][TARGET_CAM][0]
                for p in sorted(gallery_by_id)
                if p != pid and TARGET_CAM in gallery_by_id[p])
pose_img = Image.open(pose_src).convert("RGB").resize(SIZE, Image.LANCZOS)
print(f"PID={pid}, 원본={os.path.basename(c6_path)}, 자세참조={os.path.basename(pose_src)}")


# 2. skeleton 2종 추출
print("OpenPose 로드...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)


def extract(img, hand_and_face):
    """버전별 인자 차이 대응 (hand_and_face / include_hand+include_face)"""
    try:
        out = openpose(img, hand_and_face=hand_and_face)
    except TypeError:
        out = openpose(img, include_hand=hand_and_face, include_face=hand_and_face)
    if isinstance(out, tuple): out = out[0]
    if not isinstance(out, Image.Image): out = Image.fromarray(out)
    return out.resize(SIZE, Image.LANCZOS)


skel_body = extract(pose_img, hand_and_face=False)
skel_full = extract(pose_img, hand_and_face=True)
skel_body.save(f"{OUT_DIR}/_skel_body.png")
skel_full.save(f"{OUT_DIR}/_skel_full.png")
print("skeleton 2종 저장 완료")


# 3. 파이프라인 (txt2img + IPA)
print("파이프라인 로드...")
cn = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(IP_SCALE)
pipe = pipe.to(device)

PROMPT = "a photo of a person, full body, surveillance"
NEG = "blurry, low quality, deformed, multiple people, extra limbs"


def gen(skel):
    g = torch.Generator(device).manual_seed(SEED)
    return pipe(
        prompt=PROMPT, negative_prompt=NEG,
        image=skel, ip_adapter_image=base_img,
        controlnet_conditioning_scale=CN_SCALE,
        num_inference_steps=30, guidance_scale=7.5,
        width=SIZE[0], height=SIZE[1], generator=g,
    ).images[0]


print("생성 중...")
gen_body = gen(skel_body)
gen_full = gen(skel_full)


# 4. grid: [원본][자세참조][skel body][gen body][skel full][gen full]
cells = [
    ("c6 source", base_img),
    ("pose ref",  pose_img),
    ("skel BODY", skel_body),
    ("gen BODY",  gen_body),
    ("skel FULL", skel_full),
    ("gen FULL",  gen_full),
]
cw, ch, lh = 200, 400, 26
grid = Image.new("RGB", (cw * len(cells) + 10, ch + lh + 10), "white")
draw = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    font = ImageFont.load_default()
for i, (lbl, img) in enumerate(cells):
    x = i * cw + 5
    color = "navy" if "gen" in lbl else ("darkgreen" if "skel" in lbl else "black")
    draw.text((x, 4), lbl, fill=color, font=font)
    grid.paste(img.resize((cw - 5, ch), Image.LANCZOS), (x, lh))
grid.save(f"{OUT_DIR}/grid_pid{pid}.png")
print(f"\n✅ 저장: {OUT_DIR}/grid_pid{pid}.png")
print("   BODY(눈코입+몸통만) vs FULL(손가락+얼굴 포함) 자세 제어 비교")