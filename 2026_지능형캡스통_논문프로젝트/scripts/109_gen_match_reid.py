#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
109_gen_match_reid.py
  1) gen 이미지 1장 생성 (txt2img + IPA + OpenPose)
  2) c6 query(anchor) vs c1~c5 real + gen 각각 CLIP-ReID cosine
  3) gen: 열화전 / 단순열화 / c6특성매칭열화 3종 비교
"""
import os, glob, io, sys, torch, numpy as np
from collections import defaultdict
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch.nn as nn
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler)
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OUT_DIR = f"{PROJECT_DIR}/outputs/gen_match_reid"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
device, dtype = "cuda", torch.float16
SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
SEED, SIZE = 42, (384, 768)
os.makedirs(OUT_DIR, exist_ok=True)


def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]


# ── 1. 데이터: c6 + c1~c5 모두 query 있는 ID ──────────────
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
           if SOURCE_CAM in gallery_by_id[p]
           and all(tc in query_by_id[p] for tc in TARGET_CAMS))
c6_path = sorted(gallery_by_id[pid][SOURCE_CAM])[0]
c6_img = Image.open(c6_path).convert("RGB")
real_paths = {tc: sorted(query_by_id[pid][tc])[0] for tc in TARGET_CAMS}
print(f"PID={pid}, anchor(c6)={os.path.basename(c6_path)}")

# 자세 참조 (다른 사람 c1)
pose_src = next(gallery_by_id[p]["c1"][0] for p in sorted(gallery_by_id)
                if p != pid and "c1" in gallery_by_id[p])


# ── 2. gen 생성 ──────────────────────────────────────────
print("OpenPose + 파이프라인 로드...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
skel = openpose(Image.open(pose_src).convert("RGB").resize(SIZE, Image.LANCZOS))
if isinstance(skel, tuple): skel = skel[0]
if not isinstance(skel, Image.Image): skel = Image.fromarray(skel)
skel = skel.resize(SIZE, Image.LANCZOS)

cn = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter-plus_sd15.safetensors")
pipe.set_ip_adapter_scale(0.8)
pipe = pipe.to(device)

g = torch.Generator(device).manual_seed(SEED)
gen_raw = pipe(
    prompt="a photo of a person, full body, surveillance",
    negative_prompt="blurry, low quality, deformed, multiple people, extra limbs",
    image=skel, ip_adapter_image=c6_img,
    controlnet_conditioning_scale=1.0,
    num_inference_steps=30, guidance_scale=7.5,
    width=SIZE[0], height=SIZE[1], generator=g,
).images[0]
gen_raw.save(f"{OUT_DIR}/gen_raw.png")
del pipe; torch.cuda.empty_cache()


# ── 3. 열화 함수 2종 ─────────────────────────────────────
def degrade_simple(img, down=64, blur=0.6, jpeg_q=55):
    w, h = img.size
    img = img.resize((down, int(down*h/w)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=jpeg_q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def analyze_cctv(ref):
    """c6 원본의 화질 특성 측정 → 열화 파라미터 추정"""
    a = np.asarray(ref.convert("RGB")).astype(np.float32)
    gray = a.mean(2)
    # 선명도: Laplacian 분산 (낮을수록 흐림)
    lap = np.abs(np.gradient(np.gradient(gray, axis=0)[0], axis=0)).var()
    sharp = float(lap)
    # 노이즈: 고주파 잔차 표준편차
    blurred = Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    noise = float((gray - np.asarray(blurred)).std())
    # 밝기/대비/채도
    bright = float(gray.mean())
    contrast = float(gray.std())
    hsv = np.asarray(ref.convert("HSV")).astype(np.float32)
    sat = float(hsv[:, :, 1].mean())
    # 유효 해상도: 원본이 작을수록 down 작게
    eff = min(ref.size)
    return dict(sharp=sharp, noise=noise, bright=bright,
               contrast=contrast, sat=sat, eff_res=eff)


def degrade_matched(img, ref):
    """c6 특성에 맞춰 gen 열화"""
    s = analyze_cctv(ref)
    w, h = img.size
    # 1) 선명도 기반 다운스케일 강도: 흐릴수록 더 작게
    down = int(np.clip(48 + s["sharp"]*0.5, 40, 96))
    img = img.resize((down, int(down*h/w)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    # 2) 블러: 노이즈 적고 흐리면 강하게
    blur = float(np.clip(1.2 - s["sharp"]*0.01, 0.3, 1.2))
    img = img.filter(ImageFilter.GaussianBlur(blur))
    # 3) 밝기/대비/채도 매칭
    g = np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img = ImageEnhance.Brightness(img).enhance(np.clip(s["bright"]/(g.mean()+1e-6), 0.7, 1.3))
    g2 = np.asarray(img.convert("RGB")).astype(np.float32).mean(2)
    img = ImageEnhance.Contrast(img).enhance(np.clip(s["contrast"]/(g2.std()+1e-6), 0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(np.clip(s["sat"]/ \
          (np.asarray(img.convert("HSV")).astype(np.float32)[:,:,1].mean()+1e-6), 0.7, 1.3))
    # 4) 가우시안 노이즈 주입
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    arr += np.random.normal(0, s["noise"]*0.6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # 5) JPEG 압축
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=55); buf.seek(0)
    return Image.open(buf).convert("RGB"), s


# ── 4. CLIP-ReID ─────────────────────────────────────────
print("CLIP-ReID 로드...")
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg as reid_cfg
from model.make_model_clipreid import make_model
W = f"{CACHE_DIR}/clipreid_duke_nosie.pth"

reid_cfg.MODEL.NAME = "ViT-B-16"; reid_cfg.MODEL.STRIDE_SIZE = [16, 16]
reid_cfg.MODEL.SIE_CAMERA = False; reid_cfg.MODEL.SIE_COE = 0.0
reid_cfg.MODEL.ID_LOSS_TYPE = "softmax"
reid_cfg.INPUT.SIZE_TRAIN = [256, 128]      # ← 이 줄 누락이 원인
reid_cfg.INPUT.SIZE_TEST = [256, 128]
reid_cfg.INPUT.PIXEL_MEAN = [0.5]*3; reid_cfg.INPUT.PIXEL_STD = [0.5]*3
reid_cfg.DATASETS.NAMES = "market1501"; reid_cfg.TEST.NECK_FEAT = "before"

try:    
    _reid = make_model(reid_cfg, num_class=702, camera_num=0, view_num=1)
except Exception: 
    _reid = make_model(reid_cfg, num_class=702, camera_num=6, view_num=1)
    
_reid.load_param(W); _reid = _reid.to(device).eval()
reid_tf = T.Compose([T.Resize([256,128]), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])

@torch.no_grad()
def feat(img):
    t = reid_tf(img.convert("RGB")).unsqueeze(0).to(device)
    f = _reid(t, cam_label=None)
    if isinstance(f,(list,tuple)): f = f[0] if isinstance(f[0],torch.Tensor) else f[-1]
    if f.dim()>2: f = f.view(f.size(0),-1)
    return nn.functional.normalize(f.float(),dim=1).cpu().numpy().flatten()


# ── 5. 측정 ──────────────────────────────────────────────
anchor = feat(c6_img)
print(f"\n{'='*55}\nanchor = c6 ({pid})\n{'='*55}")
print(f"{'대상':<24}{'cosine':>10}")
print("-"*36)
for tc in TARGET_CAMS:
    cs = float(feat(Image.open(real_paths[tc]).convert("RGB")) @ anchor)
    print(f"{tc+' real':<24}{cs:>10.4f}")
print("-"*36)

gen_simple = degrade_simple(gen_raw)
gen_matched, stats = degrade_matched(gen_raw, c6_img)
gen_simple.save(f"{OUT_DIR}/gen_simple.png")
gen_matched.save(f"{OUT_DIR}/gen_matched.png")

for name, im in [("gen 열화전", gen_raw),
                 ("gen 단순열화", gen_simple),
                 ("gen c6특성매칭", gen_matched)]:
    print(f"{name:<24}{float(feat(im)@anchor):>10.4f}")
print("-"*36)
print(f"\n[c6 분석] sharp={stats['sharp']:.1f} noise={stats['noise']:.1f} "
      f"bright={stats['bright']:.0f} contrast={stats['contrast']:.0f} "
      f"sat={stats['sat']:.0f} eff_res={stats['eff_res']}")


# ── 6. grid ──────────────────────────────────────────────
cells = [("c6 anchor", c6_img)] + [(f"{tc} real", Image.open(real_paths[tc]).convert("RGB")) for tc in TARGET_CAMS]
cells += [("gen raw", gen_raw), ("gen simple", gen_simple), ("gen matched", gen_matched)]
from PIL import ImageDraw, ImageFont
cw, ch, lh = 160, 320, 24
grid = Image.new("RGB", (cw*len(cells)+10, ch+lh+10), "white")
d = ImageDraw.Draw(grid)
try: fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except Exception: fnt = ImageFont.load_default()
for i,(l,im) in enumerate(cells):
    x=i*cw+5
    d.text((x,4), l, fill="navy" if "gen" in l else "black", font=fnt)
    grid.paste(im.resize((cw-5,ch), Image.LANCZOS), (x,lh))
grid.save(f"{OUT_DIR}/grid_pid{pid}.png")
print(f"\n✅ grid: {OUT_DIR}/grid_pid{pid}.png")
print(f"✅ gen: gen_raw / gen_simple / gen_matched .png")