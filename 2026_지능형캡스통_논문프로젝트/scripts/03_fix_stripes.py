"""
가로 줄무늬 해결: fp16-fix VAE 적용
- 외형/pose는 건드리지 않고 VAE만 교체
- 다른 모든 파라미터는 02번과 동일
"""

import os
import glob
import torch
from collections import defaultdict
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    DDIMScheduler,
    AutoencoderKL,  # 🆕 VAE 별도 로드용
)
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
DEBUG_DIR = f"{PROJECT_DIR}/debug"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"

device = "cuda"
dtype = torch.float16

# ===== 같은 ID 자동 선택 =====
print("[0/6] 테스트 ID 검색...")
id_to_cams = defaultdict(set)
id_to_files = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{GALLERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    parts = fname.split("_")
    pid = parts[0]
    if pid in ("-1", "0000"):
        continue
    cam = parts[1][:2]
    id_to_cams[pid].add(cam)
    id_to_files[pid][cam].append(f)

candidates = sorted([pid for pid, cams in id_to_cams.items() 
                     if "c1" in cams and "c3" in cams])
test_pid = candidates[0]
content_img_path = sorted(id_to_files[test_pid]["c1"])[0]
pose_img_path = sorted(id_to_files[test_pid]["c3"])[0]
print(f"✅ ID: {test_pid}")

# ===== 1. 🆕 fp16-fix VAE 로드 =====
print("\n[1/6] fp16-fix VAE 로드...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",  # 안정화된 VAE
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
)
print("✅ VAE 로드 완료")

# ===== 2. ControlNet =====
print("\n[2/6] ControlNet 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)

# ===== 3. SD 파이프라인 (VAE 교체) =====
print("\n[3/6] SD 파이프라인 로드 (VAE 교체)...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    vae=vae,  # 🆕 fp16-fix VAE 주입
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
print("✅ SD 파이프라인 로드 완료")

# ===== 4. IP-Adapter =====
print("\n[4/6] IP-Adapter 로드...")
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter_sd15.safetensors", cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.7)  # 02와 동일

# ===== 5. OpenPose =====
print("\n[5/6] OpenPose 로드...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

# ===== 6. 생성 (02와 모든 파라미터 동일) =====
print("\n[6/6] 생성")
content_img = Image.open(content_img_path).convert("RGB")
pose_img = Image.open(pose_img_path).convert("RGB")

TARGET_SIZE = (256, 512)
content_up = content_img.resize(TARGET_SIZE, Image.LANCZOS)
pose_up = pose_img.resize(TARGET_SIZE, Image.LANCZOS)

pose_skeleton = openpose(pose_up)
pose_skeleton = pose_skeleton.resize(TARGET_SIZE, Image.LANCZOS)

# 02번과 모든 파라미터 동일 (VAE만 다름)
generator = torch.Generator(device=device).manual_seed(42)
result = pipe(
    prompt="a photo of a person, full body, standing, surveillance camera, photorealistic",
    negative_prompt="blurry, low quality, deformed, distorted, multiple people",
    image=pose_skeleton,
    ip_adapter_image=content_up,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
    width=TARGET_SIZE[0],
    height=TARGET_SIZE[1],
).images[0]

result.save(f"{DEBUG_DIR}/06_vae_fixed.png")
result.resize((64, 128), Image.LANCZOS).save(f"{DEBUG_DIR}/07_vae_fixed_reid.png")

print("\n✅ 완료")
print(f"  06_vae_fixed.png       - VAE 교체 결과 (256x512)")
print(f"  07_vae_fixed_reid.png  - Re-ID 크기 (64x128)")