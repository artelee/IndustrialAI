"""
외형 보존 강화 1차 시도
- IP-Adapter scale: 0.7 → 0.9
- Prompt: 외형 묘사 추가
- ControlNet scale: 1.0 → 0.7
- 해상도: 384x768 (안정 범위)
"""

import os
import glob
import torch
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
DEBUG_DIR = f"{PROJECT_DIR}/debug"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"

device = "cuda"
dtype = torch.float16

# ===== ID 선택 =====
print("[0/5] 테스트 ID 검색...")
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

# ===== 모델 로드 =====
print("\n[1/5] ControlNet 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)

print("[2/5] SD 파이프라인...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

print("[3/5] IP-Adapter...")
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter_sd15.safetensors", cache_dir=CACHE_DIR,
)

print("[4/5] OpenPose...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

# ===== 이미지 처리 =====
content_img = Image.open(content_img_path).convert("RGB")
pose_img = Image.open(pose_img_path).convert("RGB")

SIZE = (384, 768)  # 안정 범위
content_up = content_img.resize(SIZE, Image.LANCZOS)
pose_up = pose_img.resize(SIZE, Image.LANCZOS)
pose_skel = openpose(pose_up).resize(SIZE, Image.LANCZOS)

# ===== 3가지 IP-Adapter scale 비교 =====
print("\n[5/5] IP-Adapter scale 3종 비교 생성")

for scale in [0.7, 0.85, 1.0]:
    print(f"\n--- scale={scale} ---")
    pipe.set_ip_adapter_scale(scale)
    
    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        # 🔧 외형 묘사 추가
        prompt="a photo of an asian woman in white dress, long black hair, full body, standing",
        negative_prompt="blurry, low quality, deformed, multiple people, different person, man, child",
        image=pose_skel,
        ip_adapter_image=content_up,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,  # 🔧 1.0 → 0.7
        generator=generator,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    
    fname = f"id_scale{int(scale*100):03d}.png"
    result.save(f"{DEBUG_DIR}/{fname}")
    print(f"  저장: {fname}")

print("\n✅ 3개 결과 비교 가능")
print(f"  id_scale070.png  - IP-Adapter 0.70 (기본, 외형 약함)")
print(f"  id_scale085.png  - IP-Adapter 0.85 (균형)")
print(f"  id_scale100.png  - IP-Adapter 1.00 (외형 강함, pose 무시 위험)")