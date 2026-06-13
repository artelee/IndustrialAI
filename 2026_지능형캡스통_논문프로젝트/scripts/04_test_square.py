"""
종횡비 가설 검증: 512x512 정사각형으로 생성
- SD 1.5의 학습 분포에 맞춤
- 사람이 가로로 늘어나 보이지만 줄무늬 진단이 우선
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

print("\n[1/5] ControlNet 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)

print("[2/5] SD 파이프라인 로드...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

print("[3/5] IP-Adapter 로드...")
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter_sd15.safetensors", cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.7)

print("[4/5] OpenPose 로드...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

print("\n[5/5] 두 가지 해상도로 생성")
content_img = Image.open(content_img_path).convert("RGB")
pose_img = Image.open(pose_img_path).convert("RGB")

# ===== 실험 A: 512x512 정사각형 (SD가 좋아하는 비율) =====
print("\n--- 실험 A: 512x512 정사각형 ---")
SIZE_A = (512, 512)
content_a = content_img.resize(SIZE_A, Image.LANCZOS)
pose_a = pose_img.resize(SIZE_A, Image.LANCZOS)
pose_skel_a = openpose(pose_a).resize(SIZE_A, Image.LANCZOS)

pose_skel_a.save(f"{DEBUG_DIR}/A_pose_skeleton_512x512.png")

generator = torch.Generator(device=device).manual_seed(42)
result_a = pipe(
    prompt="a photo of a person, full body, standing",
    negative_prompt="blurry, low quality, deformed, multiple people, stripes, lines, pattern",
    image=pose_skel_a,
    ip_adapter_image=content_a,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
    width=SIZE_A[0], height=SIZE_A[1],
).images[0]
result_a.save(f"{DEBUG_DIR}/A_generated_512x512.png")
print(f"  저장: A_generated_512x512.png")

# ===== 실험 B: 384x768 (사람 비율, 8의 배수) =====
print("\n--- 실험 B: 384x768 (사람 비율) ---")
SIZE_B = (384, 768)
content_b = content_img.resize(SIZE_B, Image.LANCZOS)
pose_b = pose_img.resize(SIZE_B, Image.LANCZOS)
pose_skel_b = openpose(pose_b).resize(SIZE_B, Image.LANCZOS)

pose_skel_b.save(f"{DEBUG_DIR}/B_pose_skeleton_384x768.png")

generator = torch.Generator(device=device).manual_seed(42)
result_b = pipe(
    prompt="a photo of a person, full body, standing",
    negative_prompt="blurry, low quality, deformed, multiple people, stripes, lines, pattern",
    image=pose_skel_b,
    ip_adapter_image=content_b,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
    width=SIZE_B[0], height=SIZE_B[1],
).images[0]
result_b.save(f"{DEBUG_DIR}/B_generated_384x768.png")
print(f"  저장: B_generated_384x768.png")

print("\n✅ 두 실험 완료")
print(f"\n비교:")
print(f"  A_generated_512x512.png  - 정사각형 (SD가 가장 좋아함)")
print(f"  B_generated_384x768.png  - 사람 비율 1:2 (SD 학습 분포 안)")
print(f"\n둘 다 줄무늬 → ControlNet/IP-Adapter 자체 문제")
print(f"A만 깨끗 → 종횡비 문제 (256x512는 너무 극단적이었음)")