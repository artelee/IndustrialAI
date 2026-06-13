"""
Step 1-2: 첫 시점 변환 테스트 (자동 ID 선택판)
- 여러 카메라에서 등장하는 ID를 자동으로 찾음
"""

import os
import glob
import torch
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

# ===== 경로 설정 =====
HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
DEBUG_DIR = f"{PROJECT_DIR}/debug"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"

os.makedirs(DEBUG_DIR, exist_ok=True)

device = "cuda"
dtype = torch.float16

# ===== 0. 적합한 테스트 ID 자동 찾기 =====
# 같은 ID가 여러 카메라에서 찍힌 경우를 찾자.
# 카메라 1과 카메라 3 모두에서 등장하는 ID를 우선 선택.
print("[0/5] 적합한 테스트 ID 검색 중...")

id_to_cams = defaultdict(set)
id_to_files = defaultdict(lambda: defaultdict(list))

for f in glob.glob(f"{GALLERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    parts = fname.split("_")
    pid = parts[0]
    if pid in ("-1", "0000"):  # distractor 제외
        continue
    cam = parts[1][:2]  # "c1s1" -> "c1"
    id_to_cams[pid].add(cam)
    id_to_files[pid][cam].append(f)

# c1, c3 둘 다 가진 ID 우선
candidates = [pid for pid, cams in id_to_cams.items() 
              if "c1" in cams and "c3" in cams]
candidates.sort()

if not candidates:
    raise RuntimeError("c1, c3 모두에 등장하는 ID가 없음")

test_pid = candidates[0]
content_img_path = sorted(id_to_files[test_pid]["c1"])[0]
pose_img_path = sorted(id_to_files[test_pid]["c3"])[0]

print(f"✅ 선택된 ID: {test_pid}")
print(f"   카메라 등장: {sorted(id_to_cams[test_pid])}")
print(f"   Content (c1): {os.path.basename(content_img_path)}")
print(f"   Pose    (c3): {os.path.basename(pose_img_path)}")

# ===== 1. ControlNet OpenPose 로드 =====
print("\n[1/5] ControlNet 로드 중...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
)
print("✅ ControlNet 로드 완료")

# ===== 2. SD v1.5 + ControlNet 파이프라인 =====
print("\n[2/5] SD 파이프라인 로드 중...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
print("✅ SD 파이프라인 로드 완료")

# ===== 3. IP-Adapter 붙이기 =====
print("\n[3/5] IP-Adapter 로드 중...")
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.safetensors",
    cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.7)

# IP-Adapter 로드 후에 attention slicing
# pipe.enable_attention_slicing()
print("✅ IP-Adapter 로드 완료")

# ===== 4. OpenPose detector =====
print("\n[4/5] OpenPose detector 로드 중...")
openpose = OpenposeDetector.from_pretrained(
    "lllyasviel/Annotators",
    cache_dir=CACHE_DIR,
)
print("✅ OpenPose 로드 완료")

# ===== 5. 이미지 처리 + 생성 =====
print("\n[5/5] 시점 변환 생성")

# 이미지 로드
content_img = Image.open(content_img_path).convert("RGB")
pose_img = Image.open(pose_img_path).convert("RGB")
print(f"원본 크기 - Content: {content_img.size}, Pose: {pose_img.size}")

# 업스케일
TARGET_SIZE = (256, 512)
content_up = content_img.resize(TARGET_SIZE, Image.LANCZOS)
pose_up = pose_img.resize(TARGET_SIZE, Image.LANCZOS)

content_up.save(f"{DEBUG_DIR}/01_content_upscaled.png")
pose_up.save(f"{DEBUG_DIR}/02_pose_upscaled.png")

# Pose 추출
pose_skeleton = openpose(pose_up)
pose_skeleton = pose_skeleton.resize(TARGET_SIZE, Image.LANCZOS)
pose_skeleton.save(f"{DEBUG_DIR}/03_pose_skeleton.png")
print("✅ Pose 추출 완료")

# 생성
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

result.save(f"{DEBUG_DIR}/04_generated.png")
result_reid = result.resize((64, 128), Image.LANCZOS)
result_reid.save(f"{DEBUG_DIR}/05_generated_reid_size.png")

print("\n✅ 생성 완료!")
print(f"\n결과물 위치: {DEBUG_DIR}/")
print("  01_content_upscaled.png    - 외형 참조 (c1)")
print("  02_pose_upscaled.png       - 시점 참조 (c3)")
print("  03_pose_skeleton.png       - 추출된 pose")
print("  04_generated.png           - 생성 결과")
print("  05_generated_reid_size.png - Re-ID 입력 크기")
print(f"\n테스트 ID: {test_pid}")