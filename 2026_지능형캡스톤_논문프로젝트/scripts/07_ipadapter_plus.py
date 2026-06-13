"""
IP-Adapter Plus로 외형 보존 강화
- 기본 vs Plus 비교
- 같은 ID 5명에 대해 Plus 버전 결과 생성
"""

import os
import glob
import torch
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector
from huggingface_hub import hf_hub_download

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
DEBUG_DIR = f"{PROJECT_DIR}/debug"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"

device = "cuda"
dtype = torch.float16

# ===== Plus 버전 다운로드 =====
print("[Plus 가중치 다운로드]")
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter-plus_sd15.safetensors",
    cache_dir=CACHE_DIR,
)
print("✅ Plus weights 다운로드 완료\n")

# ===== ID 선택 (이전과 동일하게) =====
print("[ID 검색]")
id_to_files = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{GALLERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    parts = fname.split("_")
    pid = parts[0]
    if pid in ("-1", "0000"):
        continue
    cam = parts[1][:2]
    id_to_files[pid][cam].append(f)

# 0001, 0003, 0004, 0005, 0006 (있는 것만)
TEST_IDS = ["0001", "0003", "0004", "0005", "0006"]
valid_tests = []
for pid in TEST_IDS:
    if "c1" in id_to_files[pid] and "c3" in id_to_files[pid]:
        c = sorted(id_to_files[pid]["c1"])[0]
        p = sorted(id_to_files[pid]["c3"])[0]
        valid_tests.append((pid, c, p))

print(f"✅ {len(valid_tests)}개 ID 테스트\n")

# ===== 모델 로드 =====
print("[모델 로드]")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# 🔧 Plus 가중치 로드
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors",  # ← 변경점
    cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.8)

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")

# ===== 변환 =====
SIZE = (384, 768)

for idx, (pid, content_path, pose_path) in enumerate(valid_tests, 1):
    print(f"[{idx}/{len(valid_tests)}] ID {pid}")
    
    content_img = Image.open(content_path).convert("RGB")
    pose_img = Image.open(pose_path).convert("RGB")
    content_up = content_img.resize(SIZE, Image.LANCZOS)
    pose_up = pose_img.resize(SIZE, Image.LANCZOS)
    pose_skel = openpose(pose_up).resize(SIZE, Image.LANCZOS)
    
    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        # 🔧 배경 명시 추가 - 줄무늬 방지
        prompt="a photo of a person, full body, standing, walking on a street, clear background, photorealistic",
        negative_prompt="blurry, low quality, deformed, multiple people, stripes, lines, horizontal lines, pattern, artifacts, noise",
        image=pose_skel,
        ip_adapter_image=content_up,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=generator,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    
    result.save(f"{DEBUG_DIR}/plus_{pid}_generated.png")
    print(f"  ✅ plus_{pid}_generated.png")

print("\n🎉 Plus 버전 변환 완료")
print(f"\n저장된 파일:")
for pid, _, _ in valid_tests:
    print(f"  plus_{pid}_generated.png")