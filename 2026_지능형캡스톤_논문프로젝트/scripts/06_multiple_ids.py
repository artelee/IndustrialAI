"""
여러 ID로 변환 테스트
- 다양한 외형(남/여, 옷 색깔)으로 일반성 검증
- 384x768 안정 해상도 사용
- IP-Adapter scale 0.85 (균형)
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

# 다양성 확보: 여러 ID 선택
TEST_IDS = ["0001", "0007", "0023", "0050", "0100"]
SOURCE_CAM = "c1"
TARGET_CAM = "c3"

device = "cuda"
dtype = torch.float16

# ===== ID별 이미지 매핑 =====
print("[0/5] ID별 이미지 검색...")
id_to_files = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{GALLERY_DIR}/*.jpg"):
    fname = os.path.basename(f)
    parts = fname.split("_")
    pid = parts[0]
    if pid in ("-1", "0000"):
        continue
    cam = parts[1][:2]
    id_to_files[pid][cam].append(f)

# 유효한 ID만 (소스, 타겟 카메라 둘 다 있어야 함)
valid_tests = []
for pid in TEST_IDS:
    if SOURCE_CAM in id_to_files[pid] and TARGET_CAM in id_to_files[pid]:
        content = sorted(id_to_files[pid][SOURCE_CAM])[0]
        pose = sorted(id_to_files[pid][TARGET_CAM])[0]
        valid_tests.append((pid, content, pose))
    else:
        print(f"  ⚠️ ID {pid}: {SOURCE_CAM} 또는 {TARGET_CAM} 누락, 스킵")

# 유효 ID가 부족하면 자동으로 채우기
if len(valid_tests) < 3:
    print(f"  📌 유효 ID {len(valid_tests)}개, 자동으로 추가 검색...")
    for pid in sorted(id_to_files.keys()):
        if pid in TEST_IDS:
            continue
        if SOURCE_CAM in id_to_files[pid] and TARGET_CAM in id_to_files[pid]:
            content = sorted(id_to_files[pid][SOURCE_CAM])[0]
            pose = sorted(id_to_files[pid][TARGET_CAM])[0]
            valid_tests.append((pid, content, pose))
            if len(valid_tests) >= 5:
                break

print(f"\n✅ 총 {len(valid_tests)}개 ID 테스트 예정")
for pid, c, p in valid_tests:
    print(f"  ID {pid}: {os.path.basename(c)} → {os.path.basename(p)}")

# ===== 모델 로드 =====
print("\n[1/4] ControlNet 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)

print("[2/4] SD 파이프라인...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

print("[3/4] IP-Adapter...")
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter_sd15.safetensors", cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.85)  # 균형값

print("[4/4] OpenPose...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)

# ===== 각 ID 변환 =====
SIZE = (384, 768)
print(f"\n--- 변환 시작 (해상도 {SIZE}, IP-Adapter scale 0.85) ---")

for idx, (pid, content_path, pose_path) in enumerate(valid_tests, 1):
    print(f"\n[{idx}/{len(valid_tests)}] ID {pid} 변환 중...")
    
    content_img = Image.open(content_path).convert("RGB")
    pose_img = Image.open(pose_path).convert("RGB")
    
    content_up = content_img.resize(SIZE, Image.LANCZOS)
    pose_up = pose_img.resize(SIZE, Image.LANCZOS)
    pose_skel = openpose(pose_up).resize(SIZE, Image.LANCZOS)
    
    # 비교용으로 입력도 같이 저장
    content_img.resize(SIZE, Image.LANCZOS).save(f"{DEBUG_DIR}/multi_{pid}_1_content.png")
    pose_img.resize(SIZE, Image.LANCZOS).save(f"{DEBUG_DIR}/multi_{pid}_2_pose_target.png")
    pose_skel.save(f"{DEBUG_DIR}/multi_{pid}_3_pose_skel.png")
    
    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        # 일반적 prompt (특정 외형 명시 X) - 진짜 IP-Adapter 성능 보기 위함
        prompt="a photo of a person, full body, standing, photorealistic",
        negative_prompt="blurry, low quality, deformed, multiple people, different person",
        image=pose_skel,
        ip_adapter_image=content_up,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        generator=generator,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    
    result.save(f"{DEBUG_DIR}/multi_{pid}_4_generated.png")
    print(f"  ✅ multi_{pid}_4_generated.png 저장")

print("\n🎉 모든 변환 완료!")
print(f"\n각 ID마다 4장씩 생성됨 (input + skeleton + output):")
print(f"  multi_<ID>_1_content.png       원본 (소스 카메라)")
print(f"  multi_<ID>_2_pose_target.png   타겟 카메라 실제 이미지 (참고용)")
print(f"  multi_<ID>_3_pose_skel.png     추출된 pose")
print(f"  multi_<ID>_4_generated.png     생성 결과")