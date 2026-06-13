"""
44c_c6_base_gen_sdxl.py  ―  SDXL + ControlNet 버전 (강한 백본)

흐름:
- base = c6 이미지 (외형 보존)
- ControlNet OpenPose 가 c1~c5 자세를 condition
- 백본만 SD1.5 → SDXL 로 강화. 어댑터는 없음 → "백본 효과만" 분리

* A4000 16GB 빡빡하므로 메모리 최적화 옵션 활성:
  - fp16
  - attention slicing
  - vae slicing
  - cpu offload (필요 시 자동)
* 출력: outputs/c6base_gen_sdxl/{cam}/{pid}_gen_{cam}.png
"""

import os, sys, glob, torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
)
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/c6base_gen_sdxl"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
dtype = torch.float16
# SDXL 는 더 큰 native 해상도 필요. 768x1536 은 너무 무거우니 512x1024 (2:1 비율 유지)
SIZE = (512, 1024)
SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
STRENGTH = 0.4
NUM_IDS = None
POSE_POOL_SIZE = 30

os.makedirs(GEN_DIR, exist_ok=True)

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# ===== 데이터 =====
print(f"데이터 로드... (base={SOURCE_CAM}, targets={TARGET_CAMS})")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1', '0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

valid_ids = [pid for pid in sorted(gallery_by_id.keys())
             if SOURCE_CAM in gallery_by_id[pid]
             and any(tc in query_by_id[pid] for tc in TARGET_CAMS)]
if NUM_IDS:
    valid_ids = valid_ids[:NUM_IDS]
print(f"생성 대상 ID: {len(valid_ids)}명\n")

print("Pose reference pool 수집...")
valid_set = set(valid_ids)
pose_pools = {}
for tc in TARGET_CAMS:
    pool = []
    for pid in sorted(gallery_by_id.keys()):
        if pid in valid_set: continue
        if tc in gallery_by_id[pid]:
            pool.append(gallery_by_id[pid][tc][0])
        if len(pool) >= POSE_POOL_SIZE: break
    pose_pools[tc] = pool
    print(f"  {tc}: {len(pool)}장")
print()


# ===== SDXL + ControlNet =====
print("생성 모델 로드... (SDXL + ControlNet OpenPose, 메모리 최적화)")

# SDXL용 OpenPose ControlNet
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",       # SDXL 호환 OpenPose
    cache_dir=CACHE_DIR, torch_dtype=dtype,
)
# fp16 VAE (A4000 메모리 절약)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    cache_dir=CACHE_DIR, torch_dtype=dtype,
)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, vae=vae,
    cache_dir=CACHE_DIR, torch_dtype=dtype,
    use_safetensors=True,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# ===== 메모리 최적화 (A4000 16GB) =====
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
# GPU OOM 나면 아래 줄 주석 해제 → 느려지지만 동작:
# pipe.enable_model_cpu_offload()

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")


def generate(base_path, pose_ref_path, save_path):
    if os.path.exists(save_path):
        return
    base_img = Image.open(base_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=base_img,
        control_image=skel,
        strength=STRENGTH,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=gen,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)


print("=" * 70)
print(f"생성 시작 (SDXL): {SOURCE_CAM} → {TARGET_CAMS}, strength={STRENGTH}, size={SIZE}")
print(f"예상: {len(valid_ids)}명 × {len(TARGET_CAMS)}cam = {len(valid_ids)*len(TARGET_CAMS)}장")
print(f"⚠ SDXL 은 SD1.5 대비 ~2배 느림 (장당 ~5~7초)")
print("=" * 70)

for tc in TARGET_CAMS:
    out_dir = f"{GEN_DIR}/{tc}"
    os.makedirs(out_dir, exist_ok=True)
    pool = pose_pools[tc]
    if not pool:
        print(f"\n[{SOURCE_CAM}→{tc}] skip"); continue
    print(f"\n[{SOURCE_CAM}→{tc}] {out_dir}")
    for i, pid in enumerate(tqdm(valid_ids)):
        base_path = sorted(gallery_by_id[pid][SOURCE_CAM])[0]
        pose_ref = pool[i % len(pool)]
        save_path = f"{out_dir}/{pid}_gen_{tc}.png"
        try:
            generate(base_path, pose_ref, save_path)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n  [!] OOM pid={pid} tc={tc} → cpu_offload 켜고 재시도 권장")
        except Exception as e:
            print(f"\n  [!] pid={pid} tc={tc}: {e}")

print("\n" + "=" * 70)
print(f"✅ SDXL 생성 완료 → {GEN_DIR}/")
print("=" * 70)