"""
44_c6_base_gen.py  ―  c6를 base로 c1~c5 자세 생성 (43번의 c6 버전)

흐름:
- base = c6 이미지 (외형 보존)
- target = c1, c2, c3, c4, c5 각각 (자세, OpenPose skeleton)
- 출력: outputs/c6base_gen_all/{cam}/{pid}_gen_{cam}.png
        ← 97_c6sparse_gen_eval.py 가 그대로 읽는 형식

* 평가는 분리(97번이 함). 이 스크립트는 순수 생성만.
* 중간에 끊겨도 이어서 가능 (기존 png 있으면 skip).
"""

import os, sys, glob, torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/c6base_gen_all"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
STRENGTH = 0.4
NUM_IDS = None          # ← 먼저 100명으로 sanity check. 잘 되면 None(전체)로 늘려서 재실행
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
    if pid in ('-1', '0000'):
        continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# valid: c6 갤러리 존재 + c1~c5 중 하나라도 query 존재
valid_ids = [
    pid for pid in sorted(gallery_by_id.keys())
    if SOURCE_CAM in gallery_by_id[pid]
    and any(tc in query_by_id[pid] for tc in TARGET_CAMS)
]
if NUM_IDS:
    valid_ids = valid_ids[:NUM_IDS]
print(f"생성 대상 ID: {len(valid_ids)}명\n")


# 각 target cam 별 pose reference pool (평가 ID와 겹치지 않게)
print("Pose reference pool 수집...")
valid_set = set(valid_ids)
pose_pools = {}
for tc in TARGET_CAMS:
    pool = []
    for pid in sorted(gallery_by_id.keys()):
        if pid in valid_set:
            continue
        if tc in gallery_by_id[pid]:
            pool.append(gallery_by_id[pid][tc][0])
        if len(pool) >= POSE_POOL_SIZE:
            break
    pose_pools[tc] = pool
    print(f"  {tc}: {len(pool)}장")
print()


# ===== 생성 파이프라인 (43번과 동일) =====
print("생성 모델 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    cache_dir=CACHE_DIR, torch_dtype=dtype,
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")


def generate(base_path, pose_ref_path, save_path):
    """base 외형 + pose_ref 자세로 생성. 이미 있으면 skip."""
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


# ===== 생성 루프: target cam × ID =====
print("=" * 70)
print(f"생성 시작: {SOURCE_CAM} → {TARGET_CAMS}, strength={STRENGTH}")
print(f"예상 생성량: {len(valid_ids)}명 × {len(TARGET_CAMS)}cam "
      f"= {len(valid_ids) * len(TARGET_CAMS)}장")
print("=" * 70)

for tc in TARGET_CAMS:
    out_dir = f"{GEN_DIR}/{tc}"
    os.makedirs(out_dir, exist_ok=True)
    pool = pose_pools[tc]
    if not pool:
        print(f"\n[{SOURCE_CAM}→{tc}] skip: pose pool 비어있음")
        continue
    print(f"\n[{SOURCE_CAM}→{tc}] {out_dir}")
    for i, pid in enumerate(tqdm(valid_ids)):
        base_path = sorted(gallery_by_id[pid][SOURCE_CAM])[0]
        pose_ref = pool[i % len(pool)]
        save_path = f"{out_dir}/{pid}_gen_{tc}.png"
        try:
            generate(base_path, pose_ref, save_path)
        except Exception as e:
            print(f"\n  [!] pid={pid} tc={tc} 실패: {e}")

print("\n" + "=" * 70)
print(f"✅ 생성 완료 → {GEN_DIR}/")
print("다음 단계: python scripts/97_c6sparse_gen_eval.py")
print("=" * 70)