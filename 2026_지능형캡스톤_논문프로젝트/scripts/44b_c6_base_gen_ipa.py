"""
44b_c6_base_gen_ipa.py  ―  SD1.5 + IP-Adapter Plus 버전 (외형 보존 강화)

흐름:
- base = c6 이미지 (외형 보존)
- IP-Adapter Plus 가 c6 이미지를 image prompt 로 받아 외형 일관성 ↑
- ControlNet OpenPose 가 c1~c5 자세를 condition 으로 줌
- 출력: outputs/c6base_gen_ipa/{cam}/{pid}_gen_{cam}.png
        ← 98번 GEN_DIR 만 c6base_gen_ipa 로 바꾸면 그대로 비교 가능

* 44번(기본 SD1.5)과 동일한 ID/자세/시드 사용 → 어댑터 효과만 분리
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
GEN_DIR = f"{PROJECT_DIR}/outputs/c6base_gen_ipa"            # ← IPA 전용
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
SOURCE_CAM = "c6"
TARGET_CAMS = ["c1", "c2", "c3", "c4", "c5"]
STRENGTH = 0.4
IP_ADAPTER_SCALE = 0.8       # IPA 영향 강도 (0=어댑터 off, 1=최대). 0.8 = 외형 강하게 보존
NUM_IDS = None              # None = 전체. 빠른 sanity check 시 100.
POSE_POOL_SIZE = 30

os.makedirs(GEN_DIR, exist_ok=True)

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# ===== 데이터 (44번과 동일) =====
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


# ===== 생성 파이프라인 (SD1.5 + ControlNet + IP-Adapter Plus) =====
print("생성 모델 로드... (SD1.5 + ControlNet OpenPose + IP-Adapter Plus)")
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

# IP-Adapter Plus 로드 (외형 보존 강화)
# 네 메모리에 'ip-adapter-plus_sd15.safetensors' 가 더 좋다고 적혀 있음
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors",
)
pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")


def generate(base_path, pose_ref_path, save_path):
    if os.path.exists(save_path):
        return
    base_img = Image.open(base_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)

    # OpenPose 안전 호출 (tuple/None/ndarray 다 처리)
    skel_out = openpose(pose_img)
    if isinstance(skel_out, tuple):
        skel_out = skel_out[0]
    if skel_out is None:
        return
    if not isinstance(skel_out, Image.Image):
        try:
            skel_out = Image.fromarray(skel_out)
        except Exception:
            return
    skel = skel_out.resize(SIZE, Image.LANCZOS)

    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=base_img,
        control_image=skel,
        ip_adapter_image=base_img,
        strength=STRENGTH,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=gen,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)


print("=" * 70)
print(f"생성 시작 (SD1.5 + IPA): {SOURCE_CAM} → {TARGET_CAMS}, "
      f"strength={STRENGTH}, IPA_scale={IP_ADAPTER_SCALE}")
print(f"예상: {len(valid_ids)}명 × {len(TARGET_CAMS)}cam = {len(valid_ids)*len(TARGET_CAMS)}장")
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
        except Exception as e:
            import traceback
            print(f"\n  [!] pid={pid} tc={tc}: {e}")
            traceback.print_exc()
            sys.exit(1)        # 첫 에러 한 번만 보고 멈춤

print("\n" + "=" * 70)
print(f"✅ IPA 생성 완료 → {GEN_DIR}/")
print("=" * 70)