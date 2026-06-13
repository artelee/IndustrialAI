"""
CLIP feature 기반 갤러리 확장 효과 측정
OSNet 대신 CLIP feature extractor 사용
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/clip_gen_c6"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ("-1","0000"): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# 50명 선택 (모든 카메라 있는 ID)
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    cams = set(gallery_by_id[pid].keys())
    if cams >= {"c1","c2","c6","c4","c5","c6"} and "c6" in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= 50:
        break

print(f"선택된 ID: {len(valid_ids)}명")

# CLIP 로드
print("CLIP 로드...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

@torch.no_grad()
def feat(path_or_pil):
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    inputs = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inputs)
    f = torch.nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

# 생성 모델 로드
print("생성 모델 로드...")
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
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.7)
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")

def generate(src_path, pose_ref_path, save_path):
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    src = Image.open(src_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose).resize(SIZE, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, CCTV camera view",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=skel, ip_adapter_image=src,
        num_inference_steps=25, guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=gen, width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    return result

# ===== 비교 평가 =====
print("="*60)
print("Condition A (Baseline) vs Condition B (생성 확장)")
print("CLIP feature 기반 매칭")
print("="*60)

# Pose reference pool (선택된 ID 제외)
pose_pool = defaultdict(list)
for pid, cams in gallery_by_id.items():
    if pid in set(valid_ids): continue
    for cam, imgs in cams.items():
        pose_pool[cam].extend(imgs[:1])

results_A, results_B = [], []

for pid in valid_ids:
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c6_query = sorted(query_by_id[pid]["c6"])[0]

    f_query = feat(c6_query)
    f_c1 = feat(c1_path)

    # Condition A: c1만 갤러리
    sim_A = float(f_query @ f_c1)
    results_A.append({"pid": pid, "sim": sim_A, "correct": sim_A > 0})

    # Condition B: c1 + 생성 c6
    save_path = f"{GEN_DIR}/{pid}_gen_c6.png"
    pose_ref = pose_pool["c6"][0] if pose_pool["c6"] else c1_path
    gen_img = generate(c1_path, pose_ref, save_path)
    f_gen = feat(gen_img)

    # 갤러리: c1 + 생성c6 중 더 가까운 것
    sim_B = max(float(f_query @ f_c1), float(f_query @ f_gen))
    results_B.append({"pid": pid, "sim": sim_B,
                      "sim_c1": float(f_query @ f_c1),
                      "sim_gen": float(f_query @ f_gen)})

avg_A = np.mean([r["sim"] for r in results_A])
avg_B = np.mean([r["sim"] for r in results_B])
gen_helped = sum(1 for r in results_B if r["sim_gen"] > r["sim_c1"])

print(f"\n조건 A (c1만):          평균 sim = {avg_A:.4f}")
print(f"조건 B (c1 + 생성c6):  평균 sim = {avg_B:.4f}")
print(f"변화: {avg_B - avg_A:+.4f}")
print(f"\n생성 이미지가 c1보다 가까운 케이스: {gen_helped}/{len(valid_ids)} ({100*gen_helped/len(valid_ids):.1f}%)")

if avg_B > avg_A:
    print("\n✅ CLIP feature 공간에서 갤러리 확장 효과 있음")
else:
    print("\n❌ CLIP feature 공간에서도 효과 없음")
