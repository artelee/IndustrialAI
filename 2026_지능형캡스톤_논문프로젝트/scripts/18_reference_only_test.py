
"""
ControlNet Reference-Only 테스트
IP-Adapter 없이 원본 이미지를 attention에 직접 주입
→ 더 강한 외형 보존 가능성
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector
import torchreid

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/refonly_gen"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

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

valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and "c3" in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= 5:
        break

# OSNet
osnet = torchreid.models.build_model(name='osnet_x1_0', num_classes=751, pretrained=False)
torchreid.utils.load_pretrained_weights(osnet, OSNET_WEIGHTS)
osnet = osnet.eval().to(device)
transform = T.Compose([
    T.Resize((256,128)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@torch.no_grad()
def feat(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil
    t = transform(img).unsqueeze(0).to(device)
    f = osnet(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# ControlNet Reference 모드
print("모델 로드...")
from diffusers import UniPCMultistepScheduler

# ControlNet OpenPose (pose 제어)
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
)

# ControlNet Reference (외형 제어)
# Reference-only: 별도 체크포인트 없이 UNet attention에 ref 이미지 주입
controlnet_ref = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    cache_dir=CACHE_DIR, torch_dtype=dtype,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet_pose,
    cache_dir=CACHE_DIR, torch_dtype=dtype,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# IP-Adapter Plus (비교용 — 기존 방법)
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CACHE_DIR,
)

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")

# ===== scale 조합 실험 =====
# IP-Adapter scale을 0.7 → 1.0 → 1.2 로 올려서 외형 보존 강화
print("="*70)
print(f"{'ID':<8} {'scale=0.7':>12} {'scale=0.9':>12} {'scale=1.1':>12} {'원본c1':>10}")
print("-"*70)

all_results = []

for pid in valid_ids:
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c3_query = sorted(query_by_id[pid]["c3"])[0]
    c3_ref = sorted(gallery_by_id[pid].get("c3", gallery_by_id[pid]["c1"]))[0]

    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(c3_ref).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)

    f_real_c3 = feat(c3_query)
    f_c1 = feat(c1_path)
    sim_c1 = float(f_real_c3 @ f_c1)

    sims = []
    for scale in [0.7, 0.9, 1.1]:
        pipe.set_ip_adapter_scale(scale)
        gen = torch.Generator(device=device).manual_seed(42)
        result = pipe(
            prompt="a photo of a person, full body, standing, photorealistic",
            negative_prompt="blurry, low quality, deformed, multiple people, different person",
            image=pose_skel,
            ip_adapter_image=c1_img,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.7,
            generator=gen,
            width=SIZE[0], height=SIZE[1],
        ).images[0]

        save_path = f"{GEN_DIR}/{pid}_scale{int(scale*10)}.png"
        result.save(save_path)
        sim = float(f_real_c3 @ feat(result.resize((64,128), Image.LANCZOS)))
        sims.append(sim)

    print(f"{pid:<8} {sims[0]:>12.4f} {sims[1]:>12.4f} {sims[2]:>12.4f} {sim_c1:>10.4f}")
    all_results.append({"pid": pid, "sims": sims, "sim_c1": sim_c1})

print("="*70)
for i, scale in enumerate([0.7, 0.9, 1.1]):
    avg = np.mean([r["sims"][i] for r in all_results])
    print(f"scale={scale} 평균 sim: {avg:.4f}")
avg_c1 = np.mean([r["sim_c1"] for r in all_results])
print(f"원본 c1 평균 sim:   {avg_c1:.4f}")
print(f"이전 결과:          0.5714")
