
"""
ControlNet Reference-Only 테스트
IP-Adapter 완전 제거
원본 이미지를 UNet self-attention에 직접 주입
→ CLIP 안 거침 → 외형 디테일 훨씬 강하게 보존
"""

import os, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionControlNetReferencePipeline, ControlNetModel, DDIMScheduler
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
SIZE = (512, 512)  # Reference-Only은 정사각형이 안정적

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
    img = Image.open(path_or_pil).convert("RGB") if isinstance(path_or_pil, str) else path_or_pil
    t = transform(img).unsqueeze(0).to(device)
    f = osnet(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# ===== Reference-Only Pipeline =====
print("모델 로드...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    cache_dir=CACHE_DIR, torch_dtype=dtype
)
pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")

# ===== 실험 =====
print("="*65)
print(f"{'ID':<8} {'sim(진짜c3, Reference생성)':>28} {'sim(진짜c3, 원본c1)':>18}")
print("-"*65)

results = []
for pid in valid_ids:
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c3_query = sorted(query_by_id[pid]["c3"])[0]
    c3_gallery = sorted(gallery_by_id[pid].get("c3", gallery_by_id[pid]["c1"]))[0]

    # 원본 이미지 (reference)
    ref_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)

    # 타겟 포즈
    pose_img = Image.open(c3_gallery).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)

    # Reference-Only 생성
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, CCTV surveillance camera",
        negative_prompt="blurry, low quality, deformed, multiple people, different clothes",
        ref_image=ref_img,           # 핵심: ref_image로 원본 주입
        image=pose_skel,             # pose control
        num_inference_steps=30,
        guidance_scale=7.5,
        reference_attn=True,         # self-attention에 ref 주입
        reference_adain=True,        # AdaIN으로 스타일 주입
        generator=gen,
        width=SIZE[0], height=SIZE[1],
    ).images[0]

    save_path = f"{GEN_DIR}/{pid}_refonly.png"
    result.save(save_path)

    f_real_c3 = feat(c3_query)
    f_gen = feat(result)
    f_c1 = feat(c1_path)

    sim_gen = float(f_real_c3 @ f_gen)
    sim_c1 = float(f_real_c3 @ f_c1)
    better = "✅" if sim_gen > sim_c1 else "❌"

    results.append({"pid": pid, "sim_gen": sim_gen, "sim_c1": sim_c1})
    print(f"{pid:<8} {sim_gen:>28.4f} {sim_c1:>18.4f}  {better}")

print("="*65)
avg_gen = np.mean([r["sim_gen"] for r in results])
avg_c1  = np.mean([r["sim_c1"]  for r in results])
print(f"Reference-Only 평균:  {avg_gen:.4f}")
print(f"원본 c1 평균:          {avg_c1:.4f}")
print(f"IP-Adapter Plus 이전: 0.5714")

if avg_gen > 0.5714:
    print(f"✅ IP-Adapter보다 향상 (+{avg_gen-0.5714:.4f})")
else:
    print(f"❌ 여전히 IP-Adapter와 비슷하거나 낮음")
