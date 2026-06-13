"""
카메라 특성 반영 생성 (Img2Img + ControlNet + IP-Adapter)

흐름:
1. 타겟 카메라(c6)의 임의 이미지 → base (카메라 특성)
2. 그 base에 외형(c1) 주입 (IP-Adapter)
3. 자세 제어 (ControlNet OpenPose)
4. strength로 카메라 유지 정도 조절

[Baseline] c1 실제 갤러리만
[Ours]     생성 c6 갤러리만 (카메라 특성 반영)
Query:     실제 c6
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/camera_aware_c6"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
TARGET_CAM = "c6"
STRENGTH = 0.75   # 0.5 (카메라 유지 강함) ~ 0.9 (외형 강함)

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터
print("데이터 로드...")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1','0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# c1 + c6 query 있는 ID
valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and TARGET_CAM in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= 100:  # 일단 100명
        break

print(f"평가 ID: {len(valid_ids)}명")

# 타겟 카메라 reference pool (평가 ID 제외)
target_cam_pool = []
for pid in sorted(gallery_by_id.keys()):
    if pid in set(valid_ids):
        continue
    if TARGET_CAM in gallery_by_id[pid]:
        target_cam_pool.append(gallery_by_id[pid][TARGET_CAM][0])
    if len(target_cam_pool) >= 30:
        break
print(f"{TARGET_CAM} reference pool: {len(target_cam_pool)}장")

# === 생성 파이프라인 (Img2Img + ControlNet) ===
print("\n생성 모델 로드...")
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
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CACHE_DIR,
)
pipe.set_ip_adapter_scale(0.7)
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
print("✅ 로드 완료\n")

def generate_camera_aware(c1_path, cam_ref_path, save_path):
    """
    c1: 외형 참조
    cam_ref: 타겟 카메라의 이미지 (카메라 특성 + 자세)
    """
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    
    # 외형
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    # 카메라 base (cam_ref 자체가 base가 됨)
    cam_base = Image.open(cam_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    # 자세 skeleton
    skel = openpose(cam_base).resize(SIZE, Image.LANCZOS)
    
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=cam_base,           # base = 카메라 특성
        control_image=skel,       # 자세
        ip_adapter_image=c1_img,  # 외형
        strength=STRENGTH,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        generator=gen,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    return result

# === CLIP-ReID ===
print("CLIP-ReID 로드...")
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'market1501'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

reid = make_model(cfg, num_class=751, camera_num=6, view_num=1)
reid.load_param(cfg.TEST.WEIGHT)
reid = reid.eval().to(device)

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

@torch.no_grad()
def feat(img_or_path, cam_id):
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = reid(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# === 갤러리/Query 구성 ===
print(f"\n생성 + feature 추출 ({len(valid_ids)}명)...")
baseline_feats = []
ours_feats = []
query_feats = []
ids = []

for i, pid in enumerate(tqdm(valid_ids)):
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c6_query_path = sorted(query_by_id[pid][TARGET_CAM])[0]
    
    # Baseline: c1 실제
    baseline_feats.append(feat(c1_path, cam_to_id["c1"]))
    
    # Ours: 생성 c6 (카메라 특성 반영)
    cam_ref = target_cam_pool[i % len(target_cam_pool)]
    save_path = f"{GEN_DIR}/{pid}_gen_{TARGET_CAM}.png"
    gen_img = generate_camera_aware(c1_path, cam_ref, save_path)
    ours_feats.append(feat(gen_img, cam_to_id[TARGET_CAM]))
    
    # Query
    query_feats.append(feat(c6_query_path, cam_to_id[TARGET_CAM]))
    ids.append(pid)

baseline_feats = np.array(baseline_feats)
ours_feats = np.array(ours_feats)
query_feats = np.array(query_feats)

# === 평가 ===
def eval_rank1(gf, qf):
    sims = qf @ gf.T
    correct = 0
    for i in range(len(qf)):
        if sims[i].argmax() == i:  # 같은 인덱스 = 같은 ID
            correct += 1
    return correct / len(qf)

def avg_self_sim(gf, qf):
    return np.mean([qf[i] @ gf[i] for i in range(len(qf))])

r1_base = eval_rank1(baseline_feats, query_feats)
r1_ours = eval_rank1(ours_feats, query_feats)
avg_base = avg_self_sim(baseline_feats, query_feats)
avg_ours = avg_self_sim(ours_feats, query_feats)

print("\n" + "="*70)
print(f"카메라 특성 반영 생성 실험 (strength={STRENGTH})")
print("="*70)
print(f"평가 ID: {len(valid_ids)}명, c1 → query {TARGET_CAM}")
print()
print(f"{'':<35}{'Rank-1':<15}{'self-sim 평균':<15}")
print(f"{'Baseline (c1 실제)':<35}{r1_base*100:<15.2f}{avg_base:<15.4f}")
print(f"{'Ours (생성 c6, 카메라반영)':<35}{r1_ours*100:<15.2f}{avg_ours:<15.4f}")
print(f"{'향상':<35}{(r1_ours-r1_base)*100:<+15.2f}{avg_ours-avg_base:<+15.4f}")

print(f"\n[해석]")
if r1_ours > r1_base:
    print(f"✅ 카메라 특성 반영 생성이 c1 실제보다 더 잘 매칭")
elif r1_ours == r1_base:
    print(f"= 동일 - 카메라 특성 반영 효과 미미")
else:
    print(f"❌ c1 실제가 여전히 더 강함")
    print(f"   생성 이미지 카메라 특성 반영했지만 외형 변형 영향 큼")
