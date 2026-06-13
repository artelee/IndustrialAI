"""
c1을 base로 자세만 c6 변환 (외형 정확 유지)

흐름:
- base = c1 이미지 (외형 보존)
- control = c6 사람의 자세 (skeleton)
- strength 3개 비교: 어느 게 최적?

[Baseline] c1 실제 갤러리
[Ours]     c1 base + c6 자세 생성 (strength별)
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
GEN_DIR = f"{PROJECT_DIR}/outputs/c1base_gen_c6"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
TARGET_CAM = "c6"
NUM_IDS = 100  # 100명만
STRENGTHS = [0.3, 0.4, 0.5]

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

valid_ids = []
for pid in sorted(gallery_by_id.keys()):
    if "c1" in gallery_by_id[pid] and TARGET_CAM in query_by_id[pid]:
        valid_ids.append(pid)
    if len(valid_ids) >= NUM_IDS:
        break

print(f"평가 ID: {len(valid_ids)}명\n")

# c6 자세 reference pool
pose_pool = []
for pid in sorted(gallery_by_id.keys()):
    if pid in set(valid_ids): continue
    if TARGET_CAM in gallery_by_id[pid]:
        pose_pool.append(gallery_by_id[pid][TARGET_CAM][0])
    if len(pose_pool) >= 30: break
print(f"Pose reference pool: {len(pose_pool)}장\n")

# 생성 파이프라인
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

def generate_c1_base(c1_path, pose_ref_path, strength, save_path):
    """c1 base + c6 자세 (IP-Adapter 안 씀, 외형 정확 유지)"""
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)
    
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=c1_img,           # base = c1 (외형)
        control_image=skel,     # 자세 = c6
        strength=strength,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=gen,
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    return result

# CLIP-ReID
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

# Baseline + Query feature
print("Baseline + Query feature 추출...")
baseline_feats = []
query_feats = []
for pid in tqdm(valid_ids):
    c1_path = sorted(gallery_by_id[pid]["c1"])[0]
    c6_query = sorted(query_by_id[pid][TARGET_CAM])[0]
    baseline_feats.append(feat(c1_path, cam_to_id["c1"]))
    query_feats.append(feat(c6_query, cam_to_id[TARGET_CAM]))
baseline_feats = np.array(baseline_feats)
query_feats = np.array(query_feats)

# 평가 함수
def eval_rank1(gf, qf):
    sims = qf @ gf.T
    correct = sum(1 for i in range(len(qf)) if sims[i].argmax() == i)
    return correct / len(qf)

def avg_self_sim(gf, qf):
    return np.mean([qf[i] @ gf[i] for i in range(len(qf))])

r1_base = eval_rank1(baseline_feats, query_feats)
avg_base = avg_self_sim(baseline_feats, query_feats)

# Strength별 실험
print("\n" + "="*70)
print(f"Strength 비교 실험: c1 base + c6 pose")
print("="*70)
print(f"평가 ID: {len(valid_ids)}명, c1 → query {TARGET_CAM}")
print(f"\n각 strength마다 별도 갤러리 → 별도 평가\n")

results = {}
for strength in STRENGTHS:
    print(f"\n[Strength {strength}] 생성 중...")
    save_dir = f"{GEN_DIR}/s{int(strength*100)}"
    os.makedirs(save_dir, exist_ok=True)
    
    ours_feats = []
    for i, pid in enumerate(tqdm(valid_ids)):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        pose_ref = pose_pool[i % len(pose_pool)]
        save_path = f"{save_dir}/{pid}_gen_{TARGET_CAM}.png"
        gen_img = generate_c1_base(c1_path, pose_ref, strength, save_path)
        ours_feats.append(feat(gen_img, cam_to_id[TARGET_CAM]))
    ours_feats = np.array(ours_feats)
    
    r1 = eval_rank1(ours_feats, query_feats)
    avg = avg_self_sim(ours_feats, query_feats)
    results[strength] = (r1, avg)
    print(f"  → Rank-1: {r1*100:.2f}%, self-sim: {avg:.4f}")

# 종합
print("\n" + "="*70)
print("종합 결과")
print("="*70)
print(f"{'설정':<35}{'Rank-1':<15}{'self-sim':<15}{'향상':<10}")
print(f"{'Baseline (c1 실제)':<35}{r1_base*100:<10.2f}     {avg_base:<15.4f}{'-':<10}")
print("-"*70)
for s, (r1, avg) in results.items():
    diff = (r1 - r1_base) * 100
    mark = "✅" if r1 > r1_base else ("=" if r1 == r1_base else "❌")
    print(f"{'Ours (strength=' + str(s) + ')':<35}{r1*100:<10.2f}     {avg:<15.4f}{diff:+.2f} {mark}")

print(f"\n[해석]")
best_s = max(results, key=lambda s: results[s][0])
best_r1 = results[best_s][0]
if best_r1 > r1_base:
    print(f"✅ 최적 strength: {best_s} (Rank-1 {best_r1*100:.2f}%, 향상 {(best_r1-r1_base)*100:+.2f}%p)")
else:
    print(f"❌ 모든 strength에서 baseline보다 낮음")
    print(f"   c1 실제 데이터의 매칭 강도가 생성보다 큼")
