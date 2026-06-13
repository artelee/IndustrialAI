"""
역방향 Cross-Domain
Source: Market 학습 (CLIP-ReID Market weight)
Target: Duke 평가

1. Duke c1 → c2~c8 생성 (carry over - 한 번만)
2. Re-ranking 평가 (Top-K=5, alpha=0.7)
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
GEN_BASE = f"{PROJECT_DIR}/outputs/duke_c1base_gen"
DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
GALLERY_DIR = f"{DUKE_DIR}/bounding_box_test"
QUERY_DIR = f"{DUKE_DIR}/query"

os.makedirs(GEN_BASE, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
STRENGTH = 0.4
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6", "c8"]  # c7 제외 (너무 적음)
TOP_K = 5
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

# 데이터
print("Duke 데이터 로드...")
gallery_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    if pid in ('-1','0000'): continue
    gallery_by_id[pid][cam].append(f)

query_by_id = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
    pid, cam = parse(f)
    query_by_id[pid][cam].append(f)

# 카메라별 ID
cam_valid_ids = {}
for cam in TARGET_CAMS:
    ids = []
    for pid in sorted(gallery_by_id.keys()):
        if "c1" in gallery_by_id[pid] and cam in query_by_id[pid]:
            ids.append(pid)
    cam_valid_ids[cam] = ids

print(f"\n평가 ID 수:")
for cam in TARGET_CAMS:
    print(f"  c1 → {cam}: {len(cam_valid_ids[cam])}명")

# Pose pool
pose_pools = {}
for cam in TARGET_CAMS:
    excluded = set()
    for ids in cam_valid_ids.values():
        excluded.update(ids)
    pool = []
    for pid in sorted(gallery_by_id.keys()):
        if pid in excluded: continue
        if cam in gallery_by_id[pid]:
            pool.append(gallery_by_id[pid][cam][0])
        if len(pool) >= 30: break
    pose_pools[cam] = pool

# === 생성 ===
print("\n" + "="*70)
print("STEP 1: Duke c1 → c2~c8 생성")
print("="*70)
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
print("✅ 로드 완료")

def generate(c1_path, pose_ref_path, save_path):
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=c1_img, control_image=skel,
        strength=STRENGTH, num_inference_steps=30,
        guidance_scale=7.5, controlnet_conditioning_scale=0.8,
        generator=gen, width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    return result

for target_cam in TARGET_CAMS:
    save_dir = f"{GEN_BASE}/{target_cam}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[{target_cam}] 생성 중... ({len(cam_valid_ids[target_cam])}명)")
    for i, pid in enumerate(tqdm(cam_valid_ids[target_cam])):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        pose_ref = pose_pools[target_cam][i % len(pose_pools[target_cam])]
        save_path = f"{save_dir}/{pid}_gen_{target_cam}.png"
        generate(c1_path, pose_ref, save_path)

del pipe, controlnet, openpose
torch.cuda.empty_cache()

# === 평가 ===
print("\n" + "="*70)
print("STEP 2: Re-ranking 평가 (Market 학습 → Duke 평가)")
print("="*70)

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

# Market 학습 모델 (6 카메라), Duke는 8 카메라
# 모델 구조는 6 카메라로 빌드, c1~c6만 사용
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)
print("✅ 로드 완료")

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

# c1~c6는 직접 매핑, c7, c8은 c6으로 (안전)
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5,'c7':5,'c8':5}

@torch.no_grad()
def feat(path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# Feature 추출
print("\nFeature 추출...")
all_feats = {}
for cam in TARGET_CAMS:
    print(f"  {cam}...")
    c1_f, gen_f, q_f = [], [], []
    for pid in tqdm(cam_valid_ids[cam], leave=False):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
        q_path = sorted(query_by_id[pid][cam])[0]
        c1_f.append(feat(c1_path, cam_to_id["c1"]))
        gen_f.append(feat(gen_path, cam_to_id[cam]))
        q_f.append(feat(q_path, cam_to_id[cam]))
    all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))

# 평가
def rerank_eval(c1_feats, gen_feats, query_feats, top_k, alpha):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_correct = sum(1 for i in range(N) if s1[i].argmax() == i)
    
    rerank_correct = 0
    for i in range(N):
        topk_idx = np.argsort(-s1[i])[:min(top_k, N)]
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = alpha * s1_topk + (1 - alpha) * s2
        if topk_idx[final.argmax()] == i:
            rerank_correct += 1
    
    return baseline_correct / N, rerank_correct / N

print("\n" + "="*80)
print(f"역방향 Cross-Domain 결과 (Market 학습 → Duke 평가)")
print(f"Top-K={TOP_K}, alpha={ALPHA}")
print("="*80)
print(f"\n{'Pair':<10}{'N':<8}{'Baseline':<15}{'Ours':<15}{'향상':<10}")
print("-"*80)

total_gain, wins = 0, 0
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    r1_b, r1_o = rerank_eval(c1_f, gen_f, q_f, TOP_K, ALPHA)
    gain = (r1_o - r1_b) * 100
    total_gain += gain
    if gain > 0: wins += 1
    mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
    print(f"c1→{cam:<7}{len(q_f):<8}{r1_b*100:<15.2f}{r1_o*100:<15.2f}{gain:+.2f} {mark}")

print(f"\n평균 향상: {total_gain/len(TARGET_CAMS):+.2f}%p")
print(f"향상 페어: {wins}/{len(TARGET_CAMS)}")