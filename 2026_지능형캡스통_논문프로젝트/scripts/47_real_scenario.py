"""
진짜 산업 시나리오 평가
갤러리 100명, query 1명 → Top-1 매칭

Baseline: c1 실제 갤러리만
Ours:     생성 c? 갤러리만 (c1 없음)

모든 cross-camera 페어 + Same-Domain + Cross-Domain
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
GEN_BASE = f"{PROJECT_DIR}/outputs/c1base_gen_all"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_BASE, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
STRENGTH = 0.4
NUM_IDS = 100

TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

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

# 카메라별 ID
cam_valid_ids = {}
for target_cam in TARGET_CAMS:
    ids = []
    for pid in sorted(gallery_by_id.keys()):
        if "c1" in gallery_by_id[pid] and target_cam in query_by_id[pid]:
            ids.append(pid)
        if len(ids) >= NUM_IDS:
            break
    cam_valid_ids[target_cam] = ids

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

print("\n평가 ID 수:")
for cam in TARGET_CAMS:
    print(f"  c1 → {cam}: {len(cam_valid_ids[cam])}명")

# === 생성 ===
print("\n" + "="*70)
print("STEP 1: 모든 카메라 생성")
print("="*70)
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
    print(f"\n[{target_cam}] 생성 중...")
    for i, pid in enumerate(tqdm(cam_valid_ids[target_cam])):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        pose_ref = pose_pools[target_cam][i % len(pose_pools[target_cam])]
        save_path = f"{save_dir}/{pid}_gen_{target_cam}.png"
        generate(c1_path, pose_ref, save_path)

del pipe, controlnet, openpose
torch.cuda.empty_cache()

# === 평가 ===
def load_clipreid(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [12, 12]
    cfg.MODEL.SIE_CAMERA = True
    cfg.MODEL.SIE_COE = 1.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path
    cfg.TEST.NECK_FEAT = 'before'
    m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
cam_to_id = {'c1':0,'c2':1,'c3':2,'c4':3,'c5':4,'c6':5}

@torch.no_grad()
def feat(model, path, cam_id):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def eval_pair(model, target_cam, valid_ids):
    """100명 갤러리, 100명 query → Top-1"""
    # 갤러리 feature
    baseline_gf = []  # c1 실제 100명
    ours_gf = []      # 생성 c? 100명
    query_f = []      # query c? 100명
    
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(query_by_id[pid][target_cam])[0]
        baseline_gf.append(feat(model, c1_path, cam_to_id["c1"]))
        ours_gf.append(feat(model, gen_path, cam_to_id[target_cam]))
        query_f.append(feat(model, q_path, cam_to_id[target_cam]))
    
    bf = np.array(baseline_gf)
    of = np.array(ours_gf)
    qf = np.array(query_f)
    
    # 같은 인덱스 = 같은 사람
    def r1(gf):
        sims = qf @ gf.T
        return sum(1 for i in range(len(qf)) if sims[i].argmax() == i) / len(qf)
    def self_sim(gf):
        return np.mean([qf[i] @ gf[i] for i in range(len(qf))])
    
    return r1(bf), r1(of), self_sim(bf), self_sim(of)

# Same-Domain
print("\n" + "="*70)
print("STEP 2: Same-Domain (Market 학습)")
print("="*70)
model = load_clipreid(
    f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth",
    "market1501", 751, 6
)
same_results = {}
for cam in TARGET_CAMS:
    print(f"\n[c1 → {cam}]")
    r1_b, r1_o, sim_b, sim_o = eval_pair(model, cam, cam_valid_ids[cam])
    same_results[cam] = (r1_b, r1_o, sim_b, sim_o)
    print(f"  Baseline Rank-1: {r1_b*100:.2f}% (self-sim {sim_b:.4f})")
    print(f"  Ours     Rank-1: {r1_o*100:.2f}% (self-sim {sim_o:.4f})")

# Cross-Domain
print("\n" + "="*70)
print("STEP 3: Cross-Domain (Duke 학습)")
print("="*70)
del model
torch.cuda.empty_cache()
model = load_clipreid(
    f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth",
    "dukemtmcreid", 702, 8
)
cross_results = {}
for cam in TARGET_CAMS:
    print(f"\n[c1 → {cam}]")
    r1_b, r1_o, sim_b, sim_o = eval_pair(model, cam, cam_valid_ids[cam])
    cross_results[cam] = (r1_b, r1_o, sim_b, sim_o)
    print(f"  Baseline Rank-1: {r1_b*100:.2f}% (self-sim {sim_b:.4f})")
    print(f"  Ours     Rank-1: {r1_o*100:.2f}% (self-sim {sim_o:.4f})")

# 종합표
print("\n\n" + "="*90)
print("종합 결과")
print("="*90)
print(f"\n[Same-Domain] Market 학습 → Market 평가 (각 페어 100명 갤러리)")
print(f"{'Pair':<10}{'Baseline R1':<15}{'Ours R1':<15}{'향상':<12}{'B sim':<10}{'O sim':<10}")
print("-"*90)
for cam, (rb, ro, sb, so) in same_results.items():
    diff = (ro-rb)*100
    mark = "✅" if ro > rb else ("=" if ro == rb else "❌")
    print(f"c1→{cam:<7}{rb*100:<15.2f}{ro*100:<15.2f}{diff:+.2f} {mark:<5}{sb:<10.4f}{so:<10.4f}")

print(f"\n[Cross-Domain] Duke 학습 → Market 평가")
print(f"{'Pair':<10}{'Baseline R1':<15}{'Ours R1':<15}{'향상':<12}{'B sim':<10}{'O sim':<10}")
print("-"*90)
for cam, (rb, ro, sb, so) in cross_results.items():
    diff = (ro-rb)*100
    mark = "✅" if ro > rb else ("=" if ro == rb else "❌")
    print(f"c1→{cam:<7}{rb*100:<15.2f}{ro*100:<15.2f}{diff:+.2f} {mark:<5}{sb:<10.4f}{so:<10.4f}")
