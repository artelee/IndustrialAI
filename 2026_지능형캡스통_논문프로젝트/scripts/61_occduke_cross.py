
"""
Cross-Domain: Market 학습 → Occluded-Duke 평가
가장 어려운 시나리오 (Cross-Domain + 가림)

Oracle 분석으로 잠재력 측정
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
GEN_BASE = f"{PROJECT_DIR}/outputs/occduke_c1base_gen"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
GALLERY_DIR = f"{OCC_DIR}/bounding_box_test"
QUERY_DIR = f"{OCC_DIR}/query"

os.makedirs(GEN_BASE, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
STRENGTH = 0.4
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6", "c8"]
TOP_K = 5
ALPHA = 0.7

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1]

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
print("STEP 1: Occ-Duke c1 → c2~c8 생성")
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
    print(f"\n[{target_cam}] 생성... ({len(cam_valid_ids[target_cam])}명)")
    for i, pid in enumerate(tqdm(cam_valid_ids[target_cam])):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        pose_ref = pose_pools[target_cam][i % len(pose_pools[target_cam])]
        save_path = f"{save_dir}/{pid}_gen_{target_cam}.png"
        generate(c1_path, pose_ref, save_path)

del pipe, controlnet, openpose
torch.cuda.empty_cache()

# === 평가 ===
print("\n" + "="*70)
print("STEP 2: Market 학습 → Occ-Duke 평가 (Cross-Domain)")
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

print("CLIP-ReID Market 학습 로드...")
model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param(cfg.TEST.WEIGHT)
model = model.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])
# Occ-Duke c1~c8 → Market 6 카메라로 매핑 (c7,c8은 c6으로)
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
    c1_f, gen_f, q_f = [], [], []
    for pid in tqdm(cam_valid_ids[cam], desc=cam, leave=False):
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_BASE}/{cam}/{pid}_gen_{cam}.png"
        q_path = sorted(query_by_id[pid][cam])[0]
        c1_f.append(feat(c1_path, cam_to_id["c1"]))
        gen_f.append(feat(gen_path, cam_to_id[cam]))
        q_f.append(feat(q_path, cam_to_id[cam]))
    all_feats[cam] = (np.array(c1_f), np.array(gen_f), np.array(q_f))

# Oracle 분석
def eval_oracle(c1_feats, gen_feats, query_feats):
    N = len(query_feats)
    s1 = query_feats @ c1_feats.T
    baseline_top1 = s1.argmax(axis=1)
    baseline_correct = baseline_top1 == np.arange(N)
    
    ours_correct = baseline_correct.copy()
    recovered = 0
    in_topk = 0
    for i in range(N):
        if baseline_correct[i]: continue
        topk_idx = np.argsort(-s1[i])[:TOP_K]
        if i in topk_idx:
            in_topk += 1
        s2 = query_feats[i] @ gen_feats[topk_idx].T
        s1_topk = s1[i][topk_idx]
        final = ALPHA * s1_topk + (1 - ALPHA) * s2
        if topk_idx[final.argmax()] == i:
            ours_correct[i] = True
            recovered += 1
    
    return {
        'N': N,
        'BL_correct': baseline_correct.sum(),
        'BL_wrong': (~baseline_correct).sum(),
        'in_topk': in_topk,
        'recovered': recovered,
        'Oracle_correct': ours_correct.sum(),
    }

print("\n" + "="*100)
print(f"Occluded-Duke Cross-Domain Oracle 분석")
print(f"Market 학습 → Occ-Duke 평가")
print("="*100)
print(f"\n{'Pair':<10}{'N':<6}{'BL맞':<8}{'BL틀':<8}{'TopK내':<10}{'회복':<8}{'Oracle맞':<12}{'Baseline':<12}{'Oracle':<12}{'향상':<10}")
print("-"*100)

total = defaultdict(int)
for cam in TARGET_CAMS:
    c1_f, gen_f, q_f = all_feats[cam]
    r = eval_oracle(c1_f, gen_f, q_f)
    bl = r['BL_correct'] / r['N'] * 100
    oracle = r['Oracle_correct'] / r['N'] * 100
    gain = oracle - bl
    mark = "✅" if gain > 0 else ("=" if gain == 0 else "❌")
    print(f"c1→{cam:<7}{r['N']:<6}{r['BL_correct']:<8}{r['BL_wrong']:<8}"
          f"{r['in_topk']:<10}{r['recovered']:<8}{r['Oracle_correct']:<12}"
          f"{bl:<12.2f}{oracle:<12.2f}{gain:+.2f} {mark}")
    for k in r:
        total[k] += r[k]

bl_avg = total['BL_correct'] / total['N'] * 100
oracle_avg = total['Oracle_correct'] / total['N'] * 100
gain_avg = oracle_avg - bl_avg
print("-"*100)
print(f"{'합계':<10}{total['N']:<6}{total['BL_correct']:<8}{total['BL_wrong']:<8}"
      f"{total['in_topk']:<10}{total['recovered']:<8}{total['Oracle_correct']:<12}"
      f"{bl_avg:<12.2f}{oracle_avg:<12.2f}{gain_avg:+.2f}")

recover_rate = total['recovered'] / max(total['in_topk'], 1) * 100
overall_recover = total['recovered'] / max(total['BL_wrong'], 1) * 100
print(f"\n[회복률]")
print(f"Top-K 내 회복률: {total['recovered']}/{total['in_topk']} = {recover_rate:.1f}%")
print(f"전체 회복률:     {total['recovered']}/{total['BL_wrong']} = {overall_recover:.1f}%")

print(f"\n[Cross-Domain 비교]")
print(f"Market → Duke (일반):       Oracle 잠재력 +2.48%p")
print(f"Market → Occ-Duke (가림+CD): Oracle 잠재력 {gain_avg:+.2f}%p")
