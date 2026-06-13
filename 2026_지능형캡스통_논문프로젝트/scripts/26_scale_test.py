"""
ID 규모별 Stage-wise 매칭 효과 검증
- 50명, 100명, 200명, 500명 비교
- 각 규모에서 baseline vs Stage-wise

c6 생성 이미지가 없는 ID는 새로 생성
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
GEN_DIR = f"{PROJECT_DIR}/outputs/clip_gen_c6_large"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

os.makedirs(GEN_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)
TOP_K = 5

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

# 모든 카메라 + c6 query 있는 ID
all_valid = []
for pid in sorted(gallery_by_id.keys()):
    cams = set(gallery_by_id[pid].keys())
    if cams >= {"c1","c2","c3","c4","c5","c6"} and "c6" in query_by_id[pid]:
        all_valid.append(pid)

print(f"총 유효 ID: {len(all_valid)}명")

# Pose pool (다른 ID들의 c6)
pose_pool = []
for pid in all_valid[200:]:  # 평가 외 ID
    if "c6" in gallery_by_id[pid]:
        pose_pool.append(gallery_by_id[pid]["c6"][0])
if not pose_pool:
    pose_pool = [gallery_by_id[all_valid[0]]["c6"][0]]

# CLIP
print("CLIP 로드...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def feat(p):
    img = Image.open(p).convert("RGB") if isinstance(p, str) else p
    inp = clip_proc(images=img, return_tensors="pt").to(device)
    f = clip_model.get_image_features(**inp)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 생성 모델 (필요 시만 로드)
gen_pipe = None
openpose = None

def get_gen_pipe():
    global gen_pipe, openpose
    if gen_pipe is None:
        print("생성 모델 로드...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
        )
        gen_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
        gen_pipe.scheduler = DDIMScheduler.from_config(gen_pipe.scheduler.config)
        gen_pipe = gen_pipe.to(device)
        gen_pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CACHE_DIR,
        )
        gen_pipe.set_ip_adapter_scale(0.7)
        openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
    return gen_pipe, openpose

def generate(src_path, pose_ref_path, save_path):
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    pipe, op = get_gen_pipe()
    src = Image.open(src_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = op(pose).resize(SIZE, Image.LANCZOS)
    g = torch.Generator(device=device).manual_seed(42)
    r = pipe(
        prompt="a photo of a person, full body, CCTV camera view",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=skel, ip_adapter_image=src,
        num_inference_steps=25, guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        generator=g, width=SIZE[0], height=SIZE[1],
    ).images[0]
    r.save(save_path)
    return r

def evaluate_scale(n_ids):
    """N명 규모에서 3가지 방식 평가"""
    test_ids = all_valid[:n_ids]
    print(f"\n생성 이미지 준비 ({n_ids}명)...")

    # 갤러리 feature 준비
    gen_feats = []
    c1_feats = []
    for i, pid in enumerate(test_ids):
        c1 = sorted(gallery_by_id[pid]["c1"])[0]
        gen_path = f"{GEN_DIR}/{pid}_c6.png"
        if not os.path.exists(gen_path):
            generate(c1, pose_pool[i % len(pose_pool)], gen_path)
        c1_feats.append(feat(c1))
        gen_feats.append(feat(gen_path))
        if (i+1) % 50 == 0:
            print(f"  진행: {i+1}/{n_ids}")
    c1_feats = np.array(c1_feats)
    gen_feats = np.array(gen_feats)

    # 평가
    print(f"매칭 평가 중...")
    correct_baseline = 0
    correct_expanded = 0
    correct_stage = 0

    for i, pid in enumerate(test_ids):
        q_feat = feat(sorted(query_by_id[pid]["c6"])[0])

        sims_c1 = q_feat @ c1_feats.T
        sims_gen = q_feat @ gen_feats.T

        # 방식 1: c1만
        if test_ids[sims_c1.argmax()] == pid:
            correct_baseline += 1

        # 방식 2: c1 + 생성 (단순)
        all_sims = np.concatenate([sims_c1, sims_gen])
        all_ids = test_ids + test_ids
        if all_ids[all_sims.argmax()] == pid:
            correct_expanded += 1

        # 방식 3: Stage-wise
        topk = np.argsort(-sims_gen)[:TOP_K]
        topk_ids = [test_ids[j] for j in topk]
        c1_sims_topk = [sims_c1[test_ids.index(tid)] for tid in topk_ids]
        if topk_ids[np.argmax(c1_sims_topk)] == pid:
            correct_stage += 1

    n = len(test_ids)
    return {
        "n": n,
        "baseline": correct_baseline / n,
        "expanded": correct_expanded / n,
        "stage": correct_stage / n,
    }

# 규모별 실행
print("\n" + "="*70)
print("ID 규모별 Stage-wise 매칭 효과 검증")
print("="*70)

results = []
for n in [50, 100, 200]:
    if n > len(all_valid):
        print(f"\n[{n}명] 스킵 (유효 ID 부족)")
        continue
    r = evaluate_scale(n)
    results.append(r)
    print(f"\n[{r['n']}명]")
    print(f"  Baseline (c1만):      {r['baseline']*100:5.1f}%")
    print(f"  단순 확장 (c1+생성):  {r['expanded']*100:5.1f}%")
    print(f"  Stage-wise:           {r['stage']*100:5.1f}%")
    print(f"  → Stage 효과: {(r['stage']-r['baseline'])*100:+.1f}%p")

print("\n" + "="*70)
print("종합 결과")
print("="*70)
print(f"{'규모':<8} {'Baseline':<12} {'단순확장':<12} {'Stage-wise':<12} {'향상':<10}")
for r in results:
    diff = (r['stage'] - r['baseline']) * 100
    print(f"{r['n']:<8} {r['baseline']*100:>9.1f}%  {r['expanded']*100:>9.1f}%  {r['stage']*100:>9.1f}%  {diff:+.1f}%p")
