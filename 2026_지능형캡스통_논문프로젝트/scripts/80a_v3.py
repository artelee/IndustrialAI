"""
80a_v3.py

수정:
1. enable_attention_slicing() 제거 (IP-Adapter attention과 충돌 → tuple 에러 원인)
2. 비교 이미지 저장: [원본 c1 | pose skeleton | 생성물] 가로로 붙여 한 장
3. 두 명 생성 방지: negative prompt 강화 + "solo, single person" 강조

먼저 c5 5장 테스트. compare/ 폴더에 비교 이미지 저장됨.
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import random

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
OUT_GALLERY = f"{PROJECT_DIR}/outputs/ipa_gallery_gen"
OUT_COMPARE = f"{PROJECT_DIR}/outputs/compare"
os.makedirs(OUT_GALLERY, exist_ok=True); os.makedirs(OUT_COMPARE, exist_ok=True)

device="cuda"; SIZE=(384,768); random.seed(42)
TARGET_CAMS = ["c5"]
TEST_LIMIT = 5

def parse(f):
    p=os.path.basename(f).split("_"); return p[0],p[1][:2]

def normalize_skeleton(skel_img, target_h_ratio=0.85):
    arr = np.array(skel_img)
    mask = arr.sum(axis=2) > 20
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return skel_img
    y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = arr[y0:y1+1, x0:x1+1]
    ch, cw = crop.shape[:2]
    W, H = SIZE
    target_h = int(H * target_h_ratio)
    scale = target_h / ch
    new_w = max(1, int(cw * scale)); new_h = target_h
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", SIZE, (0,0,0))
    px = (W - new_w)//2; py = (H - new_h)//2
    canvas.paste(crop_img, (px, py))
    return canvas

def build_pipe():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", cache_dir=CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet, cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CKPT)
    pipe.set_ip_adapter_scale(1.0)
    # ★ enable_attention_slicing() 제거 — IP-Adapter attention과 충돌
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    return pipe, openpose

def collect_pose_pool(gd, qd, cam, exclude, n=30):
    paths=[]
    for f in sorted(glob.glob(f"{qd}/*.jpg"))+sorted(glob.glob(f"{gd}/*.jpg")):
        pid,c=parse(f)
        if pid in ('-1','0000'): continue
        if c==cam and pid not in exclude: paths.append(f)
    random.shuffle(paths); return paths[:n]

def make_compare(orig_path, skel_img, gen_img, save_path):
    """[원본 | pose | 생성] 가로로 붙여 저장"""
    orig = Image.open(orig_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    W, H = SIZE
    canvas = Image.new("RGB", (W*3 + 20, H), (255,255,255))
    canvas.paste(orig, (0, 0))
    canvas.paste(skel_img.resize(SIZE), (W+10, 0))
    canvas.paste(gen_img.resize(SIZE), (W*2+20, 0))
    canvas.save(save_path)

@torch.no_grad()
def generate(pipe, openpose, id_path, pose_ref_path, save_path, compare_path=None):
    if os.path.exists(save_path) and not compare_path:
        return
    id_img = Image.open(id_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img)
    skel = normalize_skeleton(skel.resize(SIZE, Image.LANCZOS))
    emb = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=id_img, ip_adapter_image_embeds=None,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    result = pipe(
        prompt="a photo of one single person, solo, full body, standing, centered, surveillance camera",
        negative_prompt="two people, multiple people, group, crowd, duplicate, blurry, low quality, deformed, cropped, empty, background only",
        ip_adapter_image_embeds=emb,
        image=skel, controlnet_conditioning_scale=1.0,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    if compare_path:
        make_compare(id_path, skel, result, compare_path)

def main():
    gby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid,cam=parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid,cam=parse(f); qby[pid][cam].append(f)
    cvi={}
    for tc in TARGET_CAMS:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids)>=100: break
        cvi[tc]=ids[:TEST_LIMIT] if TEST_LIMIT else ids

    pipe, openpose = build_pipe()
    for tc in TARGET_CAMS:
        eval_pids=set(cvi[tc])
        pose_pool=collect_pose_pool(MARKET_GALLERY, MARKET_QUERY, tc, eval_pids, 30)
        if not pose_pool: print(f"{tc}: pool 없음"); continue
        os.makedirs(f"{OUT_GALLERY}/{tc}", exist_ok=True)
        os.makedirs(f"{OUT_COMPARE}/{tc}", exist_ok=True)
        for pid in tqdm(cvi[tc], desc=f"[방식2] {tc}"):
            c1_path=sorted(gby[pid]["c1"])[0]
            pose_ref=random.choice(pose_pool)
            generate(pipe, openpose, c1_path, pose_ref,
                     f"{OUT_GALLERY}/{tc}/{pid}_gen_{tc}.png",
                     compare_path=f"{OUT_COMPARE}/{tc}/{pid}_compare.png")
    print(f"\n생성: {OUT_GALLERY}")
    print(f"비교[원본|pose|생성]: {OUT_COMPARE}")
    print("→ compare/ 이미지로 외형 보존·자세 변형·두명여부 확인")

if __name__=="__main__":
    main()