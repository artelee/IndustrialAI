"""
80a_v2_centered.py

수정: skeleton을 화면 중앙에 꽉 차게 정규화 → 인물 중앙 배치
이전 문제(인물 구석/작게) 해결.

추가: normalize_skeleton()
  - openpose 출력(흑배경+컬러 관절선)에서 비어있지 않은 영역(=skeleton) 찾아
  - 그 bbox를 화면 중앙에 목표 비율(높이 85%)로 재배치

먼저 c5만 5장 테스트 → 인물 중앙·크게 나오는지 확인 후 전체.
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
OUT_QUERY = f"{PROJECT_DIR}/outputs/ipa_query_gen"
os.makedirs(OUT_GALLERY, exist_ok=True); os.makedirs(OUT_QUERY, exist_ok=True)

device="cuda"; SIZE=(384,768); random.seed(42)
TARGET_CAMS = ["c5"]   # ★ 먼저 c5만 테스트. 잘 되면 ["c2","c3","c4","c5","c6"]
TEST_LIMIT = 5         # ★ 테스트: 페어당 5장만. 전체는 None

def parse(f):
    p=os.path.basename(f).split("_"); return p[0],p[1][:2]

def normalize_skeleton(skel_img, target_h_ratio=0.85):
    """
    openpose skeleton(흑배경+컬러선)을 화면 중앙에 꽉 차게 재배치.
    1. 비검정 픽셀(=skeleton) bbox 찾기
    2. 중앙 정렬 + 목표 높이 비율로 스케일
    """
    arr = np.array(skel_img)  # (H,W,3)
    mask = arr.sum(axis=2) > 20   # 거의 검정 아닌 곳 = skeleton
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return skel_img  # skeleton 없으면 원본
    y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = arr[y0:y1+1, x0:x1+1]
    ch, cw = crop.shape[:2]
    W, H = SIZE
    # 목표 높이
    target_h = int(H * target_h_ratio)
    scale = target_h / ch
    new_w = max(1, int(cw * scale)); new_h = target_h
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    # 검정 캔버스 중앙에 배치
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
    pipe.set_ip_adapter_scale(1.0)   # 외형 강하게 (이전 0.8서 인물 약했음)
    pipe.enable_attention_slicing()
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    return pipe, openpose

def collect_pose_pool(gd, qd, cam, exclude, n=30):
    paths=[]
    for f in sorted(glob.glob(f"{qd}/*.jpg"))+sorted(glob.glob(f"{gd}/*.jpg")):
        pid,c=parse(f)
        if pid in ('-1','0000'): continue
        if c==cam and pid not in exclude: paths.append(f)
    random.shuffle(paths); return paths[:n]

@torch.no_grad()
def generate(pipe, openpose, id_path, pose_ref_path, save_path, save_skel=None):
    if os.path.exists(save_path): return
    id_img = Image.open(id_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img)
    skel = normalize_skeleton(skel.resize(SIZE, Image.LANCZOS))  # ★ 중앙 정규화
    if save_skel:
        skel.save(save_skel)
    # ★ embeds 방식 (이 버전에서 ip_adapter_image=[...] 는 tuple 에러, embeds만 동작)
    emb = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=id_img, ip_adapter_image_embeds=None,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    result = pipe(
        prompt="a photo of one person, full body, standing, surveillance camera, centered",
        negative_prompt="blurry, low quality, deformed, multiple people, cropped, empty, background only",
        ip_adapter_image_embeds=emb,
        image=skel, controlnet_conditioning_scale=1.0,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)

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
        os.makedirs(f"{PROJECT_DIR}/debug_skel/{tc}", exist_ok=True)
        for pid in tqdm(cvi[tc], desc=f"[방식2] {tc}"):
            c1_path=sorted(gby[pid]["c1"])[0]
            pose_ref=random.choice(pose_pool)
            generate(pipe, openpose, c1_path, pose_ref,
                     f"{OUT_GALLERY}/{tc}/{pid}_gen_{tc}.png",
                     save_skel=f"{PROJECT_DIR}/debug_skel/{tc}/{pid}_skel.png")
    print(f"\n생성: {OUT_GALLERY}")
    print(f"skeleton 디버그: {PROJECT_DIR}/debug_skel/")
    print("→ 생성물 + skeleton 둘 다 눈으로 확인 (인물 중앙·크게 나왔는지)")

if __name__=="__main__":
    main()