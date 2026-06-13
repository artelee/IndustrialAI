"""
80a_generate_ipadapter.py

새 생성 방식: IP-Adapter 외형 주입 + 타겟 자세 강하게 (시점 크게 변형)
기존 strength 0.4 (c1 복사) 문제 해결 목적.

두 방식 모두 생성:
  [방식2 Gallery측] c1 인물 외형(IP-Adapter) + 타겟카메라 자세 → gallery_gen/
  [방식1 Query측]   query 인물 외형(IP-Adapter) + 타 카메라들 자세 → query_gen/

설정:
  base: 노이즈에서 시작 (text2img 가까이) 또는 타겟 자세 base
  IP-Adapter: ip-adapter-plus_sd15 (외형 보존 우수, 메모리상 채택본)
  ControlNet OpenPose: 타겟 자세, scale 1.0 (강하게)
  → 외형은 IP-Adapter가, 자세/시점은 ControlNet이 담당

생성량 제한: 평가 대상 ID만 (NUM_IDS=100), 페어별.
무거우므로 우선 Duke→Market 한 방향, c5(가장 효과 기대) 먼저 테스트 권장.

주의: diffusers 0.37에서 load_ip_adapter 후 enable_attention_slicing 호출.
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

OUT_GALLERY = f"{PROJECT_DIR}/outputs/ipa_gallery_gen"   # 방식2
OUT_QUERY = f"{PROJECT_DIR}/outputs/ipa_query_gen"       # 방식1
os.makedirs(OUT_GALLERY, exist_ok=True)
os.makedirs(OUT_QUERY, exist_ok=True)

device = "cuda"
NUM_IDS = 100
SIZE = (384, 768)
random.seed(42)

# 테스트할 방향/카메라 (우선 Duke→Market, 전체 페어)
#TARGET_CAMS = ["c2","c3","c4","c5","c6"]
TARGET_CAMS = ["c5"]
def parse(f):
    p=os.path.basename(f).split("_"); return p[0],p[1][:2]

def build_pipe():
    from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                           DDIMScheduler)
    from controlnet_aux import OpenposeDetector
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", cache_dir=CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet, cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # IP-Adapter 로드 (외형 주입)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus_sd15.safetensors",
                         cache_dir=CKPT)
    pipe.set_ip_adapter_scale(0.8)   # 외형 강도
    pipe.enable_attention_slicing()  # ★ load_ip_adapter 이후 호출 (0.37 순서 중요)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    return pipe, openpose

def collect_pose_pool(gallery_dir, query_dir, cam, exclude_pids, n=30):
    """타겟 카메라의 자세 reference (다른 사람들, 평가 ID 제외)"""
    paths=[]
    for f in sorted(glob.glob(f"{query_dir}/*.jpg"))+sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid,c=parse(f)
        if pid in ('-1','0000'): continue
        if c==cam and pid not in exclude_pids:
            paths.append(f)
    random.shuffle(paths)
    return paths[:n]

@torch.no_grad()
def generate(pipe, openpose, id_image_path, pose_ref_path, save_path):
    """id_image: 외형(IP-Adapter), pose_ref: 자세(ControlNet)"""
    if os.path.exists(save_path): return
    id_img = Image.open(id_image_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance camera",
        negative_prompt="blurry, low quality, deformed, multiple people, cropped",
        ip_adapter_image=id_img,         # 외형 주입
        image=skel,                       # ControlNet 자세
        controlnet_conditioning_scale=1.0,  # 자세 강하게
        num_inference_steps=30, guidance_scale=7.5,
        generator=gen, width=SIZE[0], height=SIZE[1],
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

    # 평가 대상 ID (페어별)
    cvi={}
    for tc in TARGET_CAMS:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids)>=NUM_IDS: break
        cvi[tc]=ids

    pipe, openpose = build_pipe()

    for tc in TARGET_CAMS:
        eval_pids=set(cvi[tc])
        pose_pool=collect_pose_pool(MARKET_GALLERY, MARKET_QUERY, tc, eval_pids, n=30)
        if not pose_pool:
            print(f"{tc}: 자세 pool 없음, skip"); continue

        # === 방식2: c1 인물 → 타겟(tc) 자세 → gallery ===
        os.makedirs(f"{OUT_GALLERY}/{tc}", exist_ok=True)
        for pid in tqdm(cvi[tc], desc=f"[방식2 gallery] {tc}"):
            c1_path=sorted(gby[pid]["c1"])[0]
            pose_ref=random.choice(pose_pool)
            save=f"{OUT_GALLERY}/{tc}/{pid}_gen_{tc}.png"
            generate(pipe, openpose, c1_path, pose_ref, save)

        # === 방식1: query(tc) 인물 → c1 자세 → query 측 ===
        # query를 c1 시점으로 변형 (c1 자세 pool 사용)
        c1_pose_pool=collect_pose_pool(MARKET_GALLERY, MARKET_QUERY, "c1", eval_pids, n=30)
        if c1_pose_pool:
            os.makedirs(f"{OUT_QUERY}/{tc}", exist_ok=True)
            for pid in tqdm(cvi[tc], desc=f"[방식1 query] {tc}"):
                q_path=sorted(qby[pid][tc])[0]
                pose_ref=random.choice(c1_pose_pool)
                save=f"{OUT_QUERY}/{tc}/{pid}_genc1.png"
                generate(pipe, openpose, q_path, pose_ref, save)

    print("\n생성 완료:")
    print(f"  방식2 gallery: {OUT_GALLERY}")
    print(f"  방식1 query  : {OUT_QUERY}")
    print("다음: 80b_eval로 평가")

if __name__=="__main__":
    main()