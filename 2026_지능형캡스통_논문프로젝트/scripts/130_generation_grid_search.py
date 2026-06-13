#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
130_generation_grid_search.py ─ 최적의 생성 파라미터 찾기 (5명 샘플링)
"""
import os, glob, torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

# ===== 환경 설정 =====
CACHE_DIR = "./checkpoints"
OUT_DIR = "./outputs/param_test"
os.makedirs(OUT_DIR, exist_ok=True)

# 테스트할 파라미터 조합
IPA_SCALES = [0.4, 0.6, 0.8]
CN_SCALES = [0.7, 1.0]

# 실사 특화 베이스 모델 추천
BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE" 
device, dtype = "cuda", torch.float16

print("1. 모델 로드 중...")
cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL, controlnet=cn, cache_dir=CACHE_DIR, torch_dtype=dtype, safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors")
pipe.to(device)

# ===== 임의의 테스트 데이터 (경로는 연구자님 환경에 맞게 수정) =====
# 테스트용 C1 이미지(외관)와 C6 포즈 이미지(뼈대)를 5세트 준비한다고 가정
test_samples = [
    {"pid": "0001", "c1_img": "path_to_c1_img_1.jpg", "c6_pose": "path_to_c6_pose_1.jpg"},
    # ... 5명 정도 추가 ...
]

print("2. 파라미터 조합 테스트 시작...")
for sample in test_samples:
    pid = sample["pid"]
    # IP-Adapter가 특징을 잘 잡도록 LANCZOS 리사이즈 적용
    c1_img = Image.open(sample["c1_img"]).convert("RGB").resize((512, 1024), Image.LANCZOS)
    c6_pose = Image.open(sample["c6_pose"]).convert("RGB").resize((512, 768), Image.LANCZOS)
    
    for ipa in IPA_SCALES:
        for cn_scale in CN_SCALES:
            print(f"생성 중 -> PID: {pid} | IPA: {ipa} | CN: {cn_scale}")
            
            pipe.set_ip_adapter_scale(ipa)
            g = torch.Generator(device).manual_seed(42) # 동일한 시드 고정 (비교를 위해)
            
            gen_img = pipe(
                prompt="a photo of a person, full body shot, standing, taken from a surveillance CCTV camera, top-down angle, highly detailed, photorealistic, correct anatomy",
                negative_prompt="blurry, low quality, deformed, multiple people, extra limbs, missing limbs, bad anatomy, bad proportions, disfigured, mutated",
                image=c6_pose,
                ip_adapter_image=c1_img,
                controlnet_conditioning_scale=cn_scale,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=512, height=768, generator=g
            ).images[0]
            
            # 파라미터가 적힌 파일명으로 저장
            filename = f"{OUT_DIR}/{pid}_IPA{ipa}_CN{cn_scale}.png"
            gen_img.save(filename)

print("✅ 테스트 생성 완료! 폴더의 이미지를 눈으로 비교해 보세요.")