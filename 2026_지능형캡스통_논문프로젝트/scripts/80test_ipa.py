"""
80test_ipa.py — IP-Adapter + ControlNet 1장만 생성해서 동작 확인

에러 'tuple object has no attribute shape' 진단:
- diffusers 0.37 + IP-Adapter + ControlNet 조합 호환성 확인
- 여러 호출 방식 시도해서 되는 걸 찾음
"""
import os, sys, torch, glob
from PIL import Image

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_GALLERY = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15/bounding_box_test"
device="cuda"; SIZE=(384,768)

import diffusers
print("diffusers:", diffusers.__version__)

from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler)
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
pipe.set_ip_adapter_scale(0.8)
print("IP-Adapter 로드 완료")
print("unet encoder_hid_proj:", type(getattr(pipe.unet, "encoder_hid_proj", None)))

# 샘플 이미지
sample = sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg"))[0]
id_img = Image.open(sample).convert("RGB").resize(SIZE, Image.LANCZOS)
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
skel = openpose(id_img).resize(SIZE, Image.LANCZOS)
gen = torch.Generator(device=device).manual_seed(42)

def try_call(label, **extra):
    try:
        out = pipe(prompt="a photo of a person, full body",
                   negative_prompt="blurry, deformed",
                   image=skel, controlnet_conditioning_scale=1.0,
                   num_inference_steps=20, guidance_scale=7.5,
                   generator=torch.Generator(device=device).manual_seed(42),
                   width=SIZE[0], height=SIZE[1], **extra).images[0]
        out.save(f"{PROJECT_DIR}/test_{label}.png")
        print(f"✅ [{label}] 성공 → test_{label}.png")
        return True
    except Exception as e:
        print(f"❌ [{label}] 실패: {type(e).__name__}: {str(e)[:120]}")
        return False

print("\n--- 호출 방식 테스트 ---")
try_call("plain_img", ip_adapter_image=id_img)
try_call("list_img", ip_adapter_image=[id_img])

# embeds 방식 (가장 호환성 높음)
try:
    emb = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=id_img, ip_adapter_image_embeds=None,
        device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    try_call("embeds", ip_adapter_image_embeds=emb)
except Exception as e:
    print(f"❌ [embeds 준비] 실패: {type(e).__name__}: {str(e)[:120]}")

print("\n→ 성공한 방식으로 80a 수정하면 됨")