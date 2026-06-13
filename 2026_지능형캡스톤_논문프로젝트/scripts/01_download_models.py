"""
Step 1-1: 필요한 사전학습 모델 다운로드
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download

CACHE_DIR = os.path.expanduser("~/reid-gallery-expansion/checkpoints")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("[1/4] Stable Diffusion v1.5 다운로드 (~4GB)")
print("=" * 60)
snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir=CACHE_DIR,
    allow_patterns=[
        "*.json",
        "*.txt",
        "tokenizer/*",
        "text_encoder/*.safetensors",
        "unet/*.safetensors",
        "vae/*.safetensors",
        "scheduler/*",
        "feature_extractor/*",
        "model_index.json",
    ],
)
print("✅ SD v1.5 완료\n")

print("=" * 60)
print("[2/4] ControlNet OpenPose 다운로드 (~1.5GB)")
print("=" * 60)
snapshot_download(
    repo_id="lllyasviel/sd-controlnet-openpose",
    cache_dir=CACHE_DIR,
    allow_patterns=["*.json", "*.safetensors"],
)
print("✅ ControlNet OpenPose 완료\n")

print("=" * 60)
print("[3/4] IP-Adapter weights 다운로드 (~200MB)")
print("=" * 60)
hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter_sd15.safetensors",
    cache_dir=CACHE_DIR,
)
snapshot_download(
    repo_id="h94/IP-Adapter",
    cache_dir=CACHE_DIR,
    allow_patterns=["models/image_encoder/*"],
)
print("✅ IP-Adapter 완료\n")

print("=" * 60)
print("[4/4] OpenPose detector (controlnet-aux용, ~600MB)")
print("=" * 60)
snapshot_download(
    repo_id="lllyasviel/Annotators",
    cache_dir=CACHE_DIR,
    allow_patterns=["body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"],
)
print("✅ OpenPose detector 완료\n")

print("🎉 모든 모델 다운로드 완료!")
print(f"저장 위치: {CACHE_DIR}")