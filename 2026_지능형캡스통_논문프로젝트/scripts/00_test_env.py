import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n--- 라이브러리 import 테스트 ---")
import diffusers
print(f"diffusers: {diffusers.__version__}")

import transformers
print(f"transformers: {transformers.__version__}")

from controlnet_aux import OpenposeDetector
print("controlnet-aux: OK")

import cv2
print(f"opencv: {cv2.__version__}")

print("\n✅ 환경 세팅 완료!")