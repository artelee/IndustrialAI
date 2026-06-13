from huggingface_hub import hf_hub_download
import os
CACHE = os.path.expanduser('~/reid-gallery-expansion/checkpoints')

# FaceID Plus weights
hf_hub_download(
    repo_id='h94/IP-Adapter-FaceID',
    filename='ip-adapter-faceid-plus_sd15.bin',
    cache_dir=CACHE,
)
# 필요한 face encoder
hf_hub_download(
    repo_id='h94/IP-Adapter-FaceID', 
    filename='ip-adapter-faceid-plus_sd15_lora.safetensors',
    cache_dir=CACHE,
)
print('다운로드 완료')