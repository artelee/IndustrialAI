"""
85_inpaint_test.py

가림 복원 빠른 테스트:
1. Market 정상 이미지에 인위적 가림 추가 (하단 30~40% 가림)
2. SD Inpainting으로 복원
3. Re-ID feature 비교:
   - 원본 feature (정답)
   - 가림 feature (가린 상태)
   - 복원 feature (inpaint 후)
   → 복원 > 가림 이면 "inpainting이 Re-ID에 도움" 증명

5장만 빠르게. 비교이미지: [원본 | 가림 | mask | 복원]
"""
import os, sys, glob, torch, numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import random

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_GALLERY = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15/bounding_box_test"
OUT = f"{PROJECT_DIR}/outputs/inpaint_test"
os.makedirs(OUT, exist_ok=True)

device = "cuda"
N_TEST = 5
OCCLUDE_RATIO = 0.35  # 하단 35% 가림

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

# ===== 가림 생성 =====
def add_occlusion(img, ratio=0.35):
    """하단 ratio 비율을 검정으로 가림 → 가림 이미지 + mask 반환"""
    w, h = img.size
    occluded = img.copy()
    draw = ImageDraw.Draw(occluded)
    y_start = int(h * (1 - ratio))
    draw.rectangle([0, y_start, w, h], fill=(0, 0, 0))
    # mask: 가린 부분=흰(복원 대상), 나머지=검정(유지)
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([0, y_start, w, h], fill=(255, 255, 255))
    return occluded, mask

# ===== SD Inpainting =====
def build_inpaint_pipe():
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

@torch.no_grad()
def inpaint(pipe, image, mask):
    """가림 부분만 복원"""
    SIZE = (512, 512)  # inpaint는 정사각형이 안정적
    img_r = image.resize(SIZE, Image.LANCZOS)
    mask_r = mask.resize(SIZE, Image.LANCZOS)
    result = pipe(
        prompt="a photo of a person, full body, legs, feet, pants, shoes, walking, surveillance camera",
        negative_prompt="blurry, deformed, multiple people, cropped",
        image=img_r, mask_image=mask_r,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    # 원래 크기로 복원
    return result.resize(image.size, Image.LANCZOS)

# ===== Re-ID feature =====
def load_nosie():
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="dukemtmcreid"; cfg.TEST.NECK_FEAT='before'
    cfg.TEST.WEIGHT=f"{CKPT}/clipreid_duke_nosie.pth"
    try: m = make_model(cfg, num_class=702, camera_num=0, view_num=1)
    except: m = make_model(cfg, num_class=702, camera_num=8, view_num=1)
    m.load_param(cfg.TEST.WEIGHT); return m.eval().to(device)

reid_tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                     T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

@torch.no_grad()
def feat(model, img):
    if isinstance(img, str): img = Image.open(img).convert("RGB")
    t = reid_tf(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    f = torch.nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

def make_compare(orig, occluded, mask, restored, save_path):
    """[원본 | 가림 | mask | 복원] 비교"""
    W, H = 256, 512; gap = 5
    imgs = [orig, occluded, mask, restored]
    canvas = Image.new("RGB", (W*4 + gap*3, H), (255,255,255))
    for i, im in enumerate(imgs):
        canvas.paste(im.resize((W, H), Image.LANCZOS), (i*(W+gap), 0))
    canvas.save(save_path)

# ===== Main =====
def main():
    # 샘플 선택 (c1, 사람 잘 보이는 것)
    files = sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg"))
    samples = []
    for f in files:
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        if cam == "c1":
            samples.append(f)
        if len(samples) >= N_TEST: break

    # Phase 1: 가림 + 복원 생성
    print("SD Inpainting 로드...")
    pipe = build_inpaint_pipe()

    originals, occludeds, restoreds = [], [], []
    for i, f in enumerate(samples):
        orig = Image.open(f).convert("RGB")
        occ, mask = add_occlusion(orig, OCCLUDE_RATIO)
        print(f"  [{i+1}/{N_TEST}] 복원 중: {os.path.basename(f)}")
        restored = inpaint(pipe, occ, mask)
        originals.append(orig); occludeds.append(occ); restoreds.append(restored)
        make_compare(orig, occ, mask, restored, f"{OUT}/{os.path.basename(f).replace('.jpg','_compare.png')}")

    del pipe; torch.cuda.empty_cache()

    # Phase 2: Re-ID feature 비교
    print("\nRe-ID feature 비교...")
    reid = load_nosie()

    print(f"\n{'file':<25}{'orig↔occ':<12}{'orig↔rest':<12}{'회복':<10}")
    print("-" * 60)
    total_occ, total_rest = 0, 0
    for i, f in enumerate(samples):
        f_orig = feat(reid, originals[i])
        f_occ = feat(reid, occludeds[i])
        f_rest = feat(reid, restoreds[i])
        sim_occ = f_orig @ f_occ    # 원본↔가림
        sim_rest = f_orig @ f_rest   # 원본↔복원
        recover = sim_rest - sim_occ
        total_occ += sim_occ; total_rest += sim_rest
        print(f"{os.path.basename(f):<25}{sim_occ:<12.4f}{sim_rest:<12.4f}{recover:+.4f}")
    print("-" * 60)
    n = len(samples)
    print(f"{'평균':<25}{total_occ/n:<12.4f}{total_rest/n:<12.4f}{(total_rest-total_occ)/n:+.4f}")
    print(f"\n비교이미지: {OUT}/")
    print("[원본 | 가림({:.0f}%) | mask | 복원]".format(OCCLUDE_RATIO*100))

    improve = (total_rest - total_occ) / n
    if improve > 0:
        print(f"\n✅ 복원이 가림보다 원본에 {improve:.4f} 더 가까움")
        print("→ SD Inpainting이 Re-ID feature를 회복시킴!")
        print("→ 가림 복원 방향으로 생성형 AI 활용 가능")
    else:
        print(f"\n❌ 복원이 가림보다 나쁨 ({improve:.4f})")
        print("→ Inpainting도 ID 보존 못함")

    del reid; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()