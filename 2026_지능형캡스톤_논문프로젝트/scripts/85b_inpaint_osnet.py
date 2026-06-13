"""
85b_inpaint_osnet.py

OSNet (CNN) + 가림 복원 테스트
+ 가림 위치 3가지 비교 (하단/중간/상단)

기대: CNN은 로컬 feature → 가림에 더 취약 → 복원 효과 더 클 수 있음
ViT는 0.971 → 0.974 (+0.003)이었는데, CNN에서는 차이가 더 클지?

5장 × 3가지 가림 = 15번 inpaint. 5분 내외.
"""
import os, sys, glob, torch, numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import random

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_GALLERY = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15/bounding_box_test"
OUT = f"{PROJECT_DIR}/outputs/inpaint_osnet"
os.makedirs(OUT, exist_ok=True)

device = "cuda"
N_TEST = 5
OCCLUDE_RATIO = 0.35

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

# ===== 가림 3가지 위치 =====
def add_occlusion(img, position="bottom", ratio=0.35):
    """position: bottom/middle/top"""
    w, h = img.size
    occluded = img.copy()
    draw = ImageDraw.Draw(occluded)
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)
    block_h = int(h * ratio)
    if position == "bottom":
        y0 = h - block_h; y1 = h
    elif position == "top":
        y0 = 0; y1 = block_h
    else:  # middle
        y0 = (h - block_h) // 2; y1 = y0 + block_h
    draw.rectangle([0, y0, w, y1], fill=(0, 0, 0))
    mask_draw.rectangle([0, y0, w, y1], fill=(255, 255, 255))
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
    SIZE = (512, 512)
    img_r = image.resize(SIZE, Image.LANCZOS)
    mask_r = mask.resize(SIZE, Image.LANCZOS)
    result = pipe(
        prompt="a photo of a person, full body, surveillance camera",
        negative_prompt="blurry, deformed, multiple people",
        image=img_r, mask_image=mask_r,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    return result.resize(image.size, Image.LANCZOS)

# ===== OSNet Re-ID =====
def load_osnet():
    sys.path.insert(0, os.path.expanduser("~/osnet-reid"))
    try:
        import torchreid
        model = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=751,
            loss='softmax', pretrained=False)
        # Market-1501 학습 weight 로드
        weight_paths = [
            f"{CKPT}/osnet_x1_0_market.pth",
            f"{CKPT}/osnet_x1_0_market1501.pth",
            os.path.expanduser("~/osnet-reid/osnet_x1_0_market.pth"),
        ]
        loaded = False
        for wp in weight_paths:
            if os.path.exists(wp):
                torchreid.utils.load_pretrained_weights(model, wp)
                print(f"  OSNet weight: {wp}")
                loaded = True; break
        if not loaded:
            # torchreid 자동 다운로드 시도
            model = torchreid.models.build_model(
                name='osnet_x1_0', num_classes=751,
                loss='softmax', pretrained=True)
            print("  OSNet weight: torchreid pretrained")
        model = model.eval().to(device)
        return model, "torchreid"
    except ImportError:
        print("  torchreid 없음, torchvision ResNet50 fallback")
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Identity()  # feature extractor
        model = model.eval().to(device)
        return model, "resnet50"

osnet_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def feat_osnet(model, img, model_type="torchreid"):
    if isinstance(img, str): img = Image.open(img).convert("RGB")
    t = osnet_tf(img).unsqueeze(0).to(device)
    if model_type == "torchreid":
        f = model(t)
    else:
        f = model(t)
    f = torch.nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

# ===== CLIP-ReID (ViT) 비교용 =====
def load_clipreid():
    sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="dukemtmcreid"; cfg.TEST.NECK_FEAT='before'
    wp = f"{CKPT}/clipreid_duke_nosie.pth"
    cfg.TEST.WEIGHT = wp
    try: m = make_model(cfg, num_class=702, camera_num=0, view_num=1)
    except: m = make_model(cfg, num_class=702, camera_num=8, view_num=1)
    m.load_param(wp); return m.eval().to(device)

vit_tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

@torch.no_grad()
def feat_vit(model, img):
    if isinstance(img, str): img = Image.open(img).convert("RGB")
    t = vit_tf(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    f = torch.nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

def make_compare(orig, occ, mask, restored, save_path):
    W, H = 256, 512; gap = 5
    canvas = Image.new("RGB", (W*4+gap*3, H), (255,255,255))
    for i, im in enumerate([orig, occ, mask, restored]):
        canvas.paste(im.resize((W, H), Image.LANCZOS), (i*(W+gap), 0))
    canvas.save(save_path)

# ===== Main =====
def main():
    files = sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg"))
    samples = []
    for f in files:
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        if cam == "c1": samples.append(f)
        if len(samples) >= N_TEST: break

    # 생성
    print("SD Inpainting 로드...")
    pipe = build_inpaint_pipe()
    positions = ["bottom", "middle", "top"]
    cache = {}  # (file, pos) → (occ, mask, restored)
    for f in samples:
        orig = Image.open(f).convert("RGB")
        for pos in positions:
            occ, mask = add_occlusion(orig, pos, OCCLUDE_RATIO)
            print(f"  복원: {os.path.basename(f)} [{pos}]")
            restored = inpaint(pipe, occ, mask)
            cache[(f, pos)] = (occ, mask, restored)
            make_compare(orig, occ, mask, restored,
                         f"{OUT}/{os.path.basename(f).replace('.jpg','')}_{pos}.png")
    del pipe; torch.cuda.empty_cache()

    # Re-ID 비교: OSNet(CNN) vs CLIP-ReID(ViT)
    print("\n모델 로드...")
    osnet, osnet_type = load_osnet()
    print("  CLIP-ReID 로드...")
    vit = load_clipreid()

    for model_name, model, feat_fn in [
        ("OSNet(CNN)", osnet, lambda img: feat_osnet(model, img, osnet_type)),
        ("CLIP-ReID(ViT)", vit, lambda img: feat_vit(model, img)),
    ]:
        print(f"\n{'='*75}")
        print(f"  모델: {model_name}")
        print(f"{'='*75}")
        print(f"{'위치':<10}{'orig↔occ':<14}{'orig↔rest':<14}{'회복':<12}")
        print("-" * 50)
        for pos in positions:
            sims_occ, sims_rest = [], []
            for f in samples:
                orig = Image.open(f).convert("RGB")
                occ, mask, restored = cache[(f, pos)]
                f_orig = feat_fn(orig)
                f_occ = feat_fn(occ)
                f_rest = feat_fn(restored)
                sims_occ.append(f_orig @ f_occ)
                sims_rest.append(f_orig @ f_rest)
            avg_occ = np.mean(sims_occ)
            avg_rest = np.mean(sims_rest)
            recover = avg_rest - avg_occ
            print(f"{pos:<10}{avg_occ:<14.4f}{avg_rest:<14.4f}{recover:+.4f}")
        print("-" * 50)

    print(f"\n비교이미지: {OUT}/")
    print("""
해석:
  CNN 회복 > ViT 회복 → CNN이 가림에 취약, 복원 효과 큼 → CNN 대상으로 논문
  middle 회복 > bottom → 상체 가림이 더 치명적, 복원 가치 큼
  회복 큼 (>0.01) → Inpainting Re-ID 논문 가능!
  회복 작음 (<0.005) → 두 모델 다 가림에 강건 → 다른 길 필요""")

    del osnet, vit; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()