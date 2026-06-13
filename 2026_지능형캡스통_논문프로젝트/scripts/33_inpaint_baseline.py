
"""
실험 1: Inpainting 가능성 검증
가림된 query → 하단 30% inpainting → 매칭
Baseline 대비 향상폭 측정
"""

import os, sys, glob, torch, numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

from diffusers import StableDiffusionInpaintPipeline

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
GALLERY_DIR = f"{OCC_DIR}/bounding_box_test"
QUERY_DIR = f"{OCC_DIR}/query"
INPAINT_DIR = f"{PROJECT_DIR}/outputs/inpainted_queries"
os.makedirs(INPAINT_DIR, exist_ok=True)

device = "cuda"
dtype = torch.float16

def parse(f):
    name = os.path.basename(f).split(".")[0]
    parts = name.split("_")
    return parts[0], parts[1]

# === Inpainting 파이프라인 ===
print("Stable Diffusion Inpainting 로드...")
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)
print("✅ Inpainting 로드 완료")

# === 단순 마스크: 하단 30% ===
def create_bottom_mask(img_size, ratio=0.3):
    """이미지 하단 ratio 만큼 mask (white = inpaint 대상)"""
    w, h = img_size
    mask = Image.new("L", (w, h), 0)  # black = keep
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    occ_h = int(h * ratio)
    draw.rectangle([0, h - occ_h, w, h], fill=255)  # white = inpaint
    return mask

# === Inpainting 실행 ===
def inpaint(img_path, save_path):
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    
    img = Image.open(img_path).convert("RGB")
    # SD inpainting은 512x512 권장
    orig_size = img.size
    img_resized = img.resize((512, 512), Image.LANCZOS)
    mask = create_bottom_mask((512, 512), ratio=0.3)
    
    gen = torch.Generator(device=device).manual_seed(42)
    result = inpaint_pipe(
        prompt="a photo of a person, full body, clear",
        negative_prompt="blurry, deformed, multiple people",
        image=img_resized,
        mask_image=mask,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    
    result = result.resize(orig_size, Image.LANCZOS)
    result.save(save_path)
    return result

# === 데이터 ===
print("\n데이터 로드...")
gallery_files = sorted(glob.glob(f"{GALLERY_DIR}/*.jpg"))
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

all_cams = set()
for f in gallery_files + query_files:
    _, cam = parse(f)
    all_cams.add(cam)
cam_list = sorted(all_cams)
cam_to_id = {c: i for i, c in enumerate(cam_list)}

train_ids = set()
for f in glob.glob(f"{OCC_DIR}/bounding_box_train/*.jpg"):
    pid, _ = parse(f)
    if pid not in ('-1', '0000'):
        train_ids.add(pid)
NUM_CLASSES = len(train_ids)

# === CLIP-ReID ===
print("CLIP-ReID 로드...")
cfg.MODEL.NAME = 'ViT-B-16'
cfg.MODEL.STRIDE_SIZE = [12, 12]
cfg.MODEL.SIE_CAMERA = True
cfg.MODEL.SIE_COE = 1.0
cfg.MODEL.ID_LOSS_TYPE = 'softmax'
cfg.INPUT.SIZE_TRAIN = [256, 128]
cfg.INPUT.SIZE_TEST = [256, 128]
cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
cfg.DATASETS.NAMES = 'occ_duke'
cfg.TEST.WEIGHT = f"{PROJECT_DIR}/checkpoints/clipreid_occduke/ViT-B-16_60.pth"
cfg.TEST.NECK_FEAT = 'before'

reid_model = make_model(cfg, num_class=NUM_CLASSES, camera_num=len(cam_list), view_num=1)
reid_model.load_param(cfg.TEST.WEIGHT)
reid_model = reid_model.eval().to(device)

transform = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),
    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
])

@torch.no_grad()
def feat(img, cam_id):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    c = torch.tensor([cam_id]).to(device)
    f = reid_model(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# === 갤러리 feature (한 번만) ===
print("\n갤러리 feature 추출...")
gf, gp, gc = [], [], []
for f in tqdm(gallery_files):
    pid, cam = parse(f)
    gf.append(feat(f, cam_to_id[cam]))
    gp.append(pid)
    gc.append(cam)
gf, gp, gc = np.array(gf), np.array(gp), np.array(gc)

# === Query feature (1) Baseline + (2) Inpainted ===
print("\nBaseline query feature 추출...")
qf_base, qp, qc = [], [], []
for f in tqdm(query_files):
    pid, cam = parse(f)
    qf_base.append(feat(f, cam_to_id[cam]))
    qp.append(pid)
    qc.append(cam)
qf_base = np.array(qf_base)
qp, qc = np.array(qp), np.array(qc)

print("\nInpainting 진행 (시간 걸림)...")
qf_inp = []
for i, f in enumerate(tqdm(query_files)):
    pid, cam = parse(f)
    save_path = f"{INPAINT_DIR}/{os.path.basename(f)}"
    img_inp = inpaint(f, save_path)
    qf_inp.append(feat(img_inp, cam_to_id[cam]))
qf_inp = np.array(qf_inp)

# === 평가 함수 ===
def evaluate(qf, qp, qc):
    sims = qf @ gf.T
    all_cmc, all_AP = [], []
    for q_idx in range(len(qp)):
        q_pid, q_cam = qp[q_idx], qc[q_idx]
        order = np.argsort(-sims[q_idx])
        keep = ~((gp == q_pid) & (gc == q_cam))
        keep[gp == "0000"] = False
        keep[gp == "-1"] = False
        order_valid = order[keep[order]]
        matches = (gp[order_valid] == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])
        num_rel = matches.sum()
        tmp = matches.cumsum()
        tmp = [x/(i+1.0) for i, x in enumerate(tmp)]
        tmp = np.asarray(tmp) * matches
        all_AP.append(tmp.sum() / num_rel)
    cmc = np.array(all_cmc).mean(axis=0)
    return np.mean(all_AP), cmc[0], cmc[4], cmc[9]

mAP_b, r1_b, r5_b, r10_b = evaluate(qf_base, qp, qc)
mAP_o, r1_o, r5_o, r10_o = evaluate(qf_inp, qp, qc)

print("\n" + "="*65)
print("실험 1 결과: Inpainting 가능성 검증")
print("="*65)
print(f"{'':<30}{'mAP':<10}{'Rank-1':<10}{'Rank-5':<10}{'Rank-10':<10}")
print(f"{'Baseline (가림 그대로)':<30}{mAP_b*100:<10.2f}{r1_b*100:<10.2f}{r5_b*100:<10.2f}{r10_b*100:<10.2f}")
print(f"{'Ours (하단 30% Inpaint)':<30}{mAP_o*100:<10.2f}{r1_o*100:<10.2f}{r5_o*100:<10.2f}{r10_o*100:<10.2f}")
print(f"\n향상폭:")
print(f"  mAP:    {(mAP_o-mAP_b)*100:+.2f}%p")
print(f"  Rank-1: {(r1_o-r1_b)*100:+.2f}%p")
