
"""
실험: Keypoint 기반 선택적 Inpainting
- YOLO-x pose로 가림 검출
- Box conf < 0.5 → 검출 실패, 원본 사용
- 가림 부위만 SD-Inpaint
- 가림 없으면 원본 사용
"""

import os, sys, glob, torch, numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm import tqdm
from ultralytics import YOLO

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
INPAINT_DIR = f"{PROJECT_DIR}/outputs/keypoint_inpainted"
MASK_DIR = f"{PROJECT_DIR}/outputs/keypoint_masks"
os.makedirs(INPAINT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

device = "cuda"
dtype = torch.float16

BOX_CONF_THRESHOLD = 0.5
KPT_CONF_THRESHOLD = 0.5

def parse(f):
    name = os.path.basename(f).split(".")[0]
    parts = name.split("_")
    return parts[0], parts[1]

# YOLO-x pose
print("YOLO-x pose 로드...")
pose_model = YOLO('yolov8x-pose.pt')

BODY_REGIONS = {
    "head":  {"kpts": [0,1,2,3,4],          "y_range": (0.00, 0.20)},
    "upper": {"kpts": [5,6,7,8,9,10],       "y_range": (0.15, 0.55)},
    "lower": {"kpts": [11,12,13,14,15,16],  "y_range": (0.50, 1.00)},
}

def detect_occlusion_mask(img):
    """가림 영역 mask 생성"""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    
    results = pose_model(np.array(img), verbose=False)
    if len(results)==0 or results[0].keypoints is None or results[0].keypoints.xy.shape[0]==0:
        return mask, [], "no_detection"
    
    boxes = results[0].boxes
    if boxes is None or len(boxes.conf) == 0:
        return mask, [], "no_box"
    
    box_confs = boxes.conf.cpu().numpy()
    if box_confs.max() < BOX_CONF_THRESHOLD:
        return mask, [], "low_box_conf"
    
    best_idx = box_confs.argmax()
    conf = results[0].keypoints.conf[best_idx].cpu().numpy()
    
    draw = ImageDraw.Draw(mask)
    occluded_regions = []
    
    for name, info in BODY_REGIONS.items():
        part_conf = conf[info["kpts"]]
        visible = (part_conf > KPT_CONF_THRESHOLD).sum()
        total = len(info["kpts"])
        if visible < total / 2:
            y_start, y_end = info["y_range"]
            draw.rectangle([0, int(h*y_start), w, int(h*y_end)], fill=255)
            occluded_regions.append(name)
    
    return mask, occluded_regions, "ok"

# SD Inpaint
print("Stable Diffusion Inpaint 로드...")
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

def inpaint_image(img, mask):
    orig_size = img.size
    img_r = img.resize((512, 512), Image.LANCZOS)
    mask_r = mask.resize((512, 512), Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = inpaint_pipe(
        prompt="a photo of a person, full body, clear, realistic",
        negative_prompt="blurry, deformed, multiple people, occluded",
        image=img_r,
        mask_image=mask_r,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    return result.resize(orig_size, Image.LANCZOS)

# CLIP-ReID
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
    if pid not in ('-1','0000'):
        train_ids.add(pid)

reid = make_model(cfg, num_class=len(train_ids), camera_num=len(cam_list), view_num=1)
reid.load_param(cfg.TEST.WEIGHT)
reid = reid.eval().to(device)

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
    f = reid(t, cam_label=c)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# 갤러리 feature
print("\n갤러리 feature 추출...")
gf, gp, gc = [], [], []
for f in tqdm(gallery_files):
    pid, cam = parse(f)
    gf.append(feat(f, cam_to_id[cam]))
    gp.append(pid)
    gc.append(cam)
gf, gp, gc = np.array(gf), np.array(gp), np.array(gc)

# Query 처리
print("\nQuery 가림 검출 + 선택적 Inpainting...")
qf_base, qf_inp, qp, qc = [], [], [], []
stats = {"detected_occluded": 0, "detected_clean": 0, "low_box_conf": 0, "no_detection": 0, "no_box": 0}
region_stats = {"head":0, "upper":0, "lower":0}

for f in tqdm(query_files):
    pid, cam = parse(f)
    img = Image.open(f).convert("RGB")
    
    qf_base.append(feat(img, cam_to_id[cam]))
    
    mask, regions, status = detect_occlusion_mask(img)
    
    if status == "ok" and regions:
        stats["detected_occluded"] += 1
        for r in regions:
            region_stats[r] += 1
        save_path = f"{INPAINT_DIR}/{os.path.basename(f)}"
        if os.path.exists(save_path):
            img_inp = Image.open(save_path).convert("RGB")
        else:
            img_inp = inpaint_image(img, mask)
            img_inp.save(save_path)
            mask.save(f"{MASK_DIR}/{os.path.basename(f)}")
        qf_inp.append(feat(img_inp, cam_to_id[cam]))
    else:
        if status == "ok":
            stats["detected_clean"] += 1
        else:
            stats[status] += 1
        qf_inp.append(qf_base[-1])
    
    qp.append(pid)
    qc.append(cam)

qf_base = np.array(qf_base)
qf_inp = np.array(qf_inp)
qp, qc = np.array(qp), np.array(qc)

n = len(query_files)
print(f"\n검출 통계 (총 {n}장):")
print(f"  가림 검출됨 + Inpaint:   {stats['detected_occluded']} ({100*stats['detected_occluded']/n:.1f}%)")
print(f"  검출됨 but 가림 없음:    {stats['detected_clean']} ({100*stats['detected_clean']/n:.1f}%)")
print(f"  박스 신뢰도 낮음 (skip): {stats['low_box_conf']} ({100*stats['low_box_conf']/n:.1f}%)")
print(f"  검출 실패 (skip):        {stats['no_detection']+stats['no_box']} ({100*(stats['no_detection']+stats['no_box'])/n:.1f}%)")
print(f"\n가림 부위 분포 (중복 가능):")
for k, v in region_stats.items():
    print(f"  {k}: {v}")

# 평가
def evaluate(qf):
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

mAP_b, r1_b, r5_b, r10_b = evaluate(qf_base)
mAP_o, r1_o, r5_o, r10_o = evaluate(qf_inp)

print("\n" + "="*65)
print("실험 결과: Keypoint 기반 선택적 Inpainting")
print("="*65)
print(f"{'':<35}{'mAP':<10}{'Rank-1':<10}{'Rank-5':<10}{'Rank-10':<10}")
print(f"{'Baseline (원본)':<35}{mAP_b*100:<10.2f}{r1_b*100:<10.2f}{r5_b*100:<10.2f}{r10_b*100:<10.2f}")
print(f"{'Ours (선택적 Inpaint)':<35}{mAP_o*100:<10.2f}{r1_o*100:<10.2f}{r5_o*100:<10.2f}{r10_o*100:<10.2f}")
print(f"\n향상폭:")
print(f"  mAP:     {(mAP_o-mAP_b)*100:+.2f}%p")
print(f"  Rank-1:  {(r1_o-r1_b)*100:+.2f}%p")
print(f"  Rank-5:  {(r5_o-r5_b)*100:+.2f}%p")
print(f"  Rank-10: {(r10_o-r10_b)*100:+.2f}%p")
