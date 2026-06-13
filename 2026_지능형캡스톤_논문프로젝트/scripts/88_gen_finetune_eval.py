"""
88_gen_finetune_eval.py

올인원: 생성 → 파인튜닝 → 평가

Phase 1: 학습 ID의 c1 → 타겟 자세로 생성
Phase 2: 원본 학습데이터(12,936장) + 생성 → CLIP-ReID 파인튜닝
Phase 3: gen_sim 측정 (테스트셋 기존 생성물로)

먼저 c3 하나만 (2017장 생성, ~30분). 효과 확인 후 전체.
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import random

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_TRAIN = f"{MARKET_DIR}/bounding_box_train"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
MARKET_TEST_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"  # 테스트 ID 생성물 (평가용)
TRAIN_GEN = f"{PROJECT_DIR}/outputs/train_gen"

device = "cuda"
SIZE = (384, 768)
STRENGTH = 0.4
GEN_CAMS = ["c3"]       # 먼저 c3만. 전체: ["c2","c3","c4","c5","c6"]
EVAL_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 30
BATCH = 32
LR = 1e-5              # ViT는 lr 작게
NUM_EVAL = 100

random.seed(42); np.random.seed(42)
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

def normalize_skeleton(skel_img, target_h_ratio=0.85):
    arr = np.array(skel_img); mask = arr.sum(axis=2) > 20
    ys, xs = np.where(mask)
    if len(ys) == 0: return skel_img
    y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
    crop = arr[y0:y1+1,x0:x1+1]; ch,cw = crop.shape[:2]
    W,H = SIZE; th = int(H*target_h_ratio); scale = th/ch
    nw = max(1,int(cw*scale))
    ci = Image.fromarray(crop).resize((nw,th),Image.LANCZOS)
    canvas = Image.new("RGB",SIZE,(0,0,0))
    canvas.paste(ci,((W-nw)//2,(H-th)//2)); return canvas


# ================================================================
# Phase 1: 생성
# ================================================================
def run_generation():
    print("=" * 70)
    print(f"Phase 1: 학습 ID 생성 (카메라: {GEN_CAMS})")
    print("=" * 70)
    
    # 이미 생성됐으면 스킵
    tby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_TRAIN}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        tby[pid][cam].append(f)
    c1_ids = [pid for pid in sorted(tby.keys()) if "c1" in tby[pid]]
    
    need_gen = False
    for tc in GEN_CAMS:
        for pid in c1_ids[:5]:
            if not os.path.exists(f"{TRAIN_GEN}/{tc}/{pid}_gen_{tc}.png"):
                need_gen = True; break
        if need_gen: break
    
    if not need_gen:
        print("  이미 생성됨, 스킵")
        return
    
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                          cache_dir=CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
        cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    
    # pose pool
    pose_pools = {}
    for tc in GEN_CAMS:
        pool = []
        for pid in tby:
            if tc in tby[pid]: pool.extend(tby[pid][tc])
        random.shuffle(pool)
        pose_pools[tc] = pool[:100]
        print(f"  {tc} pose pool: {len(pose_pools[tc])}장")
    
    gen_count = 0
    for tc in GEN_CAMS:
        if not pose_pools[tc]: continue
        os.makedirs(f"{TRAIN_GEN}/{tc}", exist_ok=True)
        for pid in tqdm(c1_ids, desc=f"[생성] {tc}"):
            save = f"{TRAIN_GEN}/{tc}/{pid}_gen_{tc}.png"
            if os.path.exists(save): gen_count += 1; continue
            c1_path = sorted(tby[pid]["c1"])[0]
            pose_ref = random.choice(pose_pools[tc])
            c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
            pose_img = Image.open(pose_ref).convert("RGB").resize(SIZE, Image.LANCZOS)
            skel = openpose(pose_img)
            skel = normalize_skeleton(skel.resize(SIZE, Image.LANCZOS))
            with torch.no_grad():
                result = pipe(
                    prompt="a photo of one person, full body, surveillance camera",
                    negative_prompt="two people, blurry, deformed",
                    image=c1_img, control_image=skel,
                    strength=STRENGTH, controlnet_conditioning_scale=0.8,
                    num_inference_steps=30, guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(42),
                    width=SIZE[0], height=SIZE[1],
                ).images[0]
            result.save(save)
            gen_count += 1
    
    del pipe, openpose; torch.cuda.empty_cache()
    print(f"  생성 완료: {gen_count}장")


# ================================================================
# Phase 2: CLIP-ReID 파인튜닝
# ================================================================
class TrainDataset(Dataset):
    def __init__(self, real_dir, gen_dir, gen_cams):
        self.samples = []
        self.pid_map = {}
        # 원본 학습 데이터 전체
        for f in sorted(glob.glob(f"{real_dir}/*.jpg")):
            pid, cam = parse(f)
            if pid in ('-1', '0000'): continue
            if pid not in self.pid_map:
                self.pid_map[pid] = len(self.pid_map)
            self.samples.append((f, self.pid_map[pid]))
        real_n = len(self.samples)
        # 생성 이미지
        gen_n = 0
        for tc in gen_cams:
            d = f"{gen_dir}/{tc}"
            if not os.path.isdir(d): continue
            for f in sorted(glob.glob(f"{d}/*.png")):
                pid = os.path.basename(f).split("_")[0]
                if pid in self.pid_map:
                    self.samples.append((f, self.pid_map[pid]))
                    gen_n += 1
        self.num_classes = len(self.pid_map)
        print(f"  학습: 원본 {real_n} + 생성 {gen_n} = {len(self.samples)}장, ID {self.num_classes}")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        f, pid = self.samples[idx]
        img = Image.open(f).convert("RGB")
        img = self.transform(img)
        return img, pid
    transform = T.Compose([
        T.Resize([256, 128]),
        T.RandomHorizontalFlip(0.5),
        T.Pad(10), T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

class CLIPReIDWithClassifier(nn.Module):
    """CLIP-ReID backbone + 새 classifier"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = None
        self.num_classes = num_classes
    
    def _init_head(self, feat_dim):
        self.classifier = nn.Linear(feat_dim, self.num_classes).to(device)
        print(f"  classifier 초기화: feat_dim={feat_dim}, classes={self.num_classes}")
    
    def forward(self, x, cam_label=None):
        # backbone은 eval 모드로 forward (깨끗한 feature 반환)
        # gradient는 여전히 흐름 (requires_grad는 eval/train과 무관)
        was_training = self.backbone.training
        self.backbone.eval()
        f = self.backbone(x, cam_label=cam_label)
        if was_training:
            self.backbone.train()
        if isinstance(f, (list, tuple)):
            f = f[0] if isinstance(f[0], torch.Tensor) else f[-1]
        if f.dim() > 2: f = f.view(f.size(0), -1)
        if self.classifier is None:
            self._init_head(f.size(1))
        logits = self.classifier(f)
        return logits, f

def load_clipreid_for_finetune(num_classes):
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME = 'ViT-B-16'; cfg.MODEL.STRIDE_SIZE = [16, 16]
    cfg.MODEL.SIE_CAMERA = False; cfg.MODEL.SIE_COE = 0.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]; cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]; cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = "market1501"
    wp = f"{CKPT}/clipreid_market_nosie.pth"
    cfg.TEST.WEIGHT = wp; cfg.TEST.NECK_FEAT = 'before'
    try: backbone = make_model(cfg, num_class=751, camera_num=0, view_num=1)
    except: backbone = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    backbone.load_param(wp)
    print(f"  CLIP-ReID nosie 로드: {wp}")
    model = CLIPReIDWithClassifier(backbone, num_classes)
    return model.to(device)

def run_finetune():
    print("\n" + "=" * 70)
    print("Phase 2: CLIP-ReID 파인튜닝")
    print("=" * 70)
    
    dataset = TrainDataset(MARKET_TRAIN, TRAIN_GEN, GEN_CAMS)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                       num_workers=0, drop_last=True)
    model = load_clipreid_for_finetune(dataset.num_classes)
    
    # 파인튜닝 전 측정 (이때 classifier 자동 초기화됨)
    print("\n--- 파인튜닝 전 ---")
    measure(model, "BEFORE")
    
    # 파인튜닝 (backbone lr 작게, classifier lr 크게)
    params = [
        {'params': model.backbone.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': LR * 10},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0; correct = 0; total = 0
        for imgs, pids in loader:
            imgs, pids = imgs.to(device), pids.to(device)
            logits, _ = model(imgs, cam_label=None)
            loss = criterion(logits, pids)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == pids).sum().item()
            total += pids.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(loader):.3f} "
                  f"acc={correct/total*100:.1f}%")
    
    # 파인튜닝 후 측정
    print("\n--- 파인튜닝 후 ---")
    measure(model, "AFTER")
    
    # weight 저장
    save_path = f"{CKPT}/clipreid_sdaware.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n  weight 저장: {save_path}")
    return model


# ================================================================
# Phase 3: 평가
# ================================================================
eval_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

@torch.no_grad()
def feat(model, path):
    model.eval()
    img = Image.open(path).convert("RGB")
    t = eval_tf(img).unsqueeze(0).to(device)
    _, f = model(t, cam_label=None)
    f = nn.functional.normalize(f, p=2, dim=1)
    return f.cpu().numpy().flatten()

def measure(model, tag=""):
    """테스트셋 기존 생성물로 gen_sim + R1 측정"""
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    
    print(f"  [{tag}]")
    for tc in EVAL_CAMS:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]:
                gp = f"{MARKET_TEST_GEN}/{tc}/{pid}_gen_{tc}.png"
                if os.path.exists(gp): ids.append(pid)
            if len(ids) >= NUM_EVAL: break
        if not ids: continue
        
        c1_sims, gen_sims = [], []
        c1f, qf, gf = [], [], []
        for pid in ids:
            fc = feat(model, sorted(gby[pid]["c1"])[0])
            fq = feat(model, sorted(qby[pid][tc])[0])
            fg = feat(model, f"{MARKET_TEST_GEN}/{tc}/{pid}_gen_{tc}.png")
            c1_sims.append(fq @ fc); gen_sims.append(fq @ fg)
            c1f.append(fc); qf.append(fq); gf.append(fg)
        
        N = len(ids)
        c1_sim = np.mean(c1_sims); gen_sim = np.mean(gen_sims)
        # R1
        qf_a = np.array(qf); cf_a = np.array(c1f); gf_a = np.array(gf)
        sims = qf_a @ cf_a.T
        r1 = sum(1 for i in range(N) if sims[i].argmax() == i) / N * 100
        # R1 expand
        r1e = 0
        for i in range(N):
            s = np.maximum(qf_a[i] @ cf_a.T, qf_a[i] @ gf_a.T)
            if s.argmax() == i: r1e += 1
        r1e = r1e / N * 100
        
        print(f"    {tc}: c1_sim={c1_sim:.4f} gen_sim={gen_sim:.4f} "
              f"gap={gen_sim-c1_sim:+.4f} R1={r1:.0f}% R1+gen={r1e:.0f}%({r1e-r1:+.0f})")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    run_generation()
    run_finetune()
    
    print("\n" + "=" * 70)
    print("""
핵심 확인:
  파인튜닝 후 gen_sim 올라갔나?
  → 올라감: 모델이 SD 이미지 알아보기 시작 → 갤러리 확장 가능!
  → R1+gen > R1: 갤러리 확장이 실제로 매칭 향상!
  
  되면 다음:
  - GEN_CAMS를 ["c2","c3","c4","c5","c6"] 전체로
  - Duke 학습 → Market Cross-Domain 평가
  - 도메인 보정 결합
""")