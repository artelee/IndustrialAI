"""
90_alignment_train.py

핵심: 시점 변형(strength 0.65) + ID 보존을 동시에

t-SNE 발견: strength 0.4 생성은 c1과 너무 같음 (정보 중복)
→ strength 0.65로 시점 변형 (c1과 다르게)
→ alignment loss로 "시점 다른 생성도 같은 사람" 학습
→ 시점 다양성 + ID 보존 동시 달성

Phase 1: 학습 ID c3 → strength 0.65 생성
Phase 2: alignment + ID 분류 학습
   loss = α·(1-cos(real,gen)) + β·CE(분류)
   alignment = 생성을 실제로 정렬 (메인)
   분류 = collapse 방지 (보조)
Phase 3: gen_sim, 분리도, 갤러리확장 평가

c3만, ~50분
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
MARKET_TEST_GEN = f"{PROJECT_DIR}/outputs/qskel_s65"   # 테스트 ID str0.65 (83번 생성)
MARKET_TEST_GEN_FALLBACK = f"{PROJECT_DIR}/outputs/c1base_gen_all"  # 없으면 str0.4
TRAIN_GEN65 = f"{PROJECT_DIR}/outputs/train_gen65"

device = "cuda"
SIZE = (384, 768)
STRENGTH = 0.65
GEN_CAM = "c3"
EVAL_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 15           # 88번보다 적게 (오버피팅 방지)
BATCH = 16            # alignment는 쌍으로 봐야해서 작게
LR = 5e-6             # 작게
ALPHA = 1.0           # alignment 가중치 (메인)
BETA = 0.3            # 분류 가중치 (보조)
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


# ============ Phase 1: 생성 ============
def run_generation():
    print("="*70); print(f"Phase 1: 학습 ID 생성 (str={STRENGTH}, {GEN_CAM})"); print("="*70)
    tby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_TRAIN}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        tby[pid][cam].append(f)
    c1_ids = [pid for pid in sorted(tby.keys()) if "c1" in tby[pid]]

    need = False
    for pid in c1_ids[:5]:
        if not os.path.exists(f"{TRAIN_GEN65}/{GEN_CAM}/{pid}_gen_{GEN_CAM}.png"):
            need = True; break
    if not need:
        print("  이미 생성됨, 스킵"); return

    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                          cache_dir=CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
        cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device); pipe.enable_attention_slicing()
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)

    pool = []
    for pid in tby:
        if GEN_CAM in tby[pid]: pool.extend(tby[pid][GEN_CAM])
    random.shuffle(pool); pool = pool[:100]

    os.makedirs(f"{TRAIN_GEN65}/{GEN_CAM}", exist_ok=True)
    for pid in tqdm(c1_ids, desc=f"[생성 str{STRENGTH}] {GEN_CAM}"):
        save = f"{TRAIN_GEN65}/{GEN_CAM}/{pid}_gen_{GEN_CAM}.png"
        if os.path.exists(save): continue
        c1 = Image.open(sorted(tby[pid]["c1"])[0]).convert("RGB").resize(SIZE, Image.LANCZOS)
        pose = Image.open(random.choice(pool)).convert("RGB").resize(SIZE, Image.LANCZOS)
        skel = normalize_skeleton(openpose(pose).resize(SIZE, Image.LANCZOS))
        with torch.no_grad():
            r = pipe(prompt="a photo of one person, full body, surveillance camera",
                     negative_prompt="two people, blurry, deformed",
                     image=c1, control_image=skel, strength=STRENGTH,
                     controlnet_conditioning_scale=0.8, num_inference_steps=30,
                     guidance_scale=7.5,
                     generator=torch.Generator(device=device).manual_seed(42),
                     width=SIZE[0], height=SIZE[1]).images[0]
        r.save(save)
    del pipe, openpose; torch.cuda.empty_cache()
    print("  생성 완료")


# ============ Phase 2: alignment 학습 ============
class PairDataset(Dataset):
    """같은 ID의 (실제 c1, 생성) 쌍"""
    def __init__(self):
        self.pairs = []  # (real_path, gen_path, pid_int)
        self.pid_map = {}
        tby = defaultdict(lambda: defaultdict(list))
        for f in sorted(glob.glob(f"{MARKET_TRAIN}/*.jpg")):
            pid, cam = parse(f)
            if pid in ('-1','0000'): continue
            tby[pid][cam].append(f)
        for pid in sorted(tby.keys()):
            if "c1" not in tby[pid]: continue
            gp = f"{TRAIN_GEN65}/{GEN_CAM}/{pid}_gen_{GEN_CAM}.png"
            if not os.path.exists(gp): continue
            if pid not in self.pid_map:
                self.pid_map[pid] = len(self.pid_map)
            # c1 여러 장과 생성을 페어링
            for c1f in tby[pid]["c1"]:
                self.pairs.append((c1f, gp, self.pid_map[pid]))
        self.num_classes = len(self.pid_map)
        print(f"  쌍: {len(self.pairs)}, ID: {self.num_classes}")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        rp, gp, pid = self.pairs[idx]
        r = self.tf(Image.open(rp).convert("RGB"))
        g = self.tf(Image.open(gp).convert("RGB"))
        return r, g, pid
    tf = T.Compose([T.Resize([256,128]), T.RandomHorizontalFlip(0.5),
                    T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

class Wrap(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = None
        self.num_classes = num_classes
    def _init(self, d):
        self.classifier = nn.Linear(d, self.num_classes).to(device)
    def extract(self, x):
        was = self.backbone.training
        self.backbone.eval()
        f = self.backbone(x, cam_label=None)
        if was: self.backbone.train()
        if isinstance(f, (list, tuple)):
            f = f[0] if isinstance(f[0], torch.Tensor) else f[-1]
        if f.dim() > 2: f = f.view(f.size(0), -1)
        return f
    def forward(self, x):
        f = self.extract(x)
        if self.classifier is None: self._init(f.size(1))
        return self.classifier(f), f

def load_backbone():
    from config import cfg
    from model.make_model_clipreid import make_model
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="market1501"
    wp = f"{CKPT}/clipreid_market_nosie.pth"; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: b = make_model(cfg, num_class=751, camera_num=0, view_num=1)
    except: b = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    b.load_param(wp); return b.to(device)

def run_train():
    print("\n"+"="*70); print("Phase 2: alignment + ID 분류 학습"); print("="*70)
    ds = PairDataset()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    model = Wrap(load_backbone(), ds.num_classes)

    print("\n--- 학습 전 ---"); measure(model, "BEFORE")

    # optimizer
    _ = model(torch.randn(2,3,256,128).to(device))  # classifier 초기화
    opt = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': LR*10},
    ], weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for ep in range(EPOCHS):
        tot_a, tot_c, n = 0, 0, 0
        for r, g, pid in loader:
            r, g, pid = r.to(device), g.to(device), pid.to(device)
            logits_r, f_r = model(r)
            logits_g, f_g = model(g)
            # alignment: 같은 ID의 real-gen 정렬
            f_rn = nn.functional.normalize(f_r, dim=1)
            f_gn = nn.functional.normalize(f_g, dim=1)
            align = (1 - (f_rn * f_gn).sum(dim=1)).mean()
            # 분류 (collapse 방지)
            cls = ce(logits_r, pid) + ce(logits_g, pid)
            loss = ALPHA * align + BETA * cls
            opt.zero_grad(); loss.backward(); opt.step()
            tot_a += align.item(); tot_c += cls.item(); n += 1
        if (ep+1) % 3 == 0:
            print(f"  epoch {ep+1}/{EPOCHS}: align={tot_a/n:.4f} cls={tot_c/n:.3f}")

    print("\n--- 학습 후 ---"); measure(model, "AFTER")
    torch.save(model.state_dict(), f"{CKPT}/clipreid_aligned.pth")
    print(f"  저장: {CKPT}/clipreid_aligned.pth")


# ============ Phase 3: 평가 ============
eval_tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                     T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
@torch.no_grad()
def feat(model, path):
    model.eval()
    img = Image.open(path).convert("RGB")
    t = eval_tf(img).unsqueeze(0).to(device)
    f = model.extract(t)
    return nn.functional.normalize(f, dim=1).cpu().numpy().flatten()

def get_gen_path(tc, pid):
    p1 = f"{MARKET_TEST_GEN}/{tc}/{pid}_gen_{tc}.png"
    if os.path.exists(p1): return p1
    p2 = f"{MARKET_TEST_GEN_FALLBACK}/{tc}/{pid}_gen_{tc}.png"
    if os.path.exists(p2): return p2
    return None

def measure(model, tag=""):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    print(f"  [{tag}]")
    for tc in EVAL_CAMS:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid] and get_gen_path(tc, pid):
                ids.append(pid)
            if len(ids) >= NUM_EVAL: break
        if not ids: continue
        cf, qf, gf = [], [], []
        for pid in ids:
            cf.append(feat(model, sorted(gby[pid]["c1"])[0]))
            qf.append(feat(model, sorted(qby[pid][tc])[0]))
            gf.append(feat(model, get_gen_path(tc, pid)))
        cf, qf, gf = np.array(cf), np.array(qf), np.array(gf)
        N = len(ids)
        c1_sim = np.mean([qf[i]@cf[i] for i in range(N)])
        gen_sim = np.mean([qf[i]@gf[i] for i in range(N)])
        # 분리도
        same_g = gen_sim
        diff_g = np.mean([qf[i]@gf[(i+1)%N] for i in range(N)])
        sims = qf @ cf.T
        r1 = sum(1 for i in range(N) if sims[i].argmax()==i)/N*100
        r1e = 0
        for i in range(N):
            s = np.maximum(qf[i]@cf.T, qf[i]@gf.T)
            if s.argmax()==i: r1e += 1
        r1e = r1e/N*100
        print(f"    {tc}: c1_sim={c1_sim:.3f} gen_sim={gen_sim:.3f} gap={gen_sim-c1_sim:+.3f} "
              f"분리={same_g-diff_g:.3f} R1={r1:.0f} R1+gen={r1e:.0f}({r1e-r1:+.0f})")


if __name__ == "__main__":
    run_generation()
    run_train()
    print("\n"+"="*70)
    print("""핵심:
  gen_sim 올라가고 gap 0 근접 → alignment 성공
  R1+gen > R1 → 시점 다양성이 매칭 향상 (드디어!)
  분리도 유지 → ID 구분력 살아있음 (collapse 안 됨)""")