"""
91_proper_finetune.py

제대로 된 fine-tune: 전체 ID + triplet + ID loss

90번 문제: 652 ID만 학습 → 과적합 → R1 폭락
이번 해결:
  - 원본 12,936장 (751 ID) 전체 + 생성 652장
  - triplet loss (ID 구분력 유지, 폭락 방지)
  - ID loss (cross-entropy)
  - PK sampling (triplet용: P명 × K장)
  - lr 작게, epoch 적게

목표: R1 유지하면서 gen_sim 올림 → "생성형 인식 기본 모델"
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import random

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_TRAIN = f"{MARKET_DIR}/bounding_box_train"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
TRAIN_GEN = f"{PROJECT_DIR}/outputs/train_gen"       # str 0.4 (c3)
TRAIN_GEN65 = f"{PROJECT_DIR}/outputs/train_gen65"   # str 0.65 (c3)
TEST_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"

device = "cuda"
GEN_DIR = TRAIN_GEN65   # 0.65 사용 (시점 변형 큰 것)
GEN_CAMS = ["c3"]
EVAL_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 12
P = 8          # ID 수 per batch
K = 4          # 장 수 per ID
LR = 8e-6
MARGIN = 0.3
NUM_EVAL = 100

random.seed(42); np.random.seed(42); torch.manual_seed(42)
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

# ===== PK Sampler (triplet용) =====
class PKSampler(Sampler):
    def __init__(self, pid_to_idxs, P, K, num_batches):
        self.pid_to_idxs = pid_to_idxs
        self.pids = list(pid_to_idxs.keys())
        self.P, self.K = P, K
        self.num_batches = num_batches
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            sel_pids = random.sample(self.pids, min(self.P, len(self.pids)))
            for pid in sel_pids:
                idxs = self.pid_to_idxs[pid]
                if len(idxs) >= self.K:
                    batch.extend(random.sample(idxs, self.K))
                else:
                    batch.extend(random.choices(idxs, k=self.K))
            yield batch
    def __len__(self): return self.num_batches

class ReIDDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.pid_map = {}
        self.pid_to_idxs = defaultdict(list)
        # 원본
        for f in sorted(glob.glob(f"{MARKET_TRAIN}/*.jpg")):
            pid, cam = parse(f)
            if pid in ('-1','0000'): continue
            if pid not in self.pid_map:
                self.pid_map[pid] = len(self.pid_map)
            self.pid_to_idxs[self.pid_map[pid]].append(len(self.samples))
            self.samples.append((f, self.pid_map[pid]))
        real_n = len(self.samples)
        # 생성
        gen_n = 0
        for tc in GEN_CAMS:
            for f in sorted(glob.glob(f"{GEN_DIR}/{tc}/*.png")):
                pid = os.path.basename(f).split("_")[0]
                if pid in self.pid_map:
                    self.pid_to_idxs[self.pid_map[pid]].append(len(self.samples))
                    self.samples.append((f, self.pid_map[pid]))
                    gen_n += 1
        self.num_classes = len(self.pid_map)
        print(f"  원본 {real_n} + 생성 {gen_n} = {len(self.samples)}장, ID {self.num_classes}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        f, pid = self.samples[idx]
        return self.tf(Image.open(f).convert("RGB")), pid
    tf = T.Compose([T.Resize([256,128]), T.RandomHorizontalFlip(0.5),
                    T.Pad(10), T.RandomCrop([256,128]),
                    T.ToTensor(), T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

# ===== triplet loss =====
def triplet_loss(feats, pids, margin=0.3):
    n = feats.size(0)
    fn = nn.functional.normalize(feats, dim=1)
    dist = 1 - fn @ fn.t()  # cosine distance
    pids = pids.view(-1, 1)
    mask_pos = (pids == pids.t()).float()
    mask_neg = 1 - mask_pos
    # hardest positive (가장 먼 같은 ID)
    dist_ap = (dist * mask_pos).max(dim=1)[0]
    # hardest negative (가장 가까운 다른 ID)
    dist_an = (dist + mask_pos * 1e5).min(dim=1)[0]
    loss = torch.clamp(dist_ap - dist_an + margin, min=0).mean()
    return loss

# ===== model =====
class Wrap(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = None
        self.num_classes = num_classes
    def _init(self, d): self.classifier = nn.Linear(d, self.num_classes).to(device)
    def extract(self, x):
        was = self.backbone.training; self.backbone.eval()
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
    wp=f"{CKPT}/clipreid_market_nosie.pth"; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: b = make_model(cfg, num_class=751, camera_num=0, view_num=1)
    except: b = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    b.load_param(wp); return b.to(device)

# ===== eval =====
eval_tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                     T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
@torch.no_grad()
def feat(model, path):
    model.eval()
    t = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    f = model.extract(t)
    return nn.functional.normalize(f, dim=1).cpu().numpy().flatten()

def get_gen(tc, pid):
    p = f"{TEST_GEN}/{tc}/{pid}_gen_{tc}.png"
    return p if os.path.exists(p) else None

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
    sR=sRe=0; cnt=0
    for tc in EVAL_CAMS:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid] and get_gen(tc,pid): ids.append(pid)
            if len(ids)>=NUM_EVAL: break
        if not ids: continue
        cf,qf,gf=[],[],[]
        for pid in ids:
            cf.append(feat(model, sorted(gby[pid]["c1"])[0]))
            qf.append(feat(model, sorted(qby[pid][tc])[0]))
            gf.append(feat(model, get_gen(tc,pid)))
        cf,qf,gf=np.array(cf),np.array(qf),np.array(gf); N=len(ids)
        c1s=np.mean([qf[i]@cf[i] for i in range(N)])
        gs=np.mean([qf[i]@gf[i] for i in range(N)])
        sims=qf@cf.T
        r1=sum(1 for i in range(N) if sims[i].argmax()==i)/N*100
        r1e=sum(1 for i in range(N) if np.maximum(qf[i]@cf.T,qf[i]@gf.T).argmax()==i)/N*100
        print(f"    {tc}: c1_sim={c1s:.3f} gen_sim={gs:.3f} gap={gs-c1s:+.3f} "
              f"R1={r1:.0f} R1+gen={r1e:.0f}({r1e-r1:+.0f})")
        sR+=r1; sRe+=r1e; cnt+=1
    if cnt: print(f"    평균 R1={sR/cnt:.1f} R1+gen={sRe/cnt:.1f} ({(sRe-sR)/cnt:+.1f})")

def main():
    print("="*70); print("제대로 fine-tune: 전체 ID + triplet + ID loss"); print("="*70)
    ds = ReIDDataset()
    sampler = PKSampler(ds.pid_to_idxs, P, K, num_batches=len(ds)//(P*K))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    model = Wrap(load_backbone(), ds.num_classes)

    print("\n--- 학습 전 ---"); measure(model, "BEFORE")

    _ = model(torch.randn(2,3,256,128).to(device))
    opt = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': LR*10},
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for ep in range(EPOCHS):
        t_id=t_tri=0; n=0; correct=0; total=0
        for imgs, pids in loader:
            imgs, pids = imgs.to(device), pids.to(device)
            logits, feats = model(imgs)
            l_id = ce(logits, pids)
            l_tri = triplet_loss(feats, pids, MARGIN)
            loss = l_id + l_tri
            opt.zero_grad(); loss.backward(); opt.step()
            t_id+=l_id.item(); t_tri+=l_tri.item(); n+=1
            correct+=(logits.argmax(1)==pids).sum().item(); total+=pids.size(0)
        sched.step()
        if (ep+1)%3==0:
            print(f"  epoch {ep+1}/{EPOCHS}: id={t_id/n:.3f} tri={t_tri/n:.3f} acc={correct/total*100:.1f}%")

    print("\n--- 학습 후 ---"); measure(model, "AFTER")
    torch.save(model.state_dict(), f"{CKPT}/clipreid_proper.pth")
    print(f"\n저장: {CKPT}/clipreid_proper.pth")
    print("""
핵심:
  R1 유지(폭락 X) + gen_sim ↑ → "생성형 인식 모델" 성공
  R1+gen > R1 → 갤러리 확장 효과
  되면 → 전체 카메라(c2~c6) 생성 + Cross-Domain 검증""")

if __name__ == "__main__":
    main()