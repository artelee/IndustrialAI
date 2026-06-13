"""
87b_finetune_quick.py

빠른 확인: 생성 이미지로 파인튜닝하면 gen_sim 올라가나?

문제: 학습 ID ≠ 생성 ID (Market-1501 train/test 분리)
해법: 갤러리(test) 원본 + 생성 이미지로 학습 (테스트 ID 사용)
→ 논문용은 아니지만 "원리 확인"에는 충분
→ gen_sim 올라가면 → 학습 ID로 생성해서 본실험

데이터:
  갤러리 c1 원본 (test ID, ~668장)
  + 생성 이미지 (c2~c6, 같은 ID, ~3000장)
  = 약 3700장으로 파인튜닝
"""
import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"

device = "cuda"
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0003
NUM_EVAL = 100

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

class GenAwareDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.pid_map = {}
        
        # 갤러리 c1 원본 (이 ID들의 생성물이 있으니까)
        for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
            pid, cam = parse(f)
            if pid in ('-1', '0000'): continue
            if cam != "c1": continue
            if pid not in self.pid_map:
                self.pid_map[pid] = len(self.pid_map)
            self.samples.append((f, self.pid_map[pid]))
        real_count = len(self.samples)
        
        # 생성 이미지 (같은 ID)
        gen_count = 0
        for tc in TARGET_CAMS:
            for f in sorted(glob.glob(f"{MARKET_GEN}/{tc}/*.png")):
                pid = os.path.basename(f).split("_")[0]
                if pid in self.pid_map:
                    self.samples.append((f, self.pid_map[pid]))
                    gen_count += 1
        
        self.num_classes = len(self.pid_map)
        print(f"  원본 c1: {real_count}장, 생성: {gen_count}장, 합계: {len(self.samples)}장")
        print(f"  ID 수: {self.num_classes}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, pid
    
    transform = T.Compose([
        T.Resize([256, 128]),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def load_osnet(num_classes):
    sys.path.insert(0, os.path.expanduser("~/osnet-reid"))
    import torchreid
    model = torchreid.models.build_model(
        name='osnet_x1_0', num_classes=num_classes, loss='softmax', pretrained=True)
    print(f"  OSNet pretrained 로드 (num_classes={num_classes})")
    return model.to(device)

eval_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def feat(model, path):
    model.eval()
    img = Image.open(path).convert("RGB")
    t = eval_tf(img).unsqueeze(0).to(device)
    f = model(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def measure(model, tag=""):
    """gen_sim + baseline R1 측정"""
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    
    c1_sims, gen_sims = [], []
    c1_feats, q_feats, gen_feats, kept = [], [], [], []
    
    tc = "c5"  # c5만 빠르게
    ids = []
    for pid in sorted(gby.keys()):
        if "c1" in gby[pid] and tc in qby[pid]:
            gp = f"{MARKET_GEN}/{tc}/{pid}_gen_{tc}.png"
            if os.path.exists(gp): ids.append(pid)
        if len(ids) >= NUM_EVAL: break
    
    for pid in ids:
        fc1 = feat(model, sorted(gby[pid]["c1"])[0])
        fq = feat(model, sorted(qby[pid][tc])[0])
        fg = feat(model, f"{MARKET_GEN}/{tc}/{pid}_gen_{tc}.png")
        c1_sims.append(fq @ fc1)
        gen_sims.append(fq @ fg)
        c1_feats.append(fc1); q_feats.append(fq); gen_feats.append(fg)
    
    c1_avg = np.mean(c1_sims)
    gen_avg = np.mean(gen_sims)
    
    # R1 (c1만으로)
    N = len(ids)
    qf = np.array(q_feats); cf = np.array(c1_feats); gf = np.array(gen_feats)
    sims = qf @ cf.T
    r1_base = sum(1 for i in range(N) if sims[i].argmax() == i) / N * 100
    # R1 (c1 + 생성)
    r1_exp = 0
    for i in range(N):
        s = np.maximum(qf[i] @ cf.T, qf[i] @ gf.T)
        if s.argmax() == i: r1_exp += 1
    r1_exp = r1_exp / N * 100
    
    print(f"  [{tag}] c1_sim={c1_avg:.4f}  gen_sim={gen_avg:.4f}  "
          f"gap={gen_avg-c1_avg:+.4f}  R1_base={r1_base:.1f}%  R1_expand={r1_exp:.1f}%")
    return gen_avg

def train(model, dataset, epochs):
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=2, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        total_loss = 0; correct = 0; total = 0
        for imgs, pids in loader:
            imgs, pids = imgs.to(device), pids.to(device)
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                logits = [x for x in out if x.dim() == 2 and x.size(1) > 1][0]
            else:
                logits = out
            loss = criterion(logits, pids)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == pids).sum().item()
            total += pids.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}: loss={total_loss/len(loader):.3f} acc={correct/total*100:.1f}%")

def main():
    print("=" * 70)
    print("빠른 테스트: 생성 데이터 파인튜닝 → gen_sim 올라가나?")
    print("=" * 70)
    
    dataset = GenAwareDataset()
    model = load_osnet(dataset.num_classes)
    
    print("\n--- 파인튜닝 전 ---")
    before = measure(model, "BEFORE")
    
    print(f"\n--- 파인튜닝 ({EPOCHS} epochs) ---")
    train(model, dataset, EPOCHS)
    
    print("\n--- 파인튜닝 후 ---")
    after = measure(model, "AFTER")
    
    print("\n" + "=" * 70)
    print(f"gen_sim: {before:.4f} → {after:.4f} ({after-before:+.4f})")
    if after > before + 0.01:
        print("✅ gen_sim 상승! 모델이 생성물 알아보기 시작")
    else:
        print("⚠️ 변화 미미")

if __name__ == "__main__":
    main()