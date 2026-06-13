"""
87_finetune_genaware.py

핵심 질문: 생성 이미지로 파인튜닝하면 모델이 생성물을 알아보나?

[전]  gen_sim = 0.63 (OSNet, 생성물을 다른 사람으로 인식)
[후]  gen_sim = ??? (올라가면 → 갤러리 확장 효과 가능!)

방법:
  1. Market 원본 학습 데이터 + 생성 이미지를 합침 (같은 ID)
  2. OSNet fine-tune (20 에폭, 빠르게)
  3. gen_sim 재측정

OSNet + torchreid 사용 (학습 파이프라인 간단)
"""
import os, sys, glob, torch, numpy as np, shutil
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_TRAIN = f"{MARKET_DIR}/bounding_box_train"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"

device = "cuda"
NUM_IDS_EVAL = 100
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0003

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

# ===== 데이터셋: 원본 + 생성 =====
class CombinedDataset(Dataset):
    def __init__(self, real_dir, gen_dir, target_cams):
        self.samples = []  # (path, pid_int, is_gen)
        self.pid_to_int = {}
        
        # 원본 학습 데이터
        for f in sorted(glob.glob(f"{real_dir}/*.jpg")):
            pid, cam = parse(f)
            if pid in ('-1', '0000'): continue
            if pid not in self.pid_to_int:
                self.pid_to_int[pid] = len(self.pid_to_int)
            self.samples.append((f, self.pid_to_int[pid], False))
        
        # 생성 이미지 (같은 ID 라벨)
        gen_count = 0
        for tc in target_cams:
            gen_cam_dir = f"{gen_dir}/{tc}"
            if not os.path.isdir(gen_cam_dir): continue
            for f in sorted(glob.glob(f"{gen_cam_dir}/*.png")):
                bn = os.path.basename(f)
                pid = bn.split("_")[0]
                if pid in self.pid_to_int:
                    self.samples.append((f, self.pid_to_int[pid], True))
                    gen_count += 1
        
        self.num_classes = len(self.pid_to_int)
        real_count = sum(1 for s in self.samples if not s[2])
        print(f"  학습 데이터: 원본 {real_count}장 + 생성 {gen_count}장 = {len(self.samples)}장")
        print(f"  ID 수: {self.num_classes}")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, pid, is_gen = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, pid
    
    transform = T.Compose([
        T.Resize([256, 128]),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ===== OSNet 로드 =====
def load_osnet(num_classes, pretrained_weight=None):
    sys.path.insert(0, os.path.expanduser("~/osnet-reid"))
    import torchreid
    model = torchreid.models.build_model(
        name='osnet_x1_0', num_classes=num_classes, loss='softmax', pretrained=False)
    if pretrained_weight:
        # strict=False로 classifier 크기 다를 수 있음
        state = torch.load(pretrained_weight, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  OSNet pretrained 로드: {len(pretrained_dict)}/{len(model_dict)} layers")
    return model.to(device)

# ===== feature 추출 =====
eval_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def feat(model, path):
    model.eval()
    img = Image.open(path).convert("RGB")
    t = eval_tf(img).unsqueeze(0).to(device)
    f = model(t)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

# ===== gen_sim 측정 =====
def measure_gensim(model, tag=""):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    
    c1_sims, gen_sims = [], []
    n = 0
    for tc in TARGET_CAMS:
        for pid in sorted(gby.keys()):
            if "c1" not in gby[pid] or tc not in qby[pid]: continue
            gen_path = f"{MARKET_GEN}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gen_path): continue
            
            f_c1 = feat(model, sorted(gby[pid]["c1"])[0])
            f_gen = feat(model, gen_path)
            f_q = feat(model, sorted(qby[pid][tc])[0])
            
            c1_sims.append(f_q @ f_c1)
            gen_sims.append(f_q @ f_gen)
            n += 1
            if n >= NUM_IDS_EVAL * len(TARGET_CAMS): break
        if n >= NUM_IDS_EVAL * len(TARGET_CAMS): break
    
    c1_avg = np.mean(c1_sims)
    gen_avg = np.mean(gen_sims)
    print(f"  [{tag}] c1_sim={c1_avg:.4f}  gen_sim={gen_avg:.4f}  "
          f"차이={gen_avg-c1_avg:+.4f}  (n={n})")
    return c1_avg, gen_avg

# ===== 학습 =====
def train(model, dataset, epochs):
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=2, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    # feature extractor는 느리게, classifier는 빠르게
    params = [
        {'params': model.feature_extractor.parameters() 
         if hasattr(model, 'feature_extractor') else list(model.parameters())[:-2], 
         'lr': LR * 0.1},
        {'params': model.classifier.parameters() 
         if hasattr(model, 'classifier') else list(model.parameters())[-2:], 
         'lr': LR},
    ]
    try:
        optimizer = torch.optim.Adam(params)
    except:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        total_loss = 0; correct = 0; total = 0
        for imgs, pids in loader:
            imgs, pids = imgs.to(device), pids.to(device)
            logits = model(imgs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0] if logits[0].dim() == 2 else logits[1]
            loss = criterion(logits, pids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == pids).sum().item()
            total += pids.size(0)
        if (epoch + 1) % 5 == 0:
            acc = correct / total * 100
            print(f"  epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.3f} acc={acc:.1f}%")

# ===== 메인 =====
def main():
    print("=" * 70)
    print("파인튜닝 전/후 gen_sim 비교")
    print("=" * 70)
    
    # 데이터셋
    print("\n데이터 준비...")
    dataset = CombinedDataset(MARKET_TRAIN, MARKET_GEN, TARGET_CAMS)
    
    # OSNet 로드 (Market pretrained)
    print("\nOSNet 로드...")
    weight_candidates = [
        f"{CKPT}/osnet_x1_0_market.pth",
        f"{CKPT}/osnet_x1_0_market1501.pth",
        os.path.expanduser("~/osnet-reid/osnet_x1_0_market.pth"),
        os.path.expanduser("~/osnet-reid/checkpoints/osnet_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"),
    ]
    pretrained = None
    for wp in weight_candidates:
        if os.path.exists(wp):
            pretrained = wp; break
    
    model = load_osnet(dataset.num_classes, pretrained)
    
    # 파인튜닝 전 gen_sim
    print("\n--- 파인튜닝 전 ---")
    c1_before, gen_before = measure_gensim(model, "BEFORE")
    
    # 파인튜닝
    print(f"\n--- 파인튜닝 ({EPOCHS} epochs) ---")
    train(model, dataset, EPOCHS)
    
    # 파인튜닝 후 gen_sim
    print("\n--- 파인튜닝 후 ---")
    c1_after, gen_after = measure_gensim(model, "AFTER")
    
    # weight 저장
    save_path = f"{CKPT}/osnet_genaware.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nweight 저장: {save_path}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"  {'':20}{'c1_sim':<12}{'gen_sim':<12}{'gap':<12}")
    print(f"  {'파인튜닝 전':<20}{c1_before:<12.4f}{gen_before:<12.4f}{gen_before-c1_before:+.4f}")
    print(f"  {'파인튜닝 후':<20}{c1_after:<12.4f}{gen_after:<12.4f}{gen_after-c1_after:+.4f}")
    print(f"\n  gen_sim 변화: {gen_before:.4f} → {gen_after:.4f} ({gen_after-gen_before:+.4f})")
    
    if gen_after > gen_before + 0.01:
        print("\n  ✅ 파인튜닝 후 gen_sim 상승!")
        print("  → 모델이 생성 이미지를 더 잘 알아봄")
        print("  → 갤러리 확장 효과 기대 → 다음: 확장 재평가")
    elif gen_after > c1_after:
        print("\n  ✅✅ gen_sim > c1_sim!")
        print("  → 생성 이미지가 원본보다 query에 가까움")
        print("  → 갤러리 확장 확실히 효과!")
    else:
        print("\n  ⚠️ gen_sim 변화 미미")
        print("  → 에폭 더 늘리거나 학습률 조정 필요")

if __name__ == "__main__":
    main()