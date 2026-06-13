"""
92_ablation_ABC.py

밤샘 비교 실험 — 생성 효과 격리

A: 기본 모델 (clipreid_market_nosie, 학습 X)        = baseline
B: 원본만 fine-tune (12,936장)                       = 통제군
C: 원본 + 생성 fine-tune (12,936 + 652)              = 실험군

핵심 비교: C vs B (유일한 차이 = 생성 데이터) → 생성의 순수 효과
B/C 동일 설정 (epoch, lr, triplet, sampler) → 공정 비교

평가: 5개 카메라 c2~c6, R1 / gen_sim / R1+gen
결과: 마지막에 A/B/C 비교 표
"""
import os, sys, glob, torch, numpy as np, datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import random, copy

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_TRAIN = f"{MARKET_DIR}/bounding_box_train"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
GEN_DIR = f"{PROJECT_DIR}/outputs/train_gen65"   # str 0.65 (c3)
TEST_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

device = "cuda"
GEN_CAMS = ["c3"]
EVAL_CAMS = ["c2", "c3", "c4", "c5", "c6"]
EPOCHS = 12
P, K = 8, 4
LR = 8e-6
MARGIN = 0.3
NUM_EVAL = 100

random.seed(42); np.random.seed(42); torch.manual_seed(42)
sys.path.insert(0, "/home/ubuntu/CLIP-ReID")

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]
def logline(m):
    with open(LOG,"a") as fp: fp.write(m+"\n")
    print(m)

# ===== Dataset =====
class PKSampler(Sampler):
    def __init__(self, p2i, P, K, nb):
        self.p2i=p2i; self.pids=list(p2i.keys()); self.P=P; self.K=K; self.nb=nb
    def __iter__(self):
        for _ in range(self.nb):
            b=[]; sel=random.sample(self.pids, min(self.P,len(self.pids)))
            for pid in sel:
                idxs=self.p2i[pid]
                b.extend(random.sample(idxs,self.K) if len(idxs)>=self.K else random.choices(idxs,k=self.K))
            yield b
    def __len__(self): return self.nb

class ReIDDataset(Dataset):
    def __init__(self, use_gen):
        self.samples=[]; self.pid_map={}; self.p2i=defaultdict(list)
        for f in sorted(glob.glob(f"{MARKET_TRAIN}/*.jpg")):
            pid,cam=parse(f)
            if pid in ('-1','0000'): continue
            if pid not in self.pid_map: self.pid_map[pid]=len(self.pid_map)
            self.p2i[self.pid_map[pid]].append(len(self.samples))
            self.samples.append((f,self.pid_map[pid]))
        real_n=len(self.samples); gen_n=0
        if use_gen:
            for tc in GEN_CAMS:
                for f in sorted(glob.glob(f"{GEN_DIR}/{tc}/*.png")):
                    pid=os.path.basename(f).split("_")[0]
                    if pid in self.pid_map:
                        self.p2i[self.pid_map[pid]].append(len(self.samples))
                        self.samples.append((f,self.pid_map[pid])); gen_n+=1
        self.num_classes=len(self.pid_map)
        print(f"    원본 {real_n} + 생성 {gen_n} = {len(self.samples)}장")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        f,pid=self.samples[idx]
        return self.tf(Image.open(f).convert("RGB")), pid
    tf=T.Compose([T.Resize([256,128]),T.RandomHorizontalFlip(0.5),
                  T.Pad(10),T.RandomCrop([256,128]),
                  T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

def triplet_loss(feats, pids, margin=0.3):
    fn=nn.functional.normalize(feats,dim=1); dist=1-fn@fn.t()
    pids=pids.view(-1,1); mp=(pids==pids.t()).float()
    dap=(dist*mp).max(1)[0]; dan=(dist+mp*1e5).min(1)[0]
    return torch.clamp(dap-dan+margin,min=0).mean()

class Wrap(nn.Module):
    def __init__(self, backbone, nc):
        super().__init__(); self.backbone=backbone; self.classifier=None; self.nc=nc
    def _init(self,d): self.classifier=nn.Linear(d,self.nc).to(device)
    def extract(self,x):
        was=self.backbone.training; self.backbone.eval()
        f=self.backbone(x,cam_label=None)
        if was: self.backbone.train()
        if isinstance(f,(list,tuple)): f=f[0] if isinstance(f[0],torch.Tensor) else f[-1]
        if f.dim()>2: f=f.view(f.size(0),-1)
        return f
    def forward(self,x):
        f=self.extract(x)
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
    try: b=make_model(cfg,num_class=751,camera_num=0,view_num=1)
    except: b=make_model(cfg,num_class=751,camera_num=6,view_num=1)
    b.load_param(wp); return b.to(device)

eval_tf=T.Compose([T.Resize([256,128]),T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
@torch.no_grad()
def feat(model, path):
    model.eval()
    t=eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    f=model.extract(t)
    return nn.functional.normalize(f,dim=1).cpu().numpy().flatten()
def get_gen(tc,pid):
    p=f"{TEST_GEN}/{tc}/{pid}_gen_{tc}.png"; return p if os.path.exists(p) else None

def evaluate(model):
    """5개 카메라 평균 R1, gen_sim, R1+gen 반환"""
    gby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid,cam=parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid,cam=parse(f); qby[pid][cam].append(f)
    rows=[]; sR=sG=sRe=sC=0; cnt=0
    for tc in EVAL_CAMS:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid] and get_gen(tc,pid): ids.append(pid)
            if len(ids)>=NUM_EVAL: break
        if not ids: continue
        cf,qf,gf=[],[],[]
        for pid in ids:
            cf.append(feat(model,sorted(gby[pid]["c1"])[0]))
            qf.append(feat(model,sorted(qby[pid][tc])[0]))
            gf.append(feat(model,get_gen(tc,pid)))
        cf,qf,gf=np.array(cf),np.array(qf),np.array(gf); N=len(ids)
        c1s=np.mean([qf[i]@cf[i] for i in range(N)])
        gs=np.mean([qf[i]@gf[i] for i in range(N)])
        sims=qf@cf.T
        r1=sum(1 for i in range(N) if sims[i].argmax()==i)/N*100
        r1e=sum(1 for i in range(N) if np.maximum(qf[i]@cf.T,qf[i]@gf.T).argmax()==i)/N*100
        rows.append((tc,r1,gs,c1s,r1e))
        sR+=r1; sG+=gs; sC+=c1s; sRe+=r1e; cnt+=1
    return rows, (sR/cnt, sG/cnt, sC/cnt, sRe/cnt)

def train_model(use_gen, tag):
    print(f"\n  [{tag}] 학습...")
    ds=ReIDDataset(use_gen)
    sampler=PKSampler(ds.p2i,P,K,nb=len(ds)//(P*K))
    loader=DataLoader(ds,batch_sampler=sampler,num_workers=0)
    model=Wrap(load_backbone(),ds.num_classes)
    _=model(torch.randn(2,3,256,128).to(device))
    opt=torch.optim.Adam([
        {'params':model.backbone.parameters(),'lr':LR},
        {'params':model.classifier.parameters(),'lr':LR*10},
    ],weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    ce=nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    for ep in range(EPOCHS):
        ti=tt=0;n=0;c=0;t=0
        for imgs,pids in loader:
            imgs,pids=imgs.to(device),pids.to(device)
            logits,feats=model(imgs)
            loss=ce(logits,pids)+triplet_loss(feats,pids,MARGIN)
            opt.zero_grad();loss.backward();opt.step()
            ti+=ce(logits,pids).item();tt+=triplet_loss(feats,pids,MARGIN).item();n+=1
            c+=(logits.argmax(1)==pids).sum().item();t+=pids.size(0)
        sched.step()
        if (ep+1)%4==0: print(f"    epoch {ep+1}/{EPOCHS}: acc={c/t*100:.1f}%")
    return model

def main():
    logline(f"\n## [{datetime.date.today()}] script92 A/B/C 비교 (생성효과 격리)")
    logline(f"A=기본, B=원본만 fine-tune, C=원본+생성 fine-tune | str0.65 c3 | EPOCHS={EPOCHS}")
    results={}

    # A: 기본
    print("\n"+"="*70); print("A: 기본 모델 (학습 X)"); print("="*70)
    model_a=Wrap(load_backbone(), 751)
    _=model_a(torch.randn(2,3,256,128).to(device))
    rows_a, avg_a=evaluate(model_a)
    results['A']=(rows_a,avg_a)
    del model_a; torch.cuda.empty_cache()

    # B: 원본만
    print("\n"+"="*70); print("B: 원본만 fine-tune"); print("="*70)
    model_b=train_model(use_gen=False, tag="B")
    rows_b, avg_b=evaluate(model_b)
    results['B']=(rows_b,avg_b)
    torch.save(model_b.state_dict(), f"{CKPT}/clipreid_B_realonly.pth")
    del model_b; torch.cuda.empty_cache()

    # C: 원본 + 생성
    print("\n"+"="*70); print("C: 원본 + 생성 fine-tune"); print("="*70)
    model_c=train_model(use_gen=True, tag="C")
    rows_c, avg_c=evaluate(model_c)
    results['C']=(rows_c,avg_c)
    torch.save(model_c.state_dict(), f"{CKPT}/clipreid_C_withgen.pth")
    del model_c; torch.cuda.empty_cache()

    # ===== 비교 표 =====
    logline("\n"+"="*70)
    logline("결과 비교 (5개 카메라 평균)")
    logline("="*70)
    logline(f"{'설정':<25}{'R1':<10}{'gen_sim':<12}{'c1_sim':<12}{'R1+gen':<12}")
    logline("-"*70)
    names={'A':'A 기본(학습X)','B':'B 원본만','C':'C 원본+생성'}
    for k in ['A','B','C']:
        r,g,c,re=results[k][1]
        logline(f"{names[k]:<25}{r:<10.1f}{g:<12.3f}{c:<12.3f}{re:<12.1f}")
    logline("-"*70)
    # 핵심 비교
    _,(rA,gA,cA,reA)=results['A']
    _,(rB,gB,cB,reB)=results['B']
    _,(rC,gC,cC,reC)=results['C']
    logline(f"\n핵심 비교:")
    logline(f"  C vs B (생성 순수 효과):")
    logline(f"    R1:      {rB:.1f} → {rC:.1f} ({rC-rB:+.1f})")
    logline(f"    gen_sim: {gB:.3f} → {gC:.3f} ({gC-gB:+.3f})")
    logline(f"    R1+gen:  {reB:.1f} → {reC:.1f} ({reC-reB:+.1f})")
    logline(f"  B vs A (fine-tune 자체 효과): R1 {rA:.1f} → {rB:.1f} ({rB-rA:+.1f})")
    logline(f"\n해석:")
    if gC > gB + 0.02:
        logline(f"  ✅ 생성 학습이 gen_sim 올림 → 모델이 생성물 더 인식")
    if reC > reB + 1:
        logline(f"  ✅ C의 갤러리확장(R1+gen)이 B보다 나음 → 생성 효과!")
    if abs(rB - rA) < 3:
        logline(f"  ℹ️ B≈A → fine-tune 자체는 큰 변화 없음 (통제군 정상)")
    logline("\n카메라별 상세는 result_92.txt 참고")

    # 카메라별 상세
    print("\n카메라별 상세:")
    for k in ['A','B','C']:
        print(f"\n[{names[k]}]")
        for tc,r1,gs,c1s,r1e in results[k][0]:
            print(f"  {tc}: R1={r1:.0f} gen_sim={gs:.3f} c1_sim={c1s:.3f} R1+gen={r1e:.0f}({r1e-r1:+.0f})")

if __name__ == "__main__":
    main()