"""
79_confidence_selective.py

B안: confidence 기반 선별적 생성 (정답 불필요, 배포 가능)

원리:
  보정(global) 적용 후, 각 query의 confidence 계산.
  confidence 낮은 query = 불확실 = hard 추정 → 그것만 생성 재정렬.
  confidence 높은 query = 확실 = 그대로 둠 (손상 방지).

confidence 척도 (이전 세션서 Top-5 분산이 91% hard예측):
  - Top-5 유사도의 분산 (낮으면 애매 = hard)
  - 또는 Top1 - Top2 margin (작으면 애매)
  여기선 둘 다 계산, margin 기반으로 하위 ratio 선별.

ratio: 하위 confidence 몇 %에 생성 적용할지 (10/20/30/40/50%)
→ 정답 없이 confidence만으로 고름

핵심: 전체 R1이 보정 단독(B)보다 오르는 ratio가 있나?
SIE 없는 weight, 양방향, 자동 로깅.
"""

import os, sys, glob, torch, numpy as np, csv, datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import random

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
DUKE_GALLERY = f"{DUKE_DIR}/bounding_box_test"
DUKE_QUERY = f"{DUKE_DIR}/query"
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"
DUKE_GEN = f"{PROJECT_DIR}/outputs/duke_c1base_gen"

device = "cuda"
NUM_IDS = 100
TOP_K = 10
ALPHA = 0.7
N_GLOBAL = 50
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
CSV = f"{PROJECT_DIR}/results.csv"

def logline(msg):
    with open(LOG,"a") as f: f.write(msg+"\n")
    print(msg)
def log_csv(row):
    exists=os.path.exists(CSV)
    with open(CSV,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

def parse(f):
    p=os.path.basename(f).split("_"); return p[0],p[1][:2]

def setup(gd,qd,cams):
    gby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{gd}/*.jpg")):
        pid,cam=parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby=defaultdict(lambda:defaultdict(list))
    for f in sorted(glob.glob(f"{qd}/*.jpg")):
        pid,cam=parse(f); qby[pid][cam].append(f)
    all_qf=[f for f in sorted(glob.glob(f"{qd}/*.jpg")) if parse(f)[0] not in ('-1','0000')]
    cvi={}
    for tc in cams:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids)>=NUM_IDS: break
        cvi[tc]=ids
    return gby,qby,cvi,all_qf

def load_nosie(wp,ds,nc,cn):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=ds; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: m=make_model(cfg,num_class=nc,camera_num=0,view_num=1)
    except Exception: m=make_model(cfg,num_class=nc,camera_num=cn,view_num=1)
    m.load_param(wp); return m.eval().to(device)

transform=T.Compose([T.Resize([256,128]),T.ToTensor(),
                     T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
@torch.no_grad()
def feat_raw(model,path):
    img=Image.open(path).convert("RGB"); t=transform(img).unsqueeze(0).to(device)
    try: f=model(t)
    except TypeError: f=model(t,cam_label=None)
    return f.cpu().numpy().flatten()
def l2n(x): return x/(np.linalg.norm(x)+1e-9)

def run(name,tag,wp,ds,nc,cn,eg,eq,cams,gen_dir):
    print("\n"+"="*96); print(f"방향: {name}"); print("="*96)
    if not os.path.exists(wp): print("⚠️ weight 없음"); return
    gby,qby,cvi,all_qf=setup(eg,eq,cams)
    model=load_nosie(wp,ds,nc,cn)
    print("feature 추출 중...")
    pair={}
    for tc in cams:
        c1r,genr,qr,kept=[],[],[],[]
        for pid in cvi[tc]:
            gp=f"{gen_dir}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            c1r.append(feat_raw(model,sorted(gby[pid]["c1"])[0]))
            genr.append(feat_raw(model,gp))
            qr.append(feat_raw(model,sorted(qby[pid][tc])[0]))
            kept.append(pid)
        eqs=set(sorted(qby[pid][tc])[0] for pid in kept if tc in qby[pid])
        pair[tc]=(np.array(c1r),np.array(genr),np.array(qr),eqs)
    all_eval=set()
    for tc in cams: all_eval|=pair[tc][3]
    pool=[f for f in all_qf if f not in all_eval]
    random.Random(42).shuffle(pool)
    gmean=np.mean([feat_raw(model,f) for f in pool[:N_GLOBAL]],axis=0)

    # 전 페어 통합 (confidence ratio는 전체 기준으로 잘라야 의미)
    # 각 query: s1 벡터(c1 갤러리 유사도), 정답 idx(자기 위치, 페어 내), 생성 유사도
    # confidence = margin = top1 - top2 (작을수록 불확실)
    Q=[]  # (s1_vec, gen_vec, true_local_idx, pair_tag, c1f, qf)
    per_pair={}
    for tc in cams:
        c1r,genr,qr,_=pair[tc]; N=len(qr)
        if N==0: continue
        qf=np.array([l2n(x-gmean) for x in qr])
        c1f=np.array([l2n(x-gmean) for x in c1r])
        genf=np.array([l2n(x-gmean) for x in genr])
        per_pair[tc]=(qf,c1f,genf,N)

    logline(f"\n## [{datetime.date.today()}] script79 confidence 선별생성 — {name}")
    # baseline 보정 단독(B) R1 + confidence margin 수집
    all_items=[]  # dict per query
    sB=0; totN=0
    for tc,(qf,c1f,genf,N) in per_pair.items():
        sims=qf@c1f.T
        for i in range(N):
            order=np.argsort(-sims[i])
            top1,top2=sims[i][order[0]],sims[i][order[1]]
            margin=top1-top2
            base_correct=(order[0]==i)
            # 생성 재정렬 결과 미리 계산
            topk=order[:TOP_K]
            s2=qf[i]@genf[topk].T
            final=ALPHA*sims[i][topk]+(1-ALPHA)*s2
            gen_correct=(topk[final.argmax()]==i)
            all_items.append({"margin":margin,"base":base_correct,"gen":gen_correct})
            sB+=base_correct
        totN+=N
    B_r1=sB/totN*100
    logline(f"보정 단독(B) R1 = {B_r1:.2f}  (총 {totN} query)")
    logline(f"{'ratio':<8}{'적용수':<8}{'R1':<10}{'vs B':<10}{'회복':<7}{'손상':<7}")
    logline("-"*96)

    margins=np.array([it["margin"] for it in all_items])
    best=(0,B_r1)
    for ratio in RATIOS:
        thr=np.quantile(margins, ratio)  # 하위 ratio = margin 작은 것
        correct=0; rec=0; dmg=0; applied=0
        for it in all_items:
            if it["margin"]<=thr:  # 불확실 → 생성 적용
                applied+=1
                use=it["gen"]
                if it["gen"] and not it["base"]: rec+=1
                if it["base"] and not it["gen"]: dmg+=1
            else:                   # 확실 → 그대로
                use=it["base"]
            correct+=use
        r1=correct/totN*100
        logline(f"{ratio:<8.0%}{applied:<8}{r1:<10.2f}{r1-B_r1:+.2f}     {rec:<7}{dmg:<7}")
        log_csv({"date":datetime.date.today(),"script":79,"dir":tag,"ratio":ratio,
                 "applied":applied,"R1":round(r1,2),"vs_B":round(r1-B_r1,2),
                 "recover":rec,"damage":dmg,"B_r1":round(B_r1,2)})
        if r1>best[1]: best=(ratio,r1)
    logline(f"\n  최적 ratio={best[0]:.0%}, R1={best[1]:.2f} (B={B_r1:.2f}, {best[1]-B_r1:+.2f})")
    del model; torch.cuda.empty_cache()

if __name__=="__main__":
    run("Duke학습 → Market평가","D2M",
        f"{CKPT}/clipreid_duke_nosie.pth","dukemtmcreid",702,8,
        MARKET_GALLERY,MARKET_QUERY,["c2","c3","c4","c5","c6"],MARKET_GEN)
    run("Market학습 → Duke평가","M2D",
        f"{CKPT}/clipreid_market_nosie.pth","market1501",751,6,
        DUKE_GALLERY,DUKE_QUERY,["c2","c3","c4","c5","c6","c8"],DUKE_GEN)
    logline("\n"+"="*96)
    logline("""해석:
  어떤 ratio에서 vs B > 0  → confidence 선별생성이 보정보다 향상 ✅ (B안 성공)
  모든 ratio에서 vs B ≤ 0  → 선별해도 손상이 회복 못이김 → 생성 접기
  margin 작은 쪽(불확실)에 적용하므로 정답 불필요 = 배포 가능""")