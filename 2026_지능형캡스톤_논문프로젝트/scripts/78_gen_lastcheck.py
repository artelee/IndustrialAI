"""
78_gen_lastcheck.py

생성의 마지막 검증:
  보정(global mean 차감)을 적용한 상태에서,
  baseline이 '틀린' query(hard)만 골라 생성으로 재정렬하면 회복되나?

  - easy(보정 후 이미 맞은 것)는 건드리지 않음 (선별적 적용)
  - hard에서만 생성 재정렬 → 회복 vs 손상 집계
  - 순효과 = 회복 - 손상

이것도 음수/0이면 → 생성 완전히 접기
양수면 → "선별적 생성"으로 살릴 여지

기준점: 보정(global)은 이미 적용 (B 상태에서 출발)
SIE 없는 weight, 양방향. 자동 로깅.
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
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
CSV = f"{PROJECT_DIR}/results.csv"

def logline(msg):
    with open(LOG, "a") as f: f.write(msg + "\n")
    print(msg)
def log_csv(row):
    exists = os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

def setup(gd, qd, cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gd}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{qd}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    all_qf = [f for f in sorted(glob.glob(f"{qd}/*.jpg")) if parse(f)[0] not in ('-1','0000')]
    cvi = {}
    for tc in cams:
        ids=[]
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids)>=NUM_IDS: break
        cvi[tc]=ids
    return gby, qby, cvi, all_qf

def load_nosie(wp, ds, nc, cn):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=ds; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: m=make_model(cfg, num_class=nc, camera_num=0, view_num=1)
    except Exception: m=make_model(cfg, num_class=nc, camera_num=cn, view_num=1)
    m.load_param(wp); return m.eval().to(device)

transform = T.Compose([T.Resize([256,128]), T.ToTensor(),
                       T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
@torch.no_grad()
def feat_raw(model, path):
    img=Image.open(path).convert("RGB"); t=transform(img).unsqueeze(0).to(device)
    try: f=model(t)
    except TypeError: f=model(t, cam_label=None)
    return f.cpu().numpy().flatten()
def l2n(x): return x/(np.linalg.norm(x)+1e-9)

def run(name, tag, wp, ds, nc, cn, eg, eq, cams, gen_dir):
    print("\n"+"="*96); print(f"방향: {name}"); print("="*96)
    if not os.path.exists(wp): print(f"⚠️ weight 없음"); return
    gby, qby, cvi, all_qf = setup(eg, eq, cams)
    model = load_nosie(wp, ds, nc, cn)
    print("feature 추출 중...")
    pair={}
    for tc in cams:
        c1r,genr,qr,kept=[],[],[],[]
        for pid in cvi[tc]:
            gp=f"{gen_dir}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            c1r.append(feat_raw(model, sorted(gby[pid]["c1"])[0]))
            genr.append(feat_raw(model, gp))
            qr.append(feat_raw(model, sorted(qby[pid][tc])[0]))
            kept.append(pid)
        eq_set=set(sorted(qby[pid][tc])[0] for pid in kept if tc in qby[pid])
        pair[tc]=(np.array(c1r),np.array(genr),np.array(qr),eq_set)
    all_eval=set()
    for tc in cams: all_eval|=pair[tc][3]
    pool=[f for f in all_qf if f not in all_eval]
    random.Random(42).shuffle(pool)
    gmean=np.mean([feat_raw(model,f) for f in pool[:N_GLOBAL]], axis=0)

    logline(f"\n## [{datetime.date.today()}] script78 보정後 hardcase 선별생성 — {name}")
    logline(f"{'Pair':<7}{'N':<5}{'hard':<6}{'회복':<7}{'손상':<7}{'순효과':<8}{'B(보정)':<10}{'B+선별생성':<10}")
    logline("-"*96)
    tot=defaultdict(int); sB=sBg=0.0; cnt=0
    for tc in cams:
        c1r,genr,qr,_=pair[tc]; N=len(qr)
        if N==0: continue
        # 보정 적용 (global 차감)
        qf=np.array([l2n(x-gmean) for x in qr])
        c1f=np.array([l2n(x-gmean) for x in c1r])
        genf=np.array([l2n(x-gmean) for x in genr])
        base=qf@c1f.T
        pred=base.argmax(axis=1)
        correct=(pred==np.arange(N))
        B_r1=correct.mean()*100
        hard=np.where(~correct)[0]; easy=np.where(correct)[0]
        # hard에서만 생성 재정렬
        recover=0
        for i in hard:
            s1=base[i]; topk=np.argsort(-s1)[:TOP_K]
            s2=qf[i]@genf[topk].T
            final=ALPHA*s1[topk]+(1-ALPHA)*s2
            if topk[final.argmax()]==i: recover+=1
        # easy는 그대로 유지(선별적), 손상은 "easy에 만약 적용했다면" 깨지는 수 측정
        damage=0
        for i in easy:
            s1=base[i]; topk=np.argsort(-s1)[:TOP_K]
            s2=qf[i]@genf[topk].T
            final=ALPHA*s1[topk]+(1-ALPHA)*s2
            if topk[final.argmax()]!=i: damage+=1
        net=recover  # 선별적용: easy 안 건드리므로 손상 0, 순효과=회복
        # 선별 적용 후 R1 = (기존 맞은 수 + 회복) / N
        Bg_r1=(correct.sum()+recover)/N*100
        logline(f"c1→{tc:<4}{N:<5}{len(hard):<6}{recover:<7}{damage:<7}"
                f"{recover:<8}{B_r1:<10.2f}{Bg_r1:<10.2f}")
        tot['hard']+=len(hard); tot['recover']+=recover; tot['damage']+=damage
        sB+=B_r1; sBg+=Bg_r1; cnt+=1
        log_csv({"date":datetime.date.today(),"script":78,"dir":tag,"pair":f"c1{tc}",
                 "N":N,"hard":len(hard),"recover":recover,"damage_if_applied":damage,
                 "B_corrected":round(B_r1,2),"B_selgen":round(Bg_r1,2)})
    if cnt:
        logline("-"*96)
        logline(f"합계 hard={tot['hard']} 회복={tot['recover']} "
                f"(easy 적용시 손상={tot['damage']})")
        logline(f"보정 평균 {sB/cnt:.2f} → 보정+선별생성 평균 {sBg/cnt:.2f} "
                f"({(sBg-sB)/cnt:+.2f})")
        logline(f"\n  → 회복({tot['recover']}) > 손상({tot['damage']}) 이면 선별생성 의미 있음")
        logline(f"  → 회복 ≤ 손상 이면 생성 완전히 접기")
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가","D2M",
        f"{CKPT}/clipreid_duke_nosie.pth","dukemtmcreid",702,8,
        MARKET_GALLERY,MARKET_QUERY,["c2","c3","c4","c5","c6"],MARKET_GEN)
    run("Market학습 → Duke평가","M2D",
        f"{CKPT}/clipreid_market_nosie.pth","market1501",751,6,
        DUKE_GALLERY,DUKE_QUERY,["c2","c3","c4","c5","c6","c8"],DUKE_GEN)
    logline("\n"+"="*96)
    logline("""최종 판단:
  회복 > 손상 (순 +)  → 선별적 생성은 살릴 수 있음 (hard case 전용)
  회복 ≤ 손상         → 생성 완전히 접고 'Training-Free 보정' 단독으로
  단, 회복이 hard 대비 너무 적으면(<10%) 실효성 의문""")