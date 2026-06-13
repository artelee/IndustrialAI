"""
77_ablation_4way.py

핵심 ablation: 보정(global mean 차감)과 생성(pose 갤러리 확장)의 기여 분리

  A. none          : 보정X 생성X  = baseline
  B. global        : 보정O 생성X
  C. gen           : 보정X 생성O (Re-ranking)
  D. global+gen    : 보정O 생성O  = 최종 제안

핵심 비교:
  B-A = 보정 효과
  C-A = 생성 효과
  D-B = 보정 위에 생성이 더하는 값  ← 생성이 진짜 필요한지
  D-C = 생성 위에 보정이 더하는 값

global mean = 타겟 도메인 사진 N장 평균 (카메라 안 가림), leakage 제외
SIE 없는 weight. 양방향.

자동 로깅: EXPERIMENT_LOG.md + results.csv
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
N_GLOBAL = 50          # 도메인 평균 추정용 샘플 수 (검증②에서 50이 안정적)

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
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

def setup(gallery_dir, query_dir, target_cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        qby[pid][cam].append(f)
    all_query_files = [f for f in sorted(glob.glob(f"{query_dir}/*.jpg"))
                       if parse(f)[0] not in ('-1','0000')]
    cvi = {}
    for tc in target_cams:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]:
                ids.append(pid)
            if len(ids) >= NUM_IDS: break
        cvi[tc] = ids
    return gby, qby, cvi, all_query_files

def load_nosie(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0
    cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=dataset_name
    cfg.TEST.WEIGHT=weight_path; cfg.TEST.NECK_FEAT='before'
    try:
        m=make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        m=make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([T.Resize([256,128]), T.ToTensor(),
                       T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

@torch.no_grad()
def feat_raw(model, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    return f.cpu().numpy().flatten()

def l2n(x): return x/(np.linalg.norm(x)+1e-9)

def r1_base(qf, gf, N):
    sims = qf @ gf.T
    return sum(1 for i in range(N) if sims[i].argmax()==i)/N

def r1_rerank(qf, c1f, genf, N):
    base = qf @ c1f.T; c=0
    for i in range(N):
        s1=base[i]; topk=np.argsort(-s1)[:TOP_K]
        s2=qf[i]@genf[topk].T
        final=ALPHA*s1[topk]+(1-ALPHA)*s2
        if topk[final.argmax()]==i: c+=1
    return c/N

def run(name, tag, weight, dataset, num_class, camnum, eg, eq, cams, gen_dir):
    print("\n"+"="*96)
    print(f"방향: {name}")
    print("="*96)
    if not os.path.exists(weight):
        print(f"⚠️ weight 없음: {weight}"); return
    gby, qby, cvi, all_qf = setup(eg, eq, cams)
    model = load_nosie(weight, dataset, num_class, camnum)

    # raw feature 추출
    print("feature 추출 중...")
    pair = {}
    for tc in cams:
        c1r, genr, qr, kept = [], [], [], []
        for pid in cvi[tc]:
            gp=f"{gen_dir}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            c1r.append(feat_raw(model, sorted(gby[pid]["c1"])[0]))
            genr.append(feat_raw(model, gp))
            qr.append(feat_raw(model, sorted(qby[pid][tc])[0]))
            kept.append(pid)
        eval_q = set(sorted(qby[pid][tc])[0] for pid in kept if tc in qby[pid])
        pair[tc] = (np.array(c1r), np.array(genr), np.array(qr), eval_q)

    # global mean (타겟 도메인 query에서 N_GLOBAL장, 평가분 제외)
    all_eval_q = set()
    for tc in cams:
        all_eval_q |= pair[tc][3]
    pool = [f for f in all_qf if f not in all_eval_q]
    random.Random(42).shuffle(pool)
    gmean = np.mean([feat_raw(model, f) for f in pool[:N_GLOBAL]], axis=0)

    logline(f"\n## [{datetime.date.today()}] script77 4-way ablation — {name}")
    logline(f"{'Pair':<7}{'A:none':<10}{'B:global':<12}{'C:gen':<12}{'D:glob+gen':<12}")
    logline("-"*96)

    sums = defaultdict(float); cnt=0
    for tc in cams:
        c1r, genr, qr, _ = pair[tc]
        N=len(qr)
        if N==0: continue
        # A. none
        qf=np.array([l2n(x) for x in qr]); c1f=np.array([l2n(x) for x in c1r])
        genf=np.array([l2n(x) for x in genr])
        A=r1_base(qf,c1f,N)*100
        C=r1_rerank(qf,c1f,genf,N)*100
        # B/D. global 차감
        qfg=np.array([l2n(x-gmean) for x in qr])
        c1fg=np.array([l2n(x-gmean) for x in c1r])
        genfg=np.array([l2n(x-gmean) for x in genr])
        B=r1_base(qfg,c1fg,N)*100
        D=r1_rerank(qfg,c1fg,genfg,N)*100
        logline(f"c1→{tc:<4}{A:<10.2f}{B:<6.2f}({B-A:+.1f})  {C:<6.2f}({C-A:+.1f})  {D:<6.2f}({D-A:+.1f})")
        log_csv({"date":datetime.date.today(),"script":77,"dir":tag,"pair":f"c1{tc}",
                 "A_none":round(A,2),"B_global":round(B,2),"C_gen":round(C,2),
                 "D_glob_gen":round(D,2),"N":N})
        sums['A']+=A; sums['B']+=B; sums['C']+=C; sums['D']+=D; cnt+=1
    if cnt:
        A,B,C,D = sums['A']/cnt, sums['B']/cnt, sums['C']/cnt, sums['D']/cnt
        logline("-"*96)
        logline(f"{'평균':<7}{A:<10.2f}{B:<6.2f}({B-A:+.1f})  {C:<6.2f}({C-A:+.1f})  {D:<6.2f}({D-A:+.1f})")
        logline(f"\n  보정효과 B-A={B-A:+.1f} | 생성효과 C-A={C-A:+.1f}")
        logline(f"  ★ 보정 위 생성기여 D-B={D-B:+.1f} | 생성 위 보정기여 D-C={D-C:+.1f}")
        log_csv({"date":datetime.date.today(),"script":77,"dir":tag,"pair":"AVG",
                 "A_none":round(A,2),"B_global":round(B,2),"C_gen":round(C,2),
                 "D_glob_gen":round(D,2),"N":cnt})
    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가", "D2M",
        f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8,
        MARKET_GALLERY, MARKET_QUERY, ["c2","c3","c4","c5","c6"], MARKET_GEN)
    run("Market학습 → Duke평가", "M2D",
        f"{CKPT}/clipreid_market_nosie.pth", "market1501", 751, 6,
        DUKE_GALLERY, DUKE_QUERY, ["c2","c3","c4","c5","c6","c8"], DUKE_GEN)
    logline("\n"+"="*96)
    logline("""해석:
  D-B > 0 확실  → 보정 위에 생성이 기여 → 둘 다 필요한 완성된 방법 ✅
  D-B ≈ 0       → 생성 불필요, 보정만으로 충분 → 논문서 생성 비중 축소
  D 최고        → 최종 제안(보정+생성)이 베스트
결과는 EXPERIMENT_LOG.md, results.csv 에 자동 저장됨""")