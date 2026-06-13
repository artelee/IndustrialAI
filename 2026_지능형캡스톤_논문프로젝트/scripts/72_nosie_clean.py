"""
72_nosie_clean.py

SIE 없이 학습된 weight 사용 (진짜 Training-Free + Cross-Domain 정직)
- clipreid_duke_nosie.pth   (Duke 학습, SIE 없음)
- clipreid_market_nosie.pth (Market 학습, SIE 없음)

차이 (71 대비):
- SIE_CAMERA = False  (모델 구조에 SIE 아예 없음)
- cam_label 인자 자체 불필요
- 가장 깨끗한 설정: 카메라 정보 어디에도 안 씀

두 방식 다 평가:
  방식 1) 단순 갤러리 확장 (c1 + 생성, max)
  방식 2) Re-ranking (Top-K 후 생성 재정렬)

양방향: Duke학습→Market, Market학습→Duke

먼저 weight 다운로드:
  cd ~/reid-gallery-expansion/checkpoints
  gdown 1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY -O clipreid_duke_nosie.pth
  gdown 1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7 -O clipreid_market_nosie.pth
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T

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

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]

def load_split(gallery_dir, query_dir, target_cams):
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'):
            continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        qby[pid][cam].append(f)
    cvi = {}
    for tc in target_cams:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]:
                ids.append(pid)
            if len(ids) >= NUM_IDS:
                break
        cvi[tc] = ids
    return gby, qby, cvi

def load_nosie(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [16, 16]   # ← nosie weight는 표준 stride 16 (pos_embed 129)
    cfg.MODEL.SIE_CAMERA = False       # ← SIE 없음 (핵심)
    cfg.MODEL.SIE_COE = 0.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path
    cfg.TEST.NECK_FEAT = 'before'
    # SIE 없는 weight: camera_num 0으로 시도, 실패 시 원래 값 (view_num=1 유지)
    try:
        m = make_model(cfg, num_class=num_class, camera_num=0, view_num=1)
    except Exception:
        m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.no_grad()
def feat(model, path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    # SIE 없는 모델: cam_label 없이 호출. forward 시그니처에 따라 분기.
    try:
        f = model(t)
    except TypeError:
        f = model(t, cam_label=None)
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()

def eval_both(model, gby, qby, valid_ids, target_cam, gen_dir):
    c1f, gf, qf = [], [], []
    kept = []
    for pid in valid_ids:
        c1_path = sorted(gby[pid]["c1"])[0]
        gen_path = f"{gen_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(qby[pid][target_cam])[0]
        if not os.path.exists(gen_path):
            continue
        c1f.append(feat(model, c1_path))
        gf.append(feat(model, gen_path))
        qf.append(feat(model, q_path))
        kept.append(pid)
    if not kept:
        return None
    c1f = np.array(c1f); gf = np.array(gf); qf = np.array(qf)
    N = len(kept)

    base_sims = qf @ c1f.T
    baseline = sum(1 for i in range(N) if base_sims[i].argmax() == i) / N

    # 방식 1: 단순 확장 (c1 + 생성, max)
    expand = 0
    for i in range(N):
        s = np.maximum(qf[i] @ c1f.T, qf[i] @ gf.T)
        if s.argmax() == i:
            expand += 1
    expand /= N

    # 방식 2: Re-ranking
    rerank = 0
    for i in range(N):
        s1 = base_sims[i]
        topk = np.argsort(-s1)[:TOP_K]
        s2 = qf[i] @ gf[topk].T
        final = ALPHA * s1[topk] + (1 - ALPHA) * s2
        if topk[final.argmax()] == i:
            rerank += 1
    rerank /= N

    # 생성 품질 진단: 생성-query sim vs c1-query sim
    gen_sim = np.mean([qf[i] @ gf[i] for i in range(N)])
    c1_sim = np.mean([qf[i] @ c1f[i] for i in range(N)])

    return {"N": N, "baseline": baseline, "expand": expand,
            "rerank": rerank, "gen_sim": gen_sim, "c1_sim": c1_sim}

def run(name, weight, dataset, num_class, camnum, eg, eq, cams, gen_dir):
    print("\n" + "=" * 85)
    print(f"방향: {name}  (SIE 없는 weight)")
    print("=" * 85)
    if not os.path.exists(weight):
        print(f"⚠️ weight 없음: {weight}")
        print("   gdown으로 먼저 다운로드 필요")
        return
    gby, qby, cvi = load_split(eg, eq, cams)
    model = load_nosie(weight, dataset, num_class, camnum)
    print("\n평가 중...")
    results = {}
    for tc in cams:
        r = eval_both(model, gby, qby, cvi[tc], tc, gen_dir)
        if r:
            results[tc] = r
    print("\n" + "-" * 85)
    print(f"[{name}]")
    print(f"{'Pair':<8}{'N':<6}{'Base':<10}{'Expand':<14}{'Rerank':<14}"
          f"{'c1_sim':<9}{'gen_sim':<9}")
    print("-" * 85)
    sb = se = sr = 0; cnt = 0
    for tc, r in results.items():
        b, e, rr = r["baseline"]*100, r["expand"]*100, r["rerank"]*100
        print(f"c1→{tc:<5}{r['N']:<6}{b:<10.2f}{e:<6.2f}({e-b:+.1f})   "
              f"{rr:<6.2f}({rr-b:+.1f})   {r['c1_sim']:<9.3f}{r['gen_sim']:<9.3f}")
        sb += b; se += e; sr += rr; cnt += 1
    if cnt:
        print("-" * 85)
        print(f"{'평균':<8}{'':<6}{sb/cnt:<10.2f}"
              f"{se/cnt:<6.2f}({(se-sb)/cnt:+.1f})   "
              f"{sr/cnt:<6.2f}({(sr-sb)/cnt:+.1f})")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run("Duke학습 → Market평가",
        f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8,
        MARKET_GALLERY, MARKET_QUERY, ["c2","c3","c4","c5","c6"], MARKET_GEN)

    run("Market학습 → Duke평가",
        f"{CKPT}/clipreid_market_nosie.pth", "market1501", 751, 6,
        DUKE_GALLERY, DUKE_QUERY, ["c2","c3","c4","c5","c6","c8"], DUKE_GEN)

    print("\n" + "=" * 85)
    print("""
핵심 확인:
- gen_sim < c1_sim 이면 → 생성이 원본보다 query와 멀다 (변별력 부족)
- Expand/Rerank 둘 다 하락 → SIE 없이는 생성 보강 효과 없음 확정
- 만약 하나라도 +  → 그 방식/페어에서 자세 보강 효과 (희망)
""")