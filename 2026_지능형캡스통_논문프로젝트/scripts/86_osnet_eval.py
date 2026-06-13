"""
86_osnet_eval.py

핵심 질문: 시점 변환 생성물을 CNN(OSNet)으로 평가하면 효과 있나?

이전 결과:
  CLIP-ReID(ViT): gen_sim < c1_sim → 효과 없음
  근데 약한 모델에선 생성이 도움됐음 (CLIP 일반: +4%p)

OSNet = CNN, 2.2M 파라미터, ViT보다 약함
→ 생성 이미지가 도움될 수 있음

기존 생성물(strength 0.4, c1base_gen_all) + 보정(global) + 갤러리확장/rerank
새로 생성 필요 없음! feature만 다시 뽑으면 됨.
"""
import os, sys, glob, torch, numpy as np, csv, datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import random

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CKPT = f"{PROJECT_DIR}/checkpoints"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"  # 기존 생성물 (str 0.4)
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"

device = "cuda"
NUM_IDS = 100
TOP_K = 5; ALPHA = 0.7; N_GLOBAL = 50
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]

def logline(msg):
    with open(LOG, "a") as f: f.write(msg + "\n")
    print(msg)
def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

# ===== OSNet 로드 =====
def load_osnet():
    sys.path.insert(0, os.path.expanduser("~/osnet-reid"))
    import torchreid
    model = torchreid.models.build_model(
        name='osnet_x1_0', num_classes=751, loss='softmax', pretrained=False)
    # weight 찾기
    candidates = [
        f"{CKPT}/osnet_x1_0_market.pth",
        f"{CKPT}/osnet_x1_0_market1501.pth",
        os.path.expanduser("~/osnet-reid/osnet_x1_0_market.pth"),
        os.path.expanduser("~/osnet-reid/checkpoints/osnet_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"),
    ]
    loaded = False
    for wp in candidates:
        if os.path.exists(wp):
            torchreid.utils.load_pretrained_weights(model, wp)
            print(f"OSNet weight: {wp}")
            loaded = True; break
    if not loaded:
        model = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=751, loss='softmax', pretrained=True)
        print("OSNet weight: torchreid auto-download")
    return model.eval().to(device)

osnet_tf = T.Compose([
    T.Resize([256, 128]), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def feat_raw(model, path):
    img = Image.open(path).convert("RGB")
    t = osnet_tf(img).unsqueeze(0).to(device)
    f = model(t)
    return f.cpu().numpy().flatten()

def l2n(x): return x / (np.linalg.norm(x) + 1e-9)

# ===== 메인 =====
def main():
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    all_qf = [f for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg"))
              if parse(f)[0] not in ('-1', '0000')]
    cvi = {}
    for tc in TARGET_CAMS:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids) >= NUM_IDS: break
        cvi[tc] = ids

    model = load_osnet()

    # feature 추출
    print("\nfeature 추출 중...")
    pair = {}
    for tc in TARGET_CAMS:
        c1r, genr, qr, kept = [], [], [], []
        for pid in cvi[tc]:
            gp = f"{MARKET_GEN}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            c1r.append(feat_raw(model, sorted(gby[pid]["c1"])[0]))
            genr.append(feat_raw(model, gp))
            qr.append(feat_raw(model, sorted(qby[pid][tc])[0]))
            kept.append(pid)
        pair[tc] = (np.array(c1r), np.array(genr), np.array(qr), kept)

    # global mean (OSNet feature 공간에서)
    all_eval = set()
    for tc in TARGET_CAMS:
        for pid in pair[tc][3]:
            if tc in qby[pid]:
                all_eval.add(sorted(qby[pid][tc])[0])
    pool = [f for f in all_qf if f not in all_eval]
    random.Random(42).shuffle(pool)
    gmean = np.mean([feat_raw(model, f) for f in pool[:N_GLOBAL]], axis=0)

    logline(f"\n## [{datetime.date.today()}] script86 OSNet(CNN) 평가 — Duke생성→Market")
    logline("기존 생성물(str 0.4) + OSNet feature + global 보정")

    # ===== 비교: 보정 없음 / 보정 있음 × baseline / expand / rerank =====
    for use_correct, tag in [(False, "보정X"), (True, "보정O")]:
        logline(f"\n[{tag}]")
        logline(f"{'Pair':<7}{'N':<5}{'Base':<8}{'Expand':<12}{'Rerank':<12}{'c1_sim':<9}{'gen_sim':<9}")
        logline("-" * 80)
        sB = sE = sR = 0; cnt = 0
        for tc in TARGET_CAMS:
            c1r, genr, qr, kept = pair[tc]
            N = len(kept)
            if N == 0: continue
            if use_correct:
                qf = np.array([l2n(x - gmean) for x in qr])
                c1f = np.array([l2n(x - gmean) for x in c1r])
                gf = np.array([l2n(x - gmean) for x in genr])
            else:
                qf = np.array([l2n(x) for x in qr])
                c1f = np.array([l2n(x) for x in c1r])
                gf = np.array([l2n(x) for x in genr])
            # sim
            c1_sim = np.mean([qf[i] @ c1f[i] for i in range(N)])
            gen_sim = np.mean([qf[i] @ gf[i] for i in range(N)])
            # Baseline
            sims = qf @ c1f.T
            base = sum(1 for i in range(N) if sims[i].argmax() == i) / N * 100
            # Expand
            exp_c = 0
            for i in range(N):
                s = np.maximum(qf[i] @ c1f.T, qf[i] @ gf.T)
                if s.argmax() == i: exp_c += 1
            expand = exp_c / N * 100
            # Rerank
            rer_c = 0
            for i in range(N):
                s1 = sims[i]; topk = np.argsort(-s1)[:TOP_K]
                s2 = qf[i] @ gf[topk].T
                final = ALPHA * s1[topk] + (1 - ALPHA) * s2
                if topk[final.argmax()] == i: rer_c += 1
            rerank = rer_c / N * 100
            logline(f"c1→{tc:<4}{N:<5}{base:<8.1f}"
                    f"{expand:<5.1f}({expand-base:+.1f})  "
                    f"{rerank:<5.1f}({rerank-base:+.1f})  "
                    f"{c1_sim:<9.3f}{gen_sim:<9.3f}")
            sB += base; sE += expand; sR += rerank; cnt += 1
        if cnt:
            logline("-" * 80)
            logline(f"{'평균':<7}{'':<5}{sB/cnt:<8.1f}"
                    f"{sE/cnt:<5.1f}({(sE-sB)/cnt:+.1f})  "
                    f"{sR/cnt:<5.1f}({(sR-sB)/cnt:+.1f})")

    del model; torch.cuda.empty_cache()
    logline("\n" + "=" * 80)
    logline("""해석:
  gen_sim > c1_sim → CNN에서 생성이 query와 더 가까움 → 효과!
  Expand/Rerank > Base → CNN에서는 갤러리 확장이 작동
  보정O > 보정X → 도메인 보정이 CNN에서도 유효

  ViT에서 안 됐는데 CNN에서 되면:
  → "약한 모델에서 생성형 갤러리 확장이 효과적"
  → 논문 기여: 모델 강도별 생성 효과 분석 + CNN 대상 방법 제안""")

if __name__ == "__main__":
    main()