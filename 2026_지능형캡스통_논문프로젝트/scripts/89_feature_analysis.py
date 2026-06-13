"""
89_feature_analysis.py

실제 vs 생성 feature가 어떻게 다른지 분석.
→ 차이의 성질에 따라 "지도 방법"이 결정됨

분석:
  1. Systematic shift: 실제 평균 vs 생성 평균 거리 (한쪽 쏠림?)
  2. 차원별 차이: 1280차원 중 어디가 다른가
  3. 거리 분포: 같은ID c1↔gen vs 다른ID c1↔gen (구분되나)
  4. t-SNE 시각화: 실제/생성이 섞이나 분리되나
  5. shift 보정 테스트: 생성평균-실제평균 차감하면 가까워지나

SIE 없는 CLIP-ReID (파인튜닝 안 한 원본). Market 테스트셋 생성물.
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
MARKET_GEN = f"{PROJECT_DIR}/outputs/c1base_gen_all"
OUT = f"{PROJECT_DIR}/outputs/feat_analysis"
os.makedirs(OUT, exist_ok=True)

device = "cuda"
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
N_PER_CAM = 100

def parse(f):
    p = os.path.basename(f).split("_"); return p[0], p[1][:2]

def load_nosie():
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES="market1501"; cfg.TEST.NECK_FEAT='before'
    wp = f"{CKPT}/clipreid_market_nosie.pth"
    cfg.TEST.WEIGHT = wp
    try: m = make_model(cfg, num_class=751, camera_num=0, view_num=1)
    except: m = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    m.load_param(wp); return m.eval().to(device)

tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

@torch.no_grad()
def feat(model, path):
    img = Image.open(path).convert("RGB")
    t = tf(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    if isinstance(f, (list, tuple)): f = f[0]
    return f.cpu().numpy().flatten()

def l2n(x): return x/(np.linalg.norm(x)+1e-9)

def main():
    model = load_nosie()
    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1','0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)

    # feature 수집
    print("feature 추출 중...")
    real_feats, gen_feats, q_feats, pids_list = [], [], [], []
    for tc in TARGET_CAMS:
        cnt = 0
        for pid in sorted(gby.keys()):
            if "c1" not in gby[pid] or tc not in qby[pid]: continue
            gp = f"{MARKET_GEN}/{tc}/{pid}_gen_{tc}.png"
            if not os.path.exists(gp): continue
            real_feats.append(feat(model, sorted(gby[pid]["c1"])[0]))
            gen_feats.append(feat(model, gp))
            q_feats.append(feat(model, sorted(qby[pid][tc])[0]))
            pids_list.append(f"{pid}_{tc}")
            cnt += 1
            if cnt >= N_PER_CAM: break

    real = np.array(real_feats)   # 실제 c1
    gen = np.array(gen_feats)     # 생성
    query = np.array(q_feats)     # query
    N = len(real)
    print(f"수집: {N}쌍")

    # ===== 1. Systematic shift =====
    print("\n" + "="*70)
    print("[1] Systematic shift — 실제 평균 vs 생성 평균")
    print("="*70)
    real_mean = real.mean(axis=0)
    gen_mean = gen.mean(axis=0)
    shift = gen_mean - real_mean
    print(f"  실제 평균 norm: {np.linalg.norm(real_mean):.3f}")
    print(f"  생성 평균 norm: {np.linalg.norm(gen_mean):.3f}")
    print(f"  shift norm (평균 차이): {np.linalg.norm(shift):.3f}")
    print(f"  평균 feature norm: {np.linalg.norm(real, axis=1).mean():.3f}")
    rel_shift = np.linalg.norm(shift) / np.linalg.norm(real, axis=1).mean()
    print(f"  상대 shift: {rel_shift:.1%}")
    if rel_shift > 0.1:
        print("  → 큰 systematic shift! 생성이 한 방향으로 쏠림 → 보정 가능성 ↑")
    else:
        print("  → shift 작음 → 단순 평균 차이 아님")

    # ===== 2. 차원별 차이 =====
    print("\n" + "="*70)
    print("[2] 차원별 차이 — 어느 차원이 다른가")
    print("="*70)
    dim_diff = np.abs(real.mean(axis=0) - gen.mean(axis=0))
    top_dims = np.argsort(-dim_diff)[:10]
    print(f"  차이 큰 상위 10개 차원: {top_dims.tolist()}")
    print(f"  상위10 차원 차이 합: {dim_diff[top_dims].sum():.3f}")
    print(f"  전체 차원 차이 합: {dim_diff.sum():.3f}")
    print(f"  상위10 비중: {dim_diff[top_dims].sum()/dim_diff.sum():.1%}")
    if dim_diff[top_dims].sum()/dim_diff.sum() > 0.3:
        print("  → 특정 차원에 차이 집중 → 그 차원만 처리하면 될 수도")
    else:
        print("  → 차이가 전 차원에 퍼짐 → 전역적 변환 필요")

    # ===== 3. 거리 분포 =====
    print("\n" + "="*70)
    print("[3] 거리 분포 — query 기준")
    print("="*70)
    rn = np.array([l2n(x) for x in real])
    gn = np.array([l2n(x) for x in gen])
    qn = np.array([l2n(x) for x in query])
    # 같은 ID
    same_c1 = np.mean([qn[i] @ rn[i] for i in range(N)])
    same_gen = np.mean([qn[i] @ gn[i] for i in range(N)])
    # 다른 ID (랜덤)
    diff_c1 = np.mean([qn[i] @ rn[(i+1) % N] for i in range(N)])
    diff_gen = np.mean([qn[i] @ gn[(i+1) % N] for i in range(N)])
    print(f"  query↔실제c1:  같은ID {same_c1:.3f}  다른ID {diff_c1:.3f}  분리 {same_c1-diff_c1:.3f}")
    print(f"  query↔생성:    같은ID {same_gen:.3f}  다른ID {diff_gen:.3f}  분리 {same_gen-diff_gen:.3f}")
    print(f"  → 생성도 같은ID > 다른ID 면 정보는 있음 (분리도 비교)")

    # ===== 4. shift 보정 테스트 =====
    print("\n" + "="*70)
    print("[4] shift 보정 — 생성에서 (gen_mean - real_mean) 빼면?")
    print("="*70)
    gen_corrected = gen - shift  # 생성 feature에서 shift 제거
    gcn = np.array([l2n(x) for x in gen_corrected])
    same_genc = np.mean([qn[i] @ gcn[i] for i in range(N)])
    print(f"  보정 전 query↔생성: {same_gen:.3f}")
    print(f"  보정 후 query↔생성: {same_genc:.3f}  ({same_genc-same_gen:+.3f})")
    print(f"  query↔실제c1:       {same_c1:.3f}")
    if same_genc > same_gen + 0.01:
        print("  → shift 보정으로 생성이 query에 가까워짐! → 보정 기반 접근 가능")
    else:
        print("  → shift 보정 효과 미미 → 단순 shift 문제 아님")

    # ===== 5. t-SNE 시각화 =====
    print("\n" + "="*70)
    print("[5] t-SNE 시각화")
    print("="*70)
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 샘플 (너무 많으면 느림)
        ns = min(N, 150)
        combined = np.vstack([rn[:ns], gn[:ns]])
        labels = ["real"]*ns + ["gen"]*ns
        emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(combined)
        plt.figure(figsize=(8,8))
        plt.scatter(emb[:ns,0], emb[:ns,1], c='blue', label='real c1', alpha=0.6, s=20)
        plt.scatter(emb[ns:,0], emb[ns:,1], c='red', label='generated', alpha=0.6, s=20)
        plt.legend(); plt.title("Real vs Generated features (t-SNE)")
        plt.savefig(f"{OUT}/tsne.png", dpi=120, bbox_inches='tight')
        print(f"  저장: {OUT}/tsne.png")
        print("  → 두 색이 섞임: feature 비슷 (좋음)")
        print("  → 두 색이 분리: 생성이 다른 영역 (지도 필요)")
    except Exception as e:
        print(f"  t-SNE 실패: {e}")

    print("\n" + "="*70)
    print("""종합 진단 → 지도 방법 결정:
  [1] shift 큼 + [4] 보정 효과 → 단순 평균 차감으로 해결 (학습 불필요)
  [2] 특정 차원 집중 → 그 차원 제거/정규화
  [3] 생성도 분리도 있음 → 정보는 있으니 alignment 학습 가능
  [5] t-SNE 분리 → feature alignment loss로 끌어당기기 필요""")

    del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()