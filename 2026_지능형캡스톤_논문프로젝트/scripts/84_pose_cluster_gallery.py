"""
84_pose_cluster_gallery.py

선제적 갤러리 구축 — 자세 클러스터 기반 (네 원래 아이디어 개선)

핵심:
  이전: 타겟 카메라 랜덤 자세 1개 → query와 불일치
  이번: 타겟 카메라 대표 자세 K개 → 최소 1개는 query와 유사

Phase 1: 타겟 카메라별 자세 클러스터링
  c? 이미지들 → OpenPose skeleton → 이미지 기반 K-means → 대표 skeleton K개

Phase 2: 사전 생성 (오프라인, 선제적)
  c1 인물 × K개 대표 skeleton → Img2Img 생성 → 갤러리 추가

Phase 3: 평가
  보정(global mean) + 확장 갤러리(c1 + K장) 매칭
  비교: Baseline vs 갤러리 확장 vs Re-ranking

SIE 없는 weight. Duke→Market.
비교 이미지 저장: [c1 | skeleton_k | 생성물_k] × K

설정:
  K_CLUSTERS = 4 (정면/후면/좌/우 정도)
  STRENGTH = 0.6 (83번 결과 보고 조정)
"""
import os, sys, glob, torch, numpy as np, csv, datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.cluster import KMeans
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
LOG = f"{PROJECT_DIR}/EXPERIMENT_LOG.md"
CSV = f"{PROJECT_DIR}/results.csv"

device = "cuda"
NUM_IDS = 100
K_CLUSTERS = 4        # 대표 자세 수
STRENGTH = 0.6        # 83번 결과 보고 조정
N_GLOBAL = 50
TOP_K = 5; ALPHA = 0.7
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
TEST_MODE = False     # True면 c5만, ID 10개
SIZE = (384, 768)
SKEL_SMALL = (48, 96) # 클러스터링용 축소 크기

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

def normalize_skeleton(skel_img, target_h_ratio=0.85):
    arr = np.array(skel_img); mask = arr.sum(axis=2) > 20
    ys, xs = np.where(mask)
    if len(ys) == 0: return skel_img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = arr[y0:y1+1, x0:x1+1]; ch, cw = crop.shape[:2]
    W, H = SIZE; target_h = int(H * target_h_ratio)
    scale = target_h / ch
    new_w = max(1, int(cw * scale)); new_h = target_h
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", SIZE, (0, 0, 0))
    px = (W - new_w) // 2; py = (H - new_h) // 2
    canvas.paste(crop_img, (px, py)); return canvas

# ===================================================================
# Phase 1: 자세 클러스터링
# ===================================================================
def cluster_poses(openpose, image_paths, k, exclude_pids, cam, n_sample=200):
    """
    타겟 카메라 이미지들에서 skeleton 추출 → K-means 클러스터링
    → 대표 skeleton K개 반환 (정규화된 PIL Image)
    exclude_pids: 평가 대상 ID 제외 (leakage 방지)
    """
    # 샘플링 (전부 하면 느림)
    pool = []
    for f in image_paths:
        pid, c = parse(f)
        if pid in ('-1', '0000') or pid in exclude_pids: continue
        if c == cam: pool.append(f)
    random.shuffle(pool)
    pool = pool[:n_sample]
    if len(pool) < k:
        print(f"  {cam}: 샘플 부족 ({len(pool)}), skip")
        return []

    print(f"  {cam}: skeleton 추출 중 ({len(pool)}장)...")
    skeletons = []  # 정규화된 skeleton 이미지
    skel_vecs = []  # 클러스터링용 축소 벡터
    for f in tqdm(pool, desc=f"  {cam} skeleton", leave=False):
        img = Image.open(f).convert("RGB").resize(SIZE, Image.LANCZOS)
        skel = openpose(img)
        skel_norm = normalize_skeleton(skel.resize(SIZE, Image.LANCZOS))
        # 비어있는 skeleton 건너뛰기
        arr = np.array(skel_norm)
        if arr.sum() < 1000: continue
        skeletons.append(skel_norm)
        # 축소해서 벡터화 (클러스터링용)
        small = np.array(skel_norm.resize(SKEL_SMALL, Image.LANCZOS))
        skel_vecs.append(small.flatten().astype(np.float32))

    if len(skel_vecs) < k:
        print(f"  {cam}: 유효 skeleton 부족 ({len(skel_vecs)})")
        return []

    skel_vecs = np.array(skel_vecs)
    # L2 정규화
    norms = np.linalg.norm(skel_vecs, axis=1, keepdims=True) + 1e-9
    skel_vecs = skel_vecs / norms

    print(f"  {cam}: K-means (K={k})...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(skel_vecs)

    # 각 클러스터에서 centroid에 가장 가까운 skeleton = 대표
    representatives = []
    for ci in range(k):
        members = np.where(labels == ci)[0]
        if len(members) == 0: continue
        centroid = km.cluster_centers_[ci]
        dists = np.linalg.norm(skel_vecs[members] - centroid, axis=1)
        best = members[dists.argmin()]
        representatives.append(skeletons[best])
        print(f"    cluster {ci}: {len(members)}개, 대표 idx={best}")

    return representatives


# ===================================================================
# Phase 2: 사전 생성 (오프라인)
# ===================================================================
def build_gen_pipe():
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose",
                                          cache_dir=CKPT, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn,
        cache_dir=CKPT, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    op = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    return pipe, op

@torch.no_grad()
def generate_with_pose(pipe, c1_path, skel_img, strength, seed=42):
    """c1 base + 대표 skeleton → 합성"""
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    result = pipe(
        prompt="a photo of one person, full body, standing, surveillance camera",
        negative_prompt="two people, multiple, blurry, deformed, cropped",
        image=c1_img, control_image=skel_img,
        strength=strength, controlnet_conditioning_scale=0.8,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    return result


# ===================================================================
# Phase 3: 평가
# ===================================================================
def load_nosie(wp, ds, nc, cn):
    cfg.MODEL.NAME='ViT-B-16'; cfg.MODEL.STRIDE_SIZE=[16,16]
    cfg.MODEL.SIE_CAMERA=False; cfg.MODEL.SIE_COE=0.0; cfg.MODEL.ID_LOSS_TYPE='softmax'
    cfg.INPUT.SIZE_TRAIN=[256,128]; cfg.INPUT.SIZE_TEST=[256,128]
    cfg.INPUT.PIXEL_MEAN=[0.5,0.5,0.5]; cfg.INPUT.PIXEL_STD=[0.5,0.5,0.5]
    cfg.DATASETS.NAMES=ds; cfg.TEST.WEIGHT=wp; cfg.TEST.NECK_FEAT='before'
    try: m = make_model(cfg, num_class=nc, camera_num=0, view_num=1)
    except: m = make_model(cfg, num_class=nc, camera_num=cn, view_num=1)
    m.load_param(wp); return m.eval().to(device)

reid_tf = T.Compose([T.Resize([256,128]), T.ToTensor(),
                     T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
@torch.no_grad()
def feat_raw(model, path_or_img):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img).convert("RGB")
    else:
        img = path_or_img
    t = reid_tf(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
    return f.cpu().numpy().flatten()
def l2n(x): return x / (np.linalg.norm(x) + 1e-9)


# ===================================================================
# Main
# ===================================================================
def main():
    random.seed(42); np.random.seed(42)

    gby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'): continue
        gby[pid][cam].append(f)
    qby = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg")):
        pid, cam = parse(f); qby[pid][cam].append(f)
    all_files = sorted(glob.glob(f"{MARKET_GALLERY}/*.jpg")) + sorted(glob.glob(f"{MARKET_QUERY}/*.jpg"))
    all_qf = [f for f in sorted(glob.glob(f"{MARKET_QUERY}/*.jpg"))
              if parse(f)[0] not in ('-1', '0000')]

    cams = ["c5"] if TEST_MODE else TARGET_CAMS
    cvi = {}
    for tc in cams:
        ids = []
        for pid in sorted(gby.keys()):
            if "c1" in gby[pid] and tc in qby[pid]: ids.append(pid)
            if len(ids) >= NUM_IDS: break
        if TEST_MODE: ids = ids[:10]
        cvi[tc] = ids

    OUT = f"{PROJECT_DIR}/outputs/cluster_k{K_CLUSTERS}_s{int(STRENGTH*100)}"
    CMP = f"{PROJECT_DIR}/outputs/cluster_k{K_CLUSTERS}_s{int(STRENGTH*100)}_compare"

    # ====== Phase 1: 클러스터링 ======
    pipe, openpose = build_gen_pipe()

    print("=" * 80)
    print(f"Phase 1: 자세 클러스터링 (K={K_CLUSTERS})")
    print("=" * 80)
    cam_reps = {}  # cam → [skel_1, ..., skel_K]
    for tc in cams:
        exclude = set(cvi[tc])
        reps = cluster_poses(openpose, all_files, K_CLUSTERS, exclude, tc)
        cam_reps[tc] = reps
        # 대표 skeleton 저장 (확인용)
        rep_dir = f"{OUT}/rep_skeletons/{tc}"
        os.makedirs(rep_dir, exist_ok=True)
        for ki, skel in enumerate(reps):
            skel.save(f"{rep_dir}/cluster_{ki}.png")

    # ====== Phase 2: 사전 생성 ======
    print("\n" + "=" * 80)
    print(f"Phase 2: 사전 생성 (K={K_CLUSTERS}, strength={STRENGTH})")
    print("=" * 80)
    for tc in cams:
        reps = cam_reps[tc]
        if not reps:
            print(f"  {tc}: 대표 자세 없음, skip"); continue
        os.makedirs(f"{OUT}/{tc}", exist_ok=True)
        os.makedirs(f"{CMP}/{tc}", exist_ok=True)
        for pid in tqdm(cvi[tc], desc=f"[생성] {tc}"):
            c1_path = sorted(gby[pid]["c1"])[0]
            gen_imgs = []
            for ki, skel in enumerate(reps):
                save_path = f"{OUT}/{tc}/{pid}_k{ki}.png"
                if os.path.exists(save_path):
                    gen_imgs.append(Image.open(save_path))
                    continue
                gen = generate_with_pose(pipe, c1_path, skel, STRENGTH, seed=42+ki)
                gen.save(save_path)
                gen_imgs.append(gen)
            # 비교 이미지 (처음 3명만): [c1 | skel_0 | gen_0 | skel_1 | gen_1 | ...]
            if cvi[tc].index(pid) < 3:
                W, H = SIZE; gap = 5
                n_cols = 1 + len(reps) * 2
                canvas = Image.new("RGB", (W * n_cols + gap * (n_cols - 1), H), (255, 255, 255))
                c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
                canvas.paste(c1_img, (0, 0))
                for ki in range(len(reps)):
                    x_skel = (1 + ki * 2) * (W + gap)
                    x_gen = (2 + ki * 2) * (W + gap)
                    canvas.paste(reps[ki].resize(SIZE), (x_skel, 0))
                    canvas.paste(gen_imgs[ki].resize(SIZE), (x_gen, 0))
                canvas.save(f"{CMP}/{tc}/{pid}_compare.png")

    del pipe, openpose; torch.cuda.empty_cache()

    # ====== Phase 3: 평가 ======
    print("\n" + "=" * 80)
    print(f"Phase 3: 평가")
    print("=" * 80)
    reid = load_nosie(f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8)

    # feature 추출
    print("feature 추출...")
    pair = {}
    for tc in cams:
        reps = cam_reps[tc]
        if not reps: continue
        c1r, qr, c1p, qp, kept = [], [], [], [], []
        gen_feats_all = []  # [N × K] 각 인물의 K개 생성물 feature
        for pid in cvi[tc]:
            c1_path = sorted(gby[pid]["c1"])[0]
            q_path = sorted(qby[pid][tc])[0]
            # 생성물 확인
            gen_paths = [f"{OUT}/{tc}/{pid}_k{ki}.png" for ki in range(len(reps))]
            if not all(os.path.exists(gp) for gp in gen_paths): continue
            c1r.append(feat_raw(reid, c1_path))
            qr.append(feat_raw(reid, q_path))
            gfs = [feat_raw(reid, gp) for gp in gen_paths]
            gen_feats_all.append(gfs)
            c1p.append(c1_path); qp.append(q_path); kept.append(pid)
        pair[tc] = (np.array(c1r), np.array(qr), gen_feats_all, c1p, qp, kept)

    # global mean
    all_eval = set()
    for tc in cams:
        if tc in pair: all_eval |= set(pair[tc][4])
    pool = [f for f in all_qf if f not in all_eval]
    random.Random(42).shuffle(pool)
    gmean = np.mean([feat_raw(reid, f) for f in pool[:N_GLOBAL]], axis=0)

    logline(f"\n## [{datetime.date.today()}] script84 자세클러스터 갤러리확장 — Duke→Market")
    logline(f"K={K_CLUSTERS}, strength={STRENGTH}")
    logline(f"{'Pair':<7}{'N':<5}{'Base':<8}{'Expand':<12}{'Rerank':<12}{'c1_sim':<9}{'gen_max_sim':<12}")
    logline("-" * 85)

    sB = sE = sR = 0; cnt = 0
    for tc in cams:
        if tc not in pair: continue
        c1r, qr, gen_all, c1p, qp, kept = pair[tc]
        N = len(kept)
        if N == 0: continue

        # 보정
        qf = np.array([l2n(x - gmean) for x in qr])
        c1f = np.array([l2n(x - gmean) for x in c1r])
        # 생성물 feature (보정)
        genf_all = []  # N × K
        for i in range(N):
            genf_all.append([l2n(np.array(g) - gmean) for g in gen_all[i]])

        # sim 통계
        c1_sim = np.mean([qf[i] @ c1f[i] for i in range(N)])
        # gen_max_sim: K개 중 query와 가장 가까운 것의 평균
        gen_max_sims = []
        for i in range(N):
            sims_k = [qf[i] @ genf_all[i][k] for k in range(len(genf_all[i]))]
            gen_max_sims.append(max(sims_k))
        gen_max_sim = np.mean(gen_max_sims)

        # A. Baseline
        base_sims = qf @ c1f.T
        base = sum(1 for i in range(N) if base_sims[i].argmax() == i) / N * 100

        # B. Expand (c1 + K개 생성, ID별 max)
        exp_c = 0
        for i in range(N):
            s_c1 = qf[i] @ c1f.T  # N dim
            # K개 생성물 각각과의 유사도, ID별 max
            s_max = s_c1.copy()
            for k in range(len(genf_all[i])):
                gf_k = np.array([genf_all[j][k] if k < len(genf_all[j]) else genf_all[j][0]
                                 for j in range(N)])
                s_gen_k = qf[i] @ gf_k.T
                s_max = np.maximum(s_max, s_gen_k)
            if s_max.argmax() == i: exp_c += 1
        expand = exp_c / N * 100

        # C. Rerank (Top-K 후 K개 생성물 중 max로 s2)
        rer_c = 0
        for i in range(N):
            s1 = base_sims[i]
            topk = np.argsort(-s1)[:TOP_K]
            # 각 후보의 K개 생성물 중 query와 가장 높은 유사도
            s2_vals = []
            for j in topk:
                best_k = max(qf[i] @ genf_all[j][k] for k in range(len(genf_all[j])))
                s2_vals.append(best_k)
            s2 = np.array(s2_vals)
            final = ALPHA * s1[topk] + (1 - ALPHA) * s2
            if topk[final.argmax()] == i: rer_c += 1
        rerank = rer_c / N * 100

        logline(f"c1→{tc:<4}{N:<5}{base:<8.1f}"
                f"{expand:<5.1f}({expand-base:+.1f})  "
                f"{rerank:<5.1f}({rerank-base:+.1f})  "
                f"{c1_sim:<9.3f}{gen_max_sim:<12.3f}")
        log_csv({"date": datetime.date.today(), "script": 84, "dir": "D2M",
                 "K": K_CLUSTERS, "strength": STRENGTH, "pair": f"c1{tc}",
                 "base": round(base, 2), "expand": round(expand, 2),
                 "rerank": round(rerank, 2), "c1_sim": round(c1_sim, 3),
                 "gen_max_sim": round(gen_max_sim, 3)})
        sB += base; sE += expand; sR += rerank; cnt += 1

    if cnt:
        logline("-" * 85)
        logline(f"{'평균':<7}{'':<5}{sB/cnt:<8.1f}"
                f"{sE/cnt:<5.1f}({(sE-sB)/cnt:+.1f})  "
                f"{sR/cnt:<5.1f}({(sR-sB)/cnt:+.1f})")

    del reid; torch.cuda.empty_cache()
    logline("\n" + "=" * 85)
    logline("""해석:
  gen_max_sim > c1_sim → K개 중 하나가 query와 더 가까움 (커버리지 성공!)
  Expand/Rerank > Base → 선제적 갤러리 확장이 효과
  
  이전(랜덤 1개) 대비 K개 클러스터가 나은지 비교:
  gen_max_sim(K개) > gen_sim(1개) 이면 클러스터 의미 있음

비교이미지: [c1 | skel_0 | gen_0 | skel_1 | gen_1 | ...]
대표skeleton: {OUT}/rep_skeletons/""")

if __name__ == "__main__":
    main()