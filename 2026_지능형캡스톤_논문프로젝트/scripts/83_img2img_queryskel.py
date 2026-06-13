"""
83_img2img_queryskel.py

핵심 변경:
- IP-Adapter 제거 (외형 보존 실패했으니)
- c1 이미지를 base로 img2img (외형 자연 보존)
- query의 skeleton을 ControlNet 조건 (query 자세 반영)
- strength 여러 개 비교 (0.5, 0.6, 0.65)

비교:
  A. Baseline (보정만, 생성 X)
  B. 갤러리 확장 (c1 + 생성물, max 유사도)
  C. Re-ranking (Top-K 후 재정렬, α·s1 + (1-α)·s2)

모든 설정에 global mean 보정 적용.
SIE 없는 weight. Duke→Market, 전 페어.

먼저 c5 5장으로 품질 확인 → 전체.
비교 이미지 저장: [c1 | query | skeleton | 합성물]
"""
import os, sys, glob, torch, numpy as np, csv, datetime
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
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
TOP_K = 5
ALPHA = 0.7
N_GLOBAL = 50
STRENGTHS = [0.5, 0.6, 0.65]
TARGET_CAMS = ["c2", "c3", "c4", "c5", "c6"]
TEST_MODE = False  # True면 c5 5장만, False면 전체

SIZE = (384, 768)

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
    y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = arr[y0:y1+1, x0:x1+1]; ch, cw = crop.shape[:2]
    W, H = SIZE; target_h = int(H * target_h_ratio)
    scale = target_h / ch
    new_w = max(1, int(cw * scale)); new_h = target_h
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", SIZE, (0, 0, 0))
    px = (W - new_w) // 2; py = (H - new_h) // 2
    canvas.paste(crop_img, (px, py)); return canvas

# ===== 생성 =====
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
    pipe.enable_attention_slicing()  # img2img엔 IP-Adapter 없으니 OK
    op = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CKPT)
    return pipe, op

@torch.no_grad()
def generate(pipe, openpose, c1_path, query_path, strength, save_path):
    if os.path.exists(save_path): return
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    q_img = Image.open(query_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(q_img)
    skel = normalize_skeleton(skel.resize(SIZE, Image.LANCZOS))
    result = pipe(
        prompt="a photo of one person, full body, standing, surveillance camera",
        negative_prompt="two people, multiple, blurry, deformed, cropped",
        image=c1_img,          # c1을 base로 (외형 보존)
        control_image=skel,    # query 자세 (ControlNet)
        strength=strength,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=30, guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        width=SIZE[0], height=SIZE[1],
    ).images[0]
    result.save(save_path)
    return skel  # 비교 이미지용

def make_compare(c1_path, q_path, skel, gen_path, save_path):
    c1 = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    q = Image.open(q_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    gen = Image.open(gen_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    W, H = SIZE; gap = 10
    canvas = Image.new("RGB", (W*4 + gap*3, H), (255, 255, 255))
    canvas.paste(c1, (0, 0))
    canvas.paste(q, (W+gap, 0))
    canvas.paste(skel.resize(SIZE), (W*2+gap*2, 0))
    canvas.paste(gen, (W*3+gap*3, 0))
    canvas.save(save_path)

# ===== Re-ID =====
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
def feat_raw(model, path):
    img = Image.open(path).convert("RGB")
    t = reid_tf(img).unsqueeze(0).to(device)
    try: f = model(t)
    except TypeError: f = model(t, cam_label=None)
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
        if TEST_MODE: ids = ids[:5]
        cvi[tc] = ids

    # === Phase 1: 생성 (strength별) ===
    pipe, openpose = build_gen_pipe()
    for st in STRENGTHS:
        tag = f"s{int(st*100)}"
        out_dir = f"{PROJECT_DIR}/outputs/qskel_{tag}"
        cmp_dir = f"{PROJECT_DIR}/outputs/qskel_{tag}_compare"
        for tc in TARGET_CAMS:
            os.makedirs(f"{out_dir}/{tc}", exist_ok=True)
            os.makedirs(f"{cmp_dir}/{tc}", exist_ok=True)
            for pid in tqdm(cvi[tc], desc=f"[str={st}] {tc}"):
                c1_path = sorted(gby[pid]["c1"])[0]
                q_path = sorted(qby[pid][tc])[0]
                save = f"{out_dir}/{tc}/{pid}_gen_{tc}.png"
                skel = generate(pipe, openpose, c1_path, q_path, st, save)
                # 처음 3장만 비교 저장
                if skel and cvi[tc].index(pid) < 3:
                    make_compare(c1_path, q_path, skel, save,
                                 f"{cmp_dir}/{tc}/{pid}_compare.png")
    del pipe, openpose; torch.cuda.empty_cache()

    # === Phase 2: 평가 ===
    reid = load_nosie(f"{CKPT}/clipreid_duke_nosie.pth", "dukemtmcreid", 702, 8)
    # feature 추출
    print("\nfeature 추출 중...")
    pair = {}
    for tc in TARGET_CAMS:
        c1r, qr, c1p, qp, kept = [], [], [], [], []
        for pid in cvi[tc]:
            c1_path = sorted(gby[pid]["c1"])[0]
            q_path = sorted(qby[pid][tc])[0]
            c1r.append(feat_raw(reid, c1_path))
            qr.append(feat_raw(reid, q_path))
            c1p.append(c1_path); qp.append(q_path)
            kept.append(pid)
        pair[tc] = (np.array(c1r), np.array(qr), c1p, qp, kept)

    # global mean
    all_eval = set()
    for tc in TARGET_CAMS:
        all_eval |= set(pair[tc][3])
    pool = [f for f in all_qf if f not in all_eval]
    random.Random(42).shuffle(pool)
    gmean = np.mean([feat_raw(reid, f) for f in pool[:N_GLOBAL]], axis=0)

    logline(f"\n## [{datetime.date.today()}] script83 img2img+querySkel — Duke→Market")
    logline(f"{'str':<6}{'Pair':<7}{'Base':<8}{'Expand':<12}{'Rerank':<12}"
            f"{'c1_sim':<9}{'gen_sim':<9}")
    logline("-" * 90)

    for st in STRENGTHS:
        tag = f"s{int(st*100)}"
        out_dir = f"{PROJECT_DIR}/outputs/qskel_{tag}"
        sB = sE = sR = 0; cnt = 0
        for tc in TARGET_CAMS:
            c1r, qr, c1p, qp, kept = pair[tc]
            N = len(kept)
            # 생성물 feature
            genr = []
            valid = True
            for pid in kept:
                gp = f"{out_dir}/{tc}/{pid}_gen_{tc}.png"
                if not os.path.exists(gp):
                    valid = False; break
                genr.append(feat_raw(reid, gp))
            if not valid: continue
            genr = np.array(genr)
            # 보정
            qf = np.array([l2n(x - gmean) for x in qr])
            c1f = np.array([l2n(x - gmean) for x in c1r])
            gf = np.array([l2n(x - gmean) for x in genr])
            # sim
            c1_sim = np.mean([qf[i] @ c1f[i] for i in range(N)])
            gen_sim = np.mean([qf[i] @ gf[i] for i in range(N)])
            # A. Baseline
            sims = qf @ c1f.T
            base = sum(1 for i in range(N) if sims[i].argmax() == i) / N * 100
            # B. Expand (c1 + gen, max)
            exp_c = 0
            for i in range(N):
                s = np.maximum(qf[i] @ c1f.T, qf[i] @ gf.T)
                if s.argmax() == i: exp_c += 1
            expand = exp_c / N * 100
            # C. Rerank
            rer_c = 0
            for i in range(N):
                s1 = sims[i]; topk = np.argsort(-s1)[:TOP_K]
                s2 = qf[i] @ gf[topk].T
                final = ALPHA * s1[topk] + (1 - ALPHA) * s2
                if topk[final.argmax()] == i: rer_c += 1
            rerank = rer_c / N * 100

            logline(f"{st:<6}c1→{tc:<4}{base:<8.1f}"
                    f"{expand:<5.1f}({expand-base:+.1f})  "
                    f"{rerank:<5.1f}({rerank-base:+.1f})  "
                    f"{c1_sim:<9.3f}{gen_sim:<9.3f}")
            log_csv({"date": datetime.date.today(), "script": 83, "dir": "D2M",
                     "strength": st, "pair": f"c1{tc}", "N": N,
                     "base": round(base, 2), "expand": round(expand, 2),
                     "rerank": round(rerank, 2),
                     "c1_sim": round(c1_sim, 3), "gen_sim": round(gen_sim, 3)})
            sB += base; sE += expand; sR += rerank; cnt += 1
        if cnt:
            logline(f"{st:<6}{'평균':<7}{sB/cnt:<8.1f}"
                    f"{sE/cnt:<5.1f}({(sE-sB)/cnt:+.1f})  "
                    f"{sR/cnt:<5.1f}({(sR-sB)/cnt:+.1f})")
            logline("")
    del reid; torch.cuda.empty_cache()
    logline("=" * 90)
    logline("""해석:
  gen_sim > c1_sim → 생성이 query에 더 가까움 (query 자세 반영 성공)
  gen_sim ≈ c1_sim → 자세 변형 부족 (strength 더 올려야)
  Expand/Rerank > Base → 갤러리확장/재정렬 효과 있음
  Rerank > Expand → 선별 재정렬이 더 안정적""")

if __name__ == "__main__":
    main()