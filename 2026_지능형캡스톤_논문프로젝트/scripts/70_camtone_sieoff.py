"""
70_camtone_sieoff.py

목적:
1. SIE off (cam_label 미사용)로 Cross-Domain 평가 — Training-Free 원칙 일관
2. 타겟 카메라의 조명/화각 정보를 알고리즘적으로 추출 (학습 X)
3. 생성 이미지에 카메라 톤 반영 (color transfer)
4. 양방향: Duke학습→Market평가, Market학습→Duke평가

비교:
  Baseline      : c1 갤러리만
  Ours-pose     : c1 base + 자세 생성 (조명 미반영, 기존)
  Ours-camtone  : c1 base + 자세 + 카메라 톤 보정 (신규)

핵심 질문: camtone > pose 인가? (조명/화각 반영 효과)

사용법:
  conda activate reid-gen
  cd ~/reid-gallery-expansion
  cp /tmp/70_camtone_sieoff.py scripts/
  python scripts/70_camtone_sieoff.py
"""

import os, sys, glob, torch, numpy as np
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, "/home/ubuntu/CLIP-ReID")
from config import cfg
from model.make_model_clipreid import make_model

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"

MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
MARKET_GALLERY = f"{MARKET_DIR}/bounding_box_test"
MARKET_QUERY = f"{MARKET_DIR}/query"
MARKET_TRAIN = f"{MARKET_DIR}/bounding_box_train"

DUKE_DIR = "/home/ubuntu/datasets/dukemtmc-reid/DukeMTMC-reID"
DUKE_GALLERY = f"{DUKE_DIR}/bounding_box_test"
DUKE_QUERY = f"{DUKE_DIR}/query"
DUKE_TRAIN = f"{DUKE_DIR}/bounding_box_train"

# 기존 자세만 생성 이미지 (조명 미반영)
MARKET_GEN_POSE = f"{PROJECT_DIR}/outputs/c1base_gen_all"      # c2~c6
DUKE_GEN_POSE = f"{PROJECT_DIR}/outputs/duke_c1base_gen"        # c2~c8

# 신규 카메라 톤 반영 생성 저장
MARKET_GEN_TONE = f"{PROJECT_DIR}/outputs/market_camtone"
DUKE_GEN_TONE = f"{PROJECT_DIR}/outputs/duke_camtone"
os.makedirs(MARKET_GEN_TONE, exist_ok=True)
os.makedirs(DUKE_GEN_TONE, exist_ok=True)

device = "cuda"
NUM_IDS = 100  # 페어당 평가 ID 수 (전체 돌리려면 9999)

def parse(f):
    p = os.path.basename(f).split("_")
    return p[0], p[1][:2]   # pid, cam (c1,c2..)


# =========================================================
# PART 1. 카메라 톤 프로파일 추출 (알고리즘적, 학습 X)
# =========================================================
def extract_camera_tone(image_paths, n_sample=50):
    """
    타겟 카메라의 여러 이미지에서 조명/색감 통계 추출.
    표준 방식: Lab 색공간의 채널별 평균·표준편차 (Reinhard color transfer 기반).
    학습 없음. 단순 통계.
    """
    import cv2
    means, stds = [], []
    sampled = image_paths[:n_sample]
    for p in sampled:
        img = cv2.imread(p)
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        means.append(lab.reshape(-1, 3).mean(axis=0))
        stds.append(lab.reshape(-1, 3).std(axis=0))
    means = np.array(means).mean(axis=0)   # [L, a, b] 평균 밝기/색온도
    stds = np.array(stds).mean(axis=0)     # [L, a, b] 대비/채도
    return {"mean": means, "std": stds}


def apply_camera_tone(src_pil, tone):
    """
    Reinhard color transfer: 생성 이미지를 타겟 카메라 톤으로 보정.
    src의 통계를 tone(타겟 카메라 통계)으로 이동. 표준 color transfer 방식.
    """
    import cv2
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2LAB).astype(np.float32)
    s_mean = src.reshape(-1, 3).mean(axis=0)
    s_std = src.reshape(-1, 3).std(axis=0) + 1e-6
    # 표준화 후 타겟 통계로 재스케일
    out = (src - s_mean) / s_std * tone["std"] + tone["mean"]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out_rgb)


# =========================================================
# PART 2. 데이터 로드 헬퍼
# =========================================================
def load_split(gallery_dir, query_dir, target_cams):
    gallery_by_id = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, cam = parse(f)
        if pid in ('-1', '0000'):
            continue
        gallery_by_id[pid][cam].append(f)
    query_by_id = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"{query_dir}/*.jpg")):
        pid, cam = parse(f)
        query_by_id[pid][cam].append(f)
    cam_valid_ids = {}
    for tc in target_cams:
        ids = []
        for pid in sorted(gallery_by_id.keys()):
            if "c1" in gallery_by_id[pid] and tc in query_by_id[pid]:
                ids.append(pid)
            if len(ids) >= NUM_IDS:
                break
        cam_valid_ids[tc] = ids
    return gallery_by_id, query_by_id, cam_valid_ids


def collect_cam_images(gallery_dir, cam):
    """카메라 톤 추출용: 해당 카메라의 모든 이미지 경로"""
    paths = []
    for f in sorted(glob.glob(f"{gallery_dir}/*.jpg")):
        pid, c = parse(f)
        if pid in ('-1', '0000'):
            continue
        if c == cam:
            paths.append(f)
    return paths


# =========================================================
# PART 3. 생성 (자세 + 카메라 톤)
# =========================================================
def build_pipe():
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import OpenposeDetector
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        cache_dir=CACHE_DIR, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
    return pipe, openpose


def generate_with_tone(pipe, openpose, c1_path, pose_ref_path, tone, save_path):
    """
    c1 base + 자세(ControlNet) 생성 후 → 카메라 톤 color transfer 적용.
    = 조명/화각(색감 통계) 반영
    """
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")
    SIZE = (384, 768)
    c1_img = Image.open(c1_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
    skel = openpose(pose_img).resize(SIZE, Image.LANCZOS)
    gen = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt="a photo of a person, full body, surveillance",
        negative_prompt="blurry, low quality, deformed, multiple people",
        image=c1_img, control_image=skel,
        strength=0.4, num_inference_steps=30,
        guidance_scale=7.5, controlnet_conditioning_scale=0.8,
        generator=gen, width=SIZE[0], height=SIZE[1],
    ).images[0]
    # 카메라 톤 반영
    result = apply_camera_tone(result, tone)
    result.save(save_path)
    return result


# =========================================================
# PART 4. CLIP-ReID 로드 (SIE OFF가 핵심)
# =========================================================
def load_clipreid_sieoff(weight_path, dataset_name, num_class, camera_num):
    cfg.MODEL.NAME = 'ViT-B-16'
    cfg.MODEL.STRIDE_SIZE = [12, 12]
    cfg.MODEL.SIE_CAMERA = True      # 모델 구조는 SIE 포함 (weight 호환)
    cfg.MODEL.SIE_COE = 1.0
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.INPUT.SIZE_TRAIN = [256, 128]
    cfg.INPUT.SIZE_TEST = [256, 128]
    cfg.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
    cfg.DATASETS.NAMES = dataset_name
    cfg.TEST.WEIGHT = weight_path
    cfg.TEST.NECK_FEAT = 'before'
    m = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    m.load_param(weight_path)
    return m.eval().to(device)

transform = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.no_grad()
def feat_sieoff(model, img_or_path):
    """
    SIE OFF: cam_label을 주지 않음 (None).
    CLIP-ReID forward는 cam_label=None이면 SIE 더하지 않음.
    → Training-Free 원칙과 일관 (카메라 정보 매칭 단계 미사용)
    """
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path
    t = transform(img).unsqueeze(0).to(device)
    f = model(t, cam_label=None)   # ← 핵심: SIE 미적용
    return torch.nn.functional.normalize(f, p=2, dim=1).cpu().numpy().flatten()


# =========================================================
# PART 5. 평가
# =========================================================
def eval_three_ways(model, gallery_by_id, query_by_id, valid_ids,
                    target_cam, gen_pose_dir, gen_tone_dir):
    """
    Baseline : c1 갤러리만
    Ours-pose: 자세만 생성 (조명 미반영)
    Ours-tone: 자세 + 카메라 톤
    각 방법 = 갤러리에 c1 + 생성이미지(동일ID) 추가, query와 매칭
    """
    c1_feats, pose_feats, tone_feats, query_feats = [], [], [], []
    kept_ids = []
    for pid in valid_ids:
        c1_path = sorted(gallery_by_id[pid]["c1"])[0]
        pose_path = f"{gen_pose_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        tone_path = f"{gen_tone_dir}/{target_cam}/{pid}_gen_{target_cam}.png"
        q_path = sorted(query_by_id[pid][target_cam])[0]
        if not (os.path.exists(pose_path) and os.path.exists(tone_path)):
            continue
        c1_feats.append(feat_sieoff(model, c1_path))
        pose_feats.append(feat_sieoff(model, pose_path))
        tone_feats.append(feat_sieoff(model, tone_path))
        query_feats.append(feat_sieoff(model, q_path))
        kept_ids.append(pid)

    if len(kept_ids) == 0:
        return None
    c1f = np.array(c1_feats); pf = np.array(pose_feats)
    tf = np.array(tone_feats); qf = np.array(query_feats)
    N = len(kept_ids)

    def r1_single(gf):
        sims = qf @ gf.T
        return sum(1 for i in range(N) if sims[i].argmax() == i) / N

    def r1_expanded(extra_gf):
        # 갤러리 = c1 + 생성 (각 ID 2장), max 유사도로 ID 점수
        correct = 0
        for i in range(N):
            s_c1 = qf[i] @ c1f.T          # N
            s_gen = qf[i] @ extra_gf.T    # N
            s = np.maximum(s_c1, s_gen)   # ID별 max
            if s.argmax() == i:
                correct += 1
        return correct / N

    return {
        "N": N,
        "baseline": r1_single(c1f),
        "ours_pose": r1_expanded(pf),
        "ours_tone": r1_expanded(tf),
    }


# =========================================================
# MAIN
# =========================================================
def run_direction(name, train_weight, train_dataset, num_class, train_camnum,
                  eval_gallery, eval_query, eval_train_dir, target_cams,
                  gen_pose_dir, gen_tone_dir):
    print("\n" + "=" * 75)
    print(f"방향: {name}  (SIE OFF)")
    print("=" * 75)

    gallery_by_id, query_by_id, cam_valid_ids = load_split(
        eval_gallery, eval_query, target_cams)

    # --- 카메라 톤 프로파일 추출 (평가셋 갤러리에서, 알고리즘적) ---
    print("카메라 톤 프로파일 추출 중...")
    tone_profiles = {}
    for tc in target_cams:
        cam_imgs = collect_cam_images(eval_gallery, tc)
        tone_profiles[tc] = extract_camera_tone(cam_imgs)
        m = tone_profiles[tc]["mean"]
        print(f"  {tc}: L={m[0]:.1f} a={m[1]:.1f} b={m[2]:.1f}  (n_id={len(cam_valid_ids[tc])})")

    # --- 카메라 톤 반영 생성 (없으면 생성) ---
    need_gen = False
    for tc in target_cams:
        for pid in cam_valid_ids[tc]:
            if not os.path.exists(f"{gen_tone_dir}/{tc}/{pid}_gen_{tc}.png"):
                need_gen = True; break
        if need_gen: break

    if need_gen:
        print("카메라 톤 반영 생성 시작 (pose 생성물에 color transfer)...")
        # pose 생성물이 이미 있으면 거기에 톤만 입힘 (빠름)
        for tc in target_cams:
            os.makedirs(f"{gen_tone_dir}/{tc}", exist_ok=True)
            for pid in tqdm(cam_valid_ids[tc], desc=f"tone {tc}"):
                tone_path = f"{gen_tone_dir}/{tc}/{pid}_gen_{tc}.png"
                if os.path.exists(tone_path):
                    continue
                pose_path = f"{gen_pose_dir}/{tc}/{pid}_gen_{tc}.png"
                if os.path.exists(pose_path):
                    # 기존 자세 생성물에 톤만 입힘
                    base = Image.open(pose_path).convert("RGB")
                    toned = apply_camera_tone(base, tone_profiles[tc])
                    toned.save(tone_path)
                # pose 생성물 없으면 skip (먼저 46/47 돌려 생성 필요)
    else:
        print("카메라 톤 생성물 이미 존재.")

    # --- 모델 로드 (SIE off로 추론) ---
    print(f"\n{train_dataset} 학습 weight 로드 (추론 시 SIE off)...")
    model = load_clipreid_sieoff(train_weight, train_dataset, num_class, train_camnum)

    # --- 평가 ---
    print("\n평가 중...")
    results = {}
    for tc in target_cams:
        r = eval_three_ways(model, gallery_by_id, query_by_id,
                            cam_valid_ids[tc], tc, gen_pose_dir, gen_tone_dir)
        if r:
            results[tc] = r

    # --- 출력 ---
    print("\n" + "-" * 75)
    print(f"[{name}] SIE OFF 결과")
    print(f"{'Pair':<8}{'N':<6}{'Baseline':<12}{'Ours-pose':<14}{'Ours-tone':<14}")
    print("-" * 75)
    sum_b = sum_p = sum_t = 0; cnt = 0
    for tc, r in results.items():
        b, p, t = r["baseline"]*100, r["ours_pose"]*100, r["ours_tone"]*100
        dp = p - b; dt = t - b
        print(f"c1→{tc:<5}{r['N']:<6}{b:<12.2f}{p:<6.2f}({dp:+.1f})   {t:<6.2f}({dt:+.1f})")
        sum_b += b; sum_p += p; sum_t += t; cnt += 1
    if cnt:
        print("-" * 75)
        print(f"{'평균':<8}{'':<6}{sum_b/cnt:<12.2f}"
              f"{sum_p/cnt:<6.2f}({(sum_p-sum_b)/cnt:+.1f})   "
              f"{sum_t/cnt:<6.2f}({(sum_t-sum_b)/cnt:+.1f})")
    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    # === 방향 1: Duke 학습 → Market 평가 ===
    run_direction(
        name="Duke학습 → Market평가",
        train_weight=f"{PROJECT_DIR}/checkpoints/clipreid_duke/ViT-B-16_60.pth",
        train_dataset="dukemtmcreid", num_class=702, train_camnum=8,
        eval_gallery=MARKET_GALLERY, eval_query=MARKET_QUERY, eval_train_dir=MARKET_TRAIN,
        target_cams=["c2", "c3", "c4", "c5", "c6"],
        gen_pose_dir=MARKET_GEN_POSE, gen_tone_dir=MARKET_GEN_TONE,
    )

    # === 방향 2: Market 학습 → Duke 평가 ===
    run_direction(
        name="Market학습 → Duke평가",
        train_weight=f"{PROJECT_DIR}/checkpoints/clipreid/ViT-B-16_60.pth",
        train_dataset="market1501", num_class=751, train_camnum=6,
        eval_gallery=DUKE_GALLERY, eval_query=DUKE_QUERY, eval_train_dir=DUKE_TRAIN,
        target_cams=["c2", "c3", "c4", "c5", "c6", "c8"],
        gen_pose_dir=DUKE_GEN_POSE, gen_tone_dir=DUKE_GEN_TONE,
    )

    print("\n\n" + "=" * 75)
    print("해석 가이드")
    print("=" * 75)
    print("""
- Ours-tone > Ours-pose  → 카메라 조명/화각 반영이 효과 있음 (기여점 ② 입증)
- Ours-tone ≈ Ours-pose  → 색감 보정은 무의미, 자세만으로 충분
- Ours-pose > Baseline   → 시점 보강 자체가 Cross-Domain에 도움
- 모두 ≈ Baseline        → 강한 모델 feature에서 생성 변별력 부족 (기존 발견 재확인)

SIE off이므로 "카메라 정보를 매칭 단계에 안 씀" = Training-Free 일관.
카메라 정보는 오직 생성 단계(톤 보정)에서만 사용.
""")