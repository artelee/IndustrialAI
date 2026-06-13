
"""
Sequential Online Re-ID Protocol — Step 3
- Condition A (Baseline) + Condition B (Ours: 시점 합성 추가)
- 5명 셋업으로 메커니즘 검증
"""

import os
import sys
import glob
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from PIL import Image
import torchvision.transforms as T
import torchreid
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from controlnet_aux import OpenposeDetector

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
CACHE_DIR = f"{PROJECT_DIR}/checkpoints"
GEN_DIR = f"{PROJECT_DIR}/outputs/sequential_gen"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

os.makedirs(GEN_DIR, exist_ok=True)
device = "cuda"
dtype = torch.float16
SIZE = (384, 768)


# ===== 데이터 구조 (Step 1-2와 동일) =====

@dataclass
class PersonImage:
    path: str
    pid: str
    cam: str
    timestamp: int
    is_generated: bool = False


def parse_market(filepath):
    fname = os.path.basename(filepath)
    parts = fname.split("_")
    return parts[0], parts[1][:2]


def cam_to_time(cam):
    return int(cam[1:])


def load_all_images():
    all_images = []
    for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
        pid, cam = parse_market(f)
        if pid in ("-1", "0000"):
            continue
        all_images.append(PersonImage(f, pid, cam, cam_to_time(cam)))
    for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
        pid, cam = parse_market(f)
        all_images.append(PersonImage(f, pid, cam, cam_to_time(cam)))
    return all_images


def group_by_id(images):
    grouped = defaultdict(list)
    for img in images:
        grouped[img.pid].append(img)
    return grouped


def select_n_ids(grouped, n=5):
    valid_ids = []
    for pid, imgs in sorted(grouped.items()):
        cams = set(img.cam for img in imgs)
        if cams >= {"c1", "c2", "c3", "c4", "c5", "c6"}:
            valid_ids.append(pid)
        if len(valid_ids) >= n:
            break
    return valid_ids


# ===== Feature Extractor =====

class FeatureExtractor:
    def __init__(self, weights_path):
        print("OSNet 로드...")
        self.model = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=751, pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.model, weights_path)
        self.model = self.model.eval().to(device)
        
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("✅ OSNet 로드 완료")
    
    @torch.no_grad()
    def extract(self, img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(device)
        feat = self.model(tensor)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, paths, batch_size=64):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = torch.stack([
                self.transform(Image.open(p).convert("RGB")) for p in batch
            ]).to(device)
            f = self.model(imgs)
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)


# ===== View Generator =====

class ViewGenerator:
    """시점 변환 생성기 (IP-Adapter Plus + ControlNet OpenPose)"""
    
    def __init__(self):
        print("\nViewGenerator 로드 중...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", cache_dir=CACHE_DIR, torch_dtype=dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=controlnet, cache_dir=CACHE_DIR, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter-plus_sd15.safetensors", cache_dir=CACHE_DIR,
        )
        self.pipe.set_ip_adapter_scale(0.8)
        self.openpose = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=CACHE_DIR
        )
        print("✅ ViewGenerator 로드 완료")
    
    def generate(self, content_path, pose_ref_path, seed=42):
        """content_path의 외형 + pose_ref_path의 자세로 새 이미지 생성"""
        content_img = Image.open(content_path).convert("RGB").resize(SIZE, Image.LANCZOS)
        pose_img = Image.open(pose_ref_path).convert("RGB").resize(SIZE, Image.LANCZOS)
        pose_skel = self.openpose(pose_img).resize(SIZE, Image.LANCZOS)
        
        gen = torch.Generator(device=device).manual_seed(seed)
        result = self.pipe(
            prompt="a photo of a person, full body, standing, walking, photorealistic",
            negative_prompt="blurry, low quality, deformed, multiple people, stripes, artifacts",
            image=pose_skel, ip_adapter_image=content_img,
            num_inference_steps=25, guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            generator=gen, width=SIZE[0], height=SIZE[1],
        ).images[0]
        return result


# ===== Dynamic Gallery (Step 2와 동일) =====

class DynamicGallery:
    def __init__(self, extractor):
        self.extractor = extractor
        self.entries = []
    
    def add(self, img: PersonImage):
        feat = self.extractor.extract(img.path)
        self.entries.append((img, feat))
    
    def add_batch(self, imgs):
        paths = [img.path for img in imgs]
        feats = self.extractor.extract_batch(paths)
        for img, feat in zip(imgs, feats):
            self.entries.append((img, feat))
    
    def search(self, query_feat, top_k=10):
        if len(self.entries) == 0:
            return []
        gallery_feats = np.array([e[1] for e in self.entries])
        sims = gallery_feats @ query_feat
        ranking = np.argsort(-sims)[:top_k]
        return [(self.entries[i][0], sims[i]) for i in ranking]
    
    def size(self):
        return len(self.entries)


# ===== Pose Reference Pool =====

def build_pose_ref_pool(by_id, selected_ids, all_ids):
    """카메라별 pose reference 후보 (선택된 ID 제외, 다른 ID에서 가져옴)"""
    pose_pool = defaultdict(list)
    for pid, imgs in by_id.items():
        if pid in selected_ids:
            continue  # 자기 자신 cheating 방지
        for img in imgs:
            pose_pool[img.cam].append(img.path)
    return pose_pool


# ===== 시뮬레이션 실행 =====

def run_simulation(by_id, selected, extractor, condition_name, generator=None, pose_pool=None):
    """한 조건에 대한 시뮬레이션 실행"""
    print(f"\n{'='*60}")
    print(f"[{condition_name}]")
    print(f"{'='*60}")
    
    gallery = DynamicGallery(extractor)
    results = []  # (t, pid, correct, sim, top1_pid, top1_cam)
    
    for t in range(1, 7):
        cam = f"c{t}"
        print(f"\n--- t={t} ({cam}) ---")
        
        # 이번 시간 등장 (각 ID 첫 이미지)
        appearances = []
        for pid in selected:
            imgs = [img for img in by_id[pid] if img.timestamp == t]
            if imgs:
                appearances.append(imgs[0])
        
        # t > 1: 매칭 시도
        if t > 1 and gallery.size() > 0:
            for new_img in appearances:
                query_feat = extractor.extract(new_img.path)
                top_results = gallery.search(query_feat, top_k=3)
                
                if top_results:
                    top1_img, top1_sim = top_results[0]
                    is_correct = (top1_img.pid == new_img.pid)
                    status = "✅" if is_correct else "❌"
                    gen_mark = "🤖" if top1_img.is_generated else "📷"
                    print(f"    {status} ID {new_img.pid}(c{t}) → "
                          f"{gen_mark} ID {top1_img.pid}({top1_img.cam}) sim={top1_sim:.4f}")
                    results.append({
                        "t": t, "query_pid": new_img.pid,
                        "correct": is_correct, "sim": float(top1_sim),
                        "top1_pid": top1_img.pid, "top1_cam": top1_img.cam,
                        "top1_is_gen": top1_img.is_generated,
                    })
        
        # 갤러리에 추가
        gallery.add_batch(appearances)
        
        # Condition B만: 시점 합성 추가
        if generator and pose_pool:
            print(f"    [시점 합성 진행 중...]")
            for new_img in appearances:
                # 다른 카메라(c_other != cam)들의 시점으로 합성
                for other_cam in ["c1", "c2", "c3", "c4", "c5", "c6"]:
                    if other_cam == cam:
                        continue
                    
                    # Pose reference 선택 (다른 ID의 그 카메라 이미지)
                    if other_cam not in pose_pool or not pose_pool[other_cam]:
                        continue
                    
                    pose_ref = pose_pool[other_cam][0]  # 임의 첫 개
                    
                    # 생성 (캐시 활용)
                    save_path = f"{GEN_DIR}/{new_img.pid}_{cam}_to_{other_cam}.png"
                    if not os.path.exists(save_path):
                        result_img = generator.generate(new_img.path, pose_ref)
                        result_img.save(save_path)
                    
                    # 갤러리 추가
                    gen_img = PersonImage(
                        path=save_path, pid=new_img.pid, cam=other_cam,
                        timestamp=cam_to_time(other_cam), is_generated=True,
                    )
                    gallery.add(gen_img)
        
        print(f"  현재 갤러리: {gallery.size()}장")
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n[{condition_name}] 매칭 정확도: {correct}/{total} = {100*correct/total:.1f}%")
    return results


# ===== 메인 =====

if __name__ == "__main__":
    print("=" * 60)
    print("[Step 3] Condition A vs Condition B 비교")
    print("=" * 60)
    
    # 데이터
    all_images = load_all_images()
    by_id = group_by_id(all_images)
    selected = select_n_ids(by_id, n=5)
    print(f"\n선택된 5명: {selected}")
    
    # Pose reference pool (5명 제외한 다른 ID들에서)
    pose_pool = build_pose_ref_pool(by_id, set(selected), by_id.keys())
    print(f"Pose pool 크기: {[(c, len(pose_pool[c])) for c in sorted(pose_pool.keys())]}")
    
    # 모델
    extractor = FeatureExtractor(OSNET_WEIGHTS)
    
    # === Condition A: Baseline ===
    results_A = run_simulation(by_id, selected, extractor, "조건 A: Baseline")
    
    # === Condition B: Ours ===
    view_gen = ViewGenerator()
    results_B = run_simulation(
        by_id, selected, extractor, "조건 B: Ours (시점 합성)",
        generator=view_gen, pose_pool=pose_pool,
    )
    
    # === 최종 비교 ===
    print(f"\n{'='*60}")
    print(f"📊 최종 비교")
    print(f"{'='*60}")
    
    cA = sum(1 for r in results_A if r["correct"])
    cB = sum(1 for r in results_B if r["correct"])
    tA, tB = len(results_A), len(results_B)
    
    print(f"\n조건 A (Baseline):  {cA}/{tA} = {100*cA/tA:.1f}%")
    print(f"조건 B (Ours):      {cB}/{tB} = {100*cB/tB:.1f}%")
    print(f"변화: {100*(cB/tB - cA/tA):+.1f}%p")
    
    # 시간별 비교
    print(f"\n시간별 비교:")
    print(f"{'t':<5} {'A correct':<12} {'B correct':<12}")
    for t in range(2, 7):
        aA = sum(1 for r in results_A if r["t"] == t and r["correct"])
        aB = sum(1 for r in results_B if r["t"] == t and r["correct"])
        nA = sum(1 for r in results_A if r["t"] == t)
        nB = sum(1 for r in results_B if r["t"] == t)
        print(f"t={t}  {aA}/{nA}        {aB}/{nB}")
    
    # B에서 생성 이미지가 Top-1인 케이스
    gen_top1 = sum(1 for r in results_B if r.get("top1_is_gen", False))
    print(f"\n조건 B에서 생성 이미지가 Top-1 매칭: {gen_top1}/{tB}")
