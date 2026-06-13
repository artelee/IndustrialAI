"""
Sequential Online Re-ID Protocol — Phase 1 Step 2
- DynamicGallery 클래스: 시간 흐름에 따라 갤러리 동적 확장
- Feature extractor 통합
- 매칭 함수
"""

import os
import sys
import glob
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
import torchvision.transforms as T
import torchreid
from tqdm import tqdm

# ===== 경로 =====
HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"
OSNET_WEIGHTS = "/home/ubuntu/model/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"

device = "cuda"


# ===== 데이터 구조 (Step 1과 동일) =====

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
    """OSNet 기반 특징 추출기"""
    
    def __init__(self, weights_path):
        print("OSNet 로드 중...")
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
        """단일 이미지 → 512차원 정규화 벡터"""
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(device)
        feat = self.model(tensor)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, paths, batch_size=64):
        """배치 단위 추출 (속도 ↑)"""
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


# ===== Dynamic Gallery =====

class DynamicGallery:
    """시간에 따라 동적으로 항목을 추가하는 갤러리"""
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.entries = []  # list of (PersonImage, feature)
    
    def add(self, img: PersonImage):
        """갤러리에 이미지 추가 (특징 즉시 추출)"""
        feat = self.extractor.extract(img.path)
        self.entries.append((img, feat))
    
    def add_batch(self, imgs):
        """여러 이미지 한 번에 추가 (속도 ↑)"""
        paths = [img.path for img in imgs]
        feats = self.extractor.extract_batch(paths)
        for img, feat in zip(imgs, feats):
            self.entries.append((img, feat))
    
    def search(self, query_feat, top_k=10):
        """쿼리 특징으로 갤러리 검색, Top-K 반환"""
        if len(self.entries) == 0:
            return []
        gallery_feats = np.array([e[1] for e in self.entries])
        sims = gallery_feats @ query_feat
        ranking = np.argsort(-sims)[:top_k]
        return [(self.entries[i][0], sims[i]) for i in ranking]
    
    def size(self):
        return len(self.entries)


# ===== Step 2 실행: 5명 시뮬레이션 =====

if __name__ == "__main__":
    print("=" * 60)
    print("[Step 2] 동적 갤러리 + Feature 추출 검증")
    print("=" * 60)
    
    # 1. 데이터 로드
    all_images = load_all_images()
    by_id = group_by_id(all_images)
    selected = select_n_ids(by_id, n=5)
    print(f"\n선택된 5명: {selected}")
    
    # 2. Feature extractor 초기화
    print()
    extractor = FeatureExtractor(OSNET_WEIGHTS)
    
    # 3. 동적 갤러리 시뮬레이션 — Condition A (Baseline)
    print(f"\n{'='*60}")
    print(f"[조건 A: Baseline] 동적 갤러리만 (시점 합성 없음)")
    print(f"{'='*60}")
    
    gallery_A = DynamicGallery(extractor)
    
    for t in range(1, 7):
        print(f"\n--- 시간 t={t} (카메라 c{t}) ---")
        
        # 이번 시간에 등장하는 5명의 이미지 (각 ID당 첫 이미지만)
        appearances = []
        for pid in selected:
            imgs = [img for img in by_id[pid] if img.timestamp == t]
            if imgs:
                appearances.append(imgs[0])
        
        # 만약 t>1이면 먼저 매칭 시도
        if t > 1 and gallery_A.size() > 0:
            print(f"  [매칭 시도] 갤러리 크기: {gallery_A.size()}장")
            for new_img in appearances:
                query_feat = extractor.extract(new_img.path)
                top_results = gallery_A.search(query_feat, top_k=3)
                
                if top_results:
                    top1_img, top1_sim = top_results[0]
                    is_correct = (top1_img.pid == new_img.pid)
                    status = "✅" if is_correct else "❌"
                    print(f"    {status} Query ID {new_img.pid} (c{t}) "
                          f"→ Top-1: ID {top1_img.pid} (c{top1_img.cam[1]}) "
                          f"sim={top1_sim:.4f}")
        
        # 그 다음 갤러리에 추가
        print(f"  [갤러리 등록] {len(appearances)}장 추가")
        gallery_A.add_batch(appearances)
        print(f"  갤러리 현재 크기: {gallery_A.size()}장")
    
    print(f"\n{'='*60}")
    print(f"📊 Condition A 완료")
    print(f"최종 갤러리 크기: {gallery_A.size()}장 (5명 × 6 카메라)")
    print(f"{'='*60}")
