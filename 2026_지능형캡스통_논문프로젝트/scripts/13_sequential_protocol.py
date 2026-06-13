
"""
Sequential Online Re-ID Protocol — Phase 1 Step 1
- 데이터 구조 정의
- 시간 시뮬레이터
- 카메라 순서대로 등장 시뮬레이션
"""

import os
import glob
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

MARKET_DIR = "/home/ubuntu/datasets/market1501/Market-1501-v15.09.15"
GALLERY_DIR = f"{MARKET_DIR}/bounding_box_test"
QUERY_DIR = f"{MARKET_DIR}/query"

# ===== 데이터 구조 =====

@dataclass
class PersonImage:
    """한 이미지의 메타데이터"""
    path: str
    pid: str           # 사람 ID
    cam: str           # c1~c6
    timestamp: int     # 가상 시간 (카메라 번호)
    is_generated: bool = False  # 생성 이미지 여부

@dataclass
class GalleryEntry:
    """갤러리 한 항목"""
    image: PersonImage
    feature: Optional[any] = None  # 나중에 채움


# ===== 데이터 로드 =====

def parse_market(filepath):
    """Market-1501 파일명 파싱"""
    fname = os.path.basename(filepath)
    parts = fname.split("_")
    pid = parts[0]
    cam = parts[1][:2]  # "c1s1" → "c1"
    return pid, cam

def cam_to_time(cam):
    """카메라 → 시간 매핑 (c1→1, c2→2, ...)"""
    return int(cam[1:])

def load_all_images():
    """gallery + query 폴더 전부 로드, PersonImage 리스트로"""
    all_images = []
    
    # Gallery 폴더
    for f in sorted(glob.glob(f"{GALLERY_DIR}/*.jpg")):
        pid, cam = parse_market(f)
        if pid in ("-1", "0000"):  # distractor 제외
            continue
        all_images.append(PersonImage(
            path=f, pid=pid, cam=cam,
            timestamp=cam_to_time(cam),
        ))
    
    # Query 폴더도 같은 사람들의 추가 사진
    for f in sorted(glob.glob(f"{QUERY_DIR}/*.jpg")):
        pid, cam = parse_market(f)
        all_images.append(PersonImage(
            path=f, pid=pid, cam=cam,
            timestamp=cam_to_time(cam),
        ))
    
    return all_images


# ===== ID 그룹화 + 통계 =====

def group_by_id(images):
    """이미지들을 ID별로 그룹화"""
    grouped = defaultdict(list)
    for img in images:
        grouped[img.pid].append(img)
    return grouped


def group_by_time(images):
    """이미지들을 timestamp별로 그룹화"""
    grouped = defaultdict(list)
    for img in images:
        grouped[img.timestamp].append(img)
    return grouped


def select_n_ids(grouped, n=5):
    """모든 카메라(c1~c6)에 등장하는 ID 중 n명 선택"""
    valid_ids = []
    for pid, imgs in sorted(grouped.items()):
        cams = set(img.cam for img in imgs)
        if cams >= {"c1", "c2", "c3", "c4", "c5", "c6"}:
            valid_ids.append(pid)
        if len(valid_ids) >= n:
            break
    return valid_ids


# ===== 실행: 데이터 확인 =====

if __name__ == "__main__":
    print("=" * 60)
    print("[Step 1] 데이터 로드 + 시간 시뮬레이션 확인")
    print("=" * 60)
    
    # 1. 전체 로드
    all_images = load_all_images()
    print(f"\n총 이미지 수: {len(all_images)}")
    
    # 2. ID별 그룹화
    by_id = group_by_id(all_images)
    print(f"총 ID 수: {len(by_id)}")
    
    # 3. 카메라별 분포
    by_time = group_by_time(all_images)
    print(f"\n시간(카메라)별 이미지 수:")
    for t in sorted(by_time.keys()):
        print(f"  t={t} (c{t}): {len(by_time[t])}장")
    
    # 4. 5명 선택 (모든 카메라 등장하는 ID)
    selected = select_n_ids(by_id, n=5)
    print(f"\n선택된 5명: {selected}")
    
    # 5. 선택된 5명의 시간순 등장 시뮬레이션
    print(f"\n--- 5명의 시간순 등장 ---")
    for t in range(1, 7):
        print(f"\n[시간 t={t}, 카메라 c{t}]")
        appearances = []
        for pid in selected:
            imgs = [img for img in by_id[pid] if img.timestamp == t]
            if imgs:
                appearances.append((pid, imgs[0]))
        
        for pid, img in appearances:
            print(f"  ID {pid} 등장: {os.path.basename(img.path)}")
