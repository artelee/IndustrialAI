"""
YOLO-pose 테스트: 가림 검출 잘 되는지 확인
- query 이미지 5장 샘플
- keypoint 검출
- 누락된 영역 시각화
"""

import os, glob
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
QUERY_DIR = f"{OCC_DIR}/query"
OUT_DIR = f"{PROJECT_DIR}/outputs/keypoint_test"
os.makedirs(OUT_DIR, exist_ok=True)

# YOLO-pose 로드
print("YOLO-pose 로드...")
model = YOLO('yolov8n-pose.pt')  # 가장 작은 버전
print("✅ 로드 완료")

# COCO keypoint 이름
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 부위별 그룹
BODY_PARTS = {
    "head": [0, 1, 2, 3, 4],
    "upper_body": [5, 6, 7, 8, 9, 10],
    "lower_body": [11, 12, 13, 14, 15, 16],
}

# 테스트 이미지 5장
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))[:5]

print(f"\n테스트 이미지: {len(query_files)}장")

for i, f in enumerate(query_files):
    img = Image.open(f).convert("RGB")
    
    # Keypoint 검출
    results = model(np.array(img), verbose=False)
    
    if len(results) == 0 or results[0].keypoints is None:
        print(f"[{i+1}] {os.path.basename(f)}: 검출 실패")
        continue
    
    kpts = results[0].keypoints
    if kpts.xy.shape[0] == 0:
        print(f"[{i+1}] {os.path.basename(f)}: 사람 없음")
        continue
    
    # 첫 사람의 keypoint
    xy = kpts.xy[0].cpu().numpy()  # (17, 2)
    conf = kpts.conf[0].cpu().numpy()  # (17,)
    
    print(f"\n[{i+1}] {os.path.basename(f)}")
    print(f"  이미지 크기: {img.size}")
    
    # 부위별 가림 여부
    for part_name, indices in BODY_PARTS.items():
        part_conf = conf[indices]
        visible = (part_conf > 0.5).sum()
        total = len(indices)
        occluded = visible < total / 2
        print(f"  {part_name}: 보이는 keypoint {visible}/{total} {'[가림]' if occluded else ''}")
    
    # 시각화
    img_vis = img.copy()
    draw = ImageDraw.Draw(img_vis)
    for j, (pt, c) in enumerate(zip(xy, conf)):
        if c > 0.5:
            x, y = pt
            draw.ellipse([x-3, y-3, x+3, y+3], fill="green")
        else:
            x, y = pt
            if x > 0 and y > 0:
                draw.ellipse([x-3, y-3, x+3, y+3], outline="red", width=2)
    
    save_path = f"{OUT_DIR}/{os.path.basename(f)}"
    img_vis.save(save_path)

print(f"\n시각화 저장: {OUT_DIR}/")
print("\n초록 = 검출됨, 빨강 = 신뢰도 낮음 (가림 추정)")