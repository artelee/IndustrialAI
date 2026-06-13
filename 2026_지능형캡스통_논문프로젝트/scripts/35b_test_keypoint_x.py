
"""
YOLO-pose X (큰 모델) + Detection confidence 강화
"""

import os, glob
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
QUERY_DIR = f"{OCC_DIR}/query"
OUT_DIR = f"{PROJECT_DIR}/outputs/keypoint_test_x"
os.makedirs(OUT_DIR, exist_ok=True)

# YOLO-pose X (큰 버전)
print("YOLO-pose X 로드 (큰 모델, 다운로드 시간 걸림)...")
model = YOLO('yolov8x-pose.pt')  # ~99MB
print("✅ 로드 완료")

BODY_PARTS = {
    "head":       [0,1,2,3,4],
    "upper_body": [5,6,7,8,9,10],
    "lower_body": [11,12,13,14,15,16],
}

# 임계값
BOX_CONF_THRESHOLD = 0.5   # 사람 박스 신뢰도
KPT_CONF_THRESHOLD = 0.5   # keypoint 신뢰도

# 테스트 5장
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))[:5]
print(f"\n테스트: {len(query_files)}장")

for i, f in enumerate(query_files):
    img = Image.open(f).convert("RGB")
    
    results = model(np.array(img), verbose=False)
    
    if len(results) == 0 or results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
        print(f"\n[{i+1}] {os.path.basename(f)}: 검출 실패 → 가림 없음으로 가정")
        continue
    
    # Box confidence 확인
    boxes = results[0].boxes
    box_confs = boxes.conf.cpu().numpy() if boxes is not None else None
    
    if box_confs is None or len(box_confs) == 0 or box_confs.max() < BOX_CONF_THRESHOLD:
        max_box = box_confs.max() if box_confs is not None and len(box_confs) > 0 else 0
        print(f"\n[{i+1}] {os.path.basename(f)}: 박스 신뢰도 낮음 ({max_box:.2f}) → 가림 없음으로 가정")
        continue
    
    # 가장 confident한 사람 선택
    best_idx = box_confs.argmax()
    xy = results[0].keypoints.xy[best_idx].cpu().numpy()
    conf = results[0].keypoints.conf[best_idx].cpu().numpy()
    
    print(f"\n[{i+1}] {os.path.basename(f)}")
    print(f"  이미지: {img.size}, Box conf: {box_confs[best_idx]:.2f}")
    
    for part_name, indices in BODY_PARTS.items():
        part_conf = conf[indices]
        visible = (part_conf > KPT_CONF_THRESHOLD).sum()
        total = len(indices)
        occluded = visible < total / 2
        print(f"  {part_name}: {visible}/{total} {'[가림]' if occluded else ''}")
    
    # 시각화
    img_vis = img.copy()
    draw = ImageDraw.Draw(img_vis)
    for j, (pt, c) in enumerate(zip(xy, conf)):
        x, y = pt
        if x > 0 and y > 0:
            if c > KPT_CONF_THRESHOLD:
                draw.ellipse([x-3, y-3, x+3, y+3], fill="green")
            else:
                draw.ellipse([x-3, y-3, x+3, y+3], outline="red", width=2)
    
    save_path = f"{OUT_DIR}/{os.path.basename(f)}"
    img_vis.save(save_path)

print(f"\n시각화 저장: {OUT_DIR}/")
print(f"Box conf 임계값: {BOX_CONF_THRESHOLD}")
print(f"Keypoint conf 임계값: {KPT_CONF_THRESHOLD}")
