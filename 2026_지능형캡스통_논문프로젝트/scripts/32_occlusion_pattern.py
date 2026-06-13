"""
Step 1: 카메라별 가림 패턴 분석
- Occluded-Duke의 query 이미지에서
- 각 카메라별 어떤 영역이 가려지는지 분석
"""

import os, glob, numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

HOME = os.path.expanduser("~")
PROJECT_DIR = f"{HOME}/reid-gallery-expansion"
OCC_DIR = "/home/ubuntu/datasets/occluded_duke"
QUERY_DIR = f"{OCC_DIR}/query"
OUT_DIR = f"{PROJECT_DIR}/outputs/occlusion_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def parse(f):
    name = os.path.basename(f).split(".")[0]
    parts = name.split("_")
    return parts[0], parts[1]

# 가림 검출: 검은 영역 비율
def detect_occlusion_map(img_path, threshold=30):
    """이미지를 grid로 나눠서 어두운 영역 비율 계산"""
    img = np.array(Image.open(img_path).convert("L"))  # grayscale
    H, W = img.shape
    
    # 8x4 grid
    grid_h, grid_w = 8, 4
    cell_h, cell_w = H // grid_h, W // grid_w
    
    occlusion = np.zeros((grid_h, grid_w))
    for i in range(grid_h):
        for j in range(grid_w):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # 어두운 픽셀 비율
            dark_ratio = (cell < threshold).sum() / cell.size
            occlusion[i, j] = dark_ratio
    return occlusion

# 카메라별 가림 평균 계산
print("Query 이미지 분석...")
query_files = sorted(glob.glob(f"{QUERY_DIR}/*.jpg"))

cam_occlusion = defaultdict(list)
for f in tqdm(query_files):
    pid, cam = parse(f)
    occ_map = detect_occlusion_map(f)
    cam_occlusion[cam].append(occ_map)

print(f"\n분석 완료. 카메라: {sorted(cam_occlusion.keys())}")

# 카메라별 평균 가림 맵
cam_means = {}
for cam, maps in cam_occlusion.items():
    cam_means[cam] = np.mean(maps, axis=0)
    print(f"\n[{cam}] 이미지 수: {len(maps)}")
    print(f"  평균 가림 비율 영역 (상위 5개):")
    flat = cam_means[cam].flatten()
    top5_idx = np.argsort(-flat)[:5]
    for idx in top5_idx:
        i, j = idx // 4, idx % 4
        print(f"    grid [{i},{j}] (행={i}/8, 열={j}/4): {flat[idx]:.3f}")

# 시각화
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (cam, mean_map) in enumerate(sorted(cam_means.items())):
    ax = axes[i // 4, i % 4]
    im = ax.imshow(mean_map, cmap='hot', vmin=0, vmax=mean_map.max())
    ax.set_title(f'{cam} (n={len(cam_occlusion[cam])})')
    ax.set_xlabel('width grid')
    ax.set_ylabel('height grid')
    plt.colorbar(im, ax=ax)

plt.suptitle("카메라별 평균 가림 영역 (밝을수록 가려진 비율 높음)", fontsize=14)
plt.tight_layout()
save_path = f"{OUT_DIR}/camera_occlusion_pattern.png"
plt.savefig(save_path, dpi=100, bbox_inches='tight')
print(f"\n시각화 저장: {save_path}")

# 카메라별 통계 요약
print("\n" + "="*60)
print("카메라별 가림 패턴 요약")
print("="*60)
for cam in sorted(cam_means.keys()):
    m = cam_means[cam]
    total_occ = m.mean()
    upper_occ = m[:4].mean()  # 상반신
    lower_occ = m[4:].mean()  # 하반신
    print(f"{cam}: 전체 {total_occ:.3f}, 상반신 {upper_occ:.3f}, 하반신 {lower_occ:.3f}")

# 결과 저장
np.savez(f"{OUT_DIR}/camera_occlusion_means.npz", **cam_means)
print(f"\n패턴 데이터 저장: {OUT_DIR}/camera_occlusion_means.npz")
