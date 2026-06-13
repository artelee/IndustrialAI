"""
철도 건널목 - 차량 & 보행자 속도 측정 (영상 단독)
============================================================
대상 영상: KakaoTalk_20260612_175959339.mp4 (1920x1080, 30fps, 20s)

사용 알고리즘 (강의 매핑)
  - Lec 8 : 지면 호모그래피 + RANSAC   (영상 픽셀 -> 실세계 미터 BEV)
  - Lec 9 : Lucas-Kanade 광류 + Coarse-to-fine 피라미드 (객체 점 추적)
            -> 빠른 차량 / 느린 자전거를 같은 파이프라인으로 처리하므로
               서로 다른 속도 스케일에 대응하는 피라미드의 필요성을 입증

스케일 기준 : 도로 평면 위 직사각형 4점 (노란 유도선 등)
              실세계 가로/세로 길이는 아래 WIDTH_M, LENGTH_M 로 입력
왜곡 보정  : 생략 (체커보드 없음) -> 화면 가장자리/원거리 오차는 한계로 명시

------------------------------------------------------------
조작법
  [1단계] 지면 직사각형 4점 클릭 : 좌상 -> 우상 -> 우하 -> 좌하
          (네 점 모두 '도로 평면 위'에, 가능한 한 차량/자전거가 실제로
           지나는 통행로 근처에 잡을 것)
  [2단계] 추적할 객체를 클릭. 객체 종류는 키로 먼저 선택:
            'v' 누른 뒤 클릭 = 차량(vehicle) 점 등록
            'p' 누른 뒤 클릭 = 보행자/자전거(pedestrian) 점 등록
          여러 개 등록 가능. 's' 누르면 추적 시작.
          * 차량은 바퀴-노면 접점, 자전거는 바퀴 접지점을 클릭 (지면 위 점!)
  [추적] ESC 로 중단. 종료 시 객체별 평균/최대 속도 출력 및 CSV 저장.

검증 팁 (속도 GT가 없으므로)
  - 영상 내 실제 거리를 아는 두 지점 통과 시간을 손으로 재서
    평균속도를 역산 -> 본 시스템 출력과 비교 (자기검증). 발표 결과 슬라이드용.
"""

import cv2
import numpy as np
import csv

# ----------------------- 사용자 설정 -----------------------
VIDEO_PATH = "KakaoTalk_20260612_175959339.mp4"

# 클릭할 직사각형의 실세계 크기 (m). 영상 내 노란 유도선/통행로 기준으로 교체.
WIDTH_M  = 3.0     # 가로(좌->우) 실제 길이
LENGTH_M = 6.0     # 세로(상->하, 진행방향) 실제 길이

START_FRAME   = 0       # 분석 시작 프레임 (자전거 구간만 보려면 ~150 등으로)
SMOOTH_WINDOW = 7       # 속도 이동평균 프레임 수
MAX_TRACK_ERR = 30.0    # LK 추적 오차 임계 (이상이면 소실 처리)
OUT_CSV = "speed_log.csv"
# ----------------------------------------------------------

# 객체 종류별 색/라벨
KIND_STYLE = {
    "vehicle":    {"color": (0, 165, 255), "label": "CAR"},   # 주황
    "pedestrian": {"color": (0, 255, 0),   "label": "PED"},   # 초록
}


def setup_points(frame):
    """1단계: 지면 4점, 2단계: 추적 객체점들을 한 창에서 수집."""
    state = {
        "phase": "ground",          # 'ground' -> 'objects'
        "ground": [],               # [(x,y) x4]
        "objects": [],              # [{'kind':..., 'pt':(x,y)}]
        "cur_kind": "vehicle",
    }
    win = "setup"

    def redraw():
        d = frame.copy()
        # 안내문
        if state["phase"] == "ground":
            msg = f"[GROUND] click rectangle 4 pts: TL->TR->BR->BL ({len(state['ground'])}/4)"
        else:
            msg = (f"[OBJECTS] key v=CAR p=PED, then click ground-contact point. "
                   f"s=start  (cur={state['cur_kind']}, n={len(state['objects'])})")
        cv2.rectangle(d, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(d, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)
        # 지면 4점
        for i, p in enumerate(state["ground"]):
            cv2.circle(d, p, 6, (0, 0, 255), -1)
            cv2.putText(d, str(i + 1), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(state["ground"]) == 4:
            cv2.polylines(d, [np.int32(state["ground"])], True, (0, 0, 255), 2)
        # 객체점
        for o in state["objects"]:
            st = KIND_STYLE[o["kind"]]
            cv2.circle(d, o["pt"], 6, st["color"], -1)
            cv2.putText(d, st["label"], (o["pt"][0] + 6, o["pt"][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, st["color"], 2)
        cv2.imshow(win, d)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["phase"] == "ground":
            if len(state["ground"]) < 4:
                state["ground"].append((x, y))
                if len(state["ground"]) == 4:
                    state["phase"] = "objects"
        else:
            state["objects"].append({"kind": state["cur_kind"], "pt": (x, y)})
        redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('v'):
            state["cur_kind"] = "vehicle"; redraw()
        elif k == ord('p'):
            state["cur_kind"] = "pedestrian"; redraw()
        elif k == ord('s') and state["phase"] == "objects" and state["objects"]:
            break
        elif k == 27:
            cv2.destroyWindow(win)
            raise SystemExit("사용자 취소")
    cv2.destroyWindow(win)
    return state["ground"], state["objects"]


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"영상 열기 실패: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (fps > 0):
        fps = 30.0
        print("경고: fps 불명 -> 30 가정")
    print(f"fps = {fps:.2f}")

    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("첫 프레임 읽기 실패")

    ground_pts, objects = setup_points(first)

    # --- 호모그래피 (Lec 8) ---
    img_pts = np.float32(ground_pts)
    world_pts = np.float32([
        [0,        0],
        [WIDTH_M,  0],
        [WIDTH_M,  LENGTH_M],
        [0,        LENGTH_M],
    ])
    H, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise SystemExit("호모그래피 추정 실패 (4점 배치 확인)")

    def to_world(pt):
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, H)
        return w[0, 0]

    # --- LK 피라미드 광류 (Lec 9) ---
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,   # coarse-to-fine 피라미드 레벨
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # 트랙 초기화
    tracks = []
    for i, o in enumerate(objects):
        tracks.append({
            "id": i,
            "kind": o["kind"],
            "pt": np.float32(o["pt"]),
            "world": to_world(o["pt"]),
            "alive": True,
            "speeds": [],       # km/h 시계열
        })

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    log_rows = []
    frame_idx = START_FRAME

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        alive = [t for t in tracks if t["alive"]]
        if alive:
            p0 = np.float32([t["pt"] for t in alive]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None, **lk_params)

            for j, t in enumerate(alive):
                if st[j][0] == 0 or err[j][0] > MAX_TRACK_ERR:
                    t["alive"] = False
                    continue
                new_pt = p1[j, 0]
                new_world = to_world(new_pt)
                dist_m = float(np.linalg.norm(new_world - t["world"]))
                inst_kmh = dist_m * fps * 3.6

                t["pt"] = new_pt
                t["world"] = new_world
                t["speeds"].append(inst_kmh)

                sm = float(np.mean(t["speeds"][-SMOOTH_WINDOW:]))
                style = KIND_STYLE[t["kind"]]
                x, y = int(new_pt[0]), int(new_pt[1])
                cv2.circle(frame, (x, y), 6, style["color"], -1)
                cv2.putText(frame, f"{style['label']} {sm:4.1f}km/h",
                            (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, style["color"], 2)

                log_rows.append([frame_idx, t["id"], t["kind"],
                                 round(inst_kmh, 2), round(sm, 2)])

        # 지면 사각형 표시
        cv2.polylines(frame, [np.int32(ground_pts)], True, (0, 0, 255), 1)
        cv2.putText(frame, f"frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("speed", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        prev_gray = gray

        if not any(t["alive"] for t in tracks):
            print("모든 트랙 소실 - 종료")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 결과 요약
    print("\n===== 결과 =====")
    for t in tracks:
        if t["speeds"]:
            print(f"[{t['kind']:10s} id{t['id']}] "
                  f"frames={len(t['speeds']):3d}  "
                  f"avg={np.mean(t['speeds']):5.1f} km/h  "
                  f"max={np.max(t['speeds']):5.1f} km/h")
        else:
            print(f"[{t['kind']:10s} id{t['id']}] 측정 없음")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "kind", "inst_kmh", "smooth_kmh"])
        w.writerows(log_rows)
    print(f"\nCSV 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()