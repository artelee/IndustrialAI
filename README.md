# 충북대학교 산업인공지능학과 25학번 이은혜 포트폴리오

> 충북대학교 산업인공지능학과 재학 중 2025~2026년에 수행한 수업 실습 및 프로젝트 포트폴리오 저장소
> 머신러닝 기초부터 딥러닝, 강화학습, 컴퓨터 비전까지 이론 학습과 구현·프로젝트를 단계적으로 진행한 결과물을 정리함.

각 폴더는 독립된 저장소(submodule) 또는 단일 산출물 폴더로 구성되어 있으며, 폴더별로 자체 README가 함께 포함됨.

---

## 2026년

### [2026_지능형캡스톤_실습](2026_지능형캡스톤_실습)
지능형 캡스톤 수업의 OpenCV 기초 실습 저장소임. 이미지/비디오/웹캠 입출력, BGR·HSV·YUV 색공간 분리, HSV 기반 특정 색상 추출, 그레이/BGR 히스토그램, `filter2D` 기반 에지 추출 등 컴퓨터 비전 기본기를 예제(ex)와 실습과제(lab)로 나누어 정리함.

### [2026_산업컴퓨터비전실제_실습_VISION알고리즘](2026_산업컴퓨터비전실제_실습_VISION알고리즘)
산업 컴퓨터 비전 실제 수업의 알고리즘 실습 저장소임. 이미지 입출력·색공간 변환·감마 보정·히스토그램 평활화 같은 기초부터, **Unsharp Mask · DFT 기반 주파수 필터링 · 모폴로지 연산**(Week 4), **Harris / FAST / GFTT 코너 검출과 SIFT 기반 스케일 불변 특징점**(Week 7) 같은 주차별 심화 주제를 다룸.

### [2026_산업컴퓨터비전실제_중간프로젝트_날씨분류기](2026_산업컴퓨터비전실제_중간프로젝트_날씨분류기)
산업 컴퓨터 비전 실제 **중간 프로젝트** 임. 철도 건널목 CCTV 영상이 안개 / 비 / 눈 중 어떤 악천후에 해당하는지 자동 판별한 뒤, 날씨별 맞춤 전처리(예: dehazing, derain, desnow)를 적용하고 YOLO 검출 성능까지 비교하는 **날씨 적응형 영상 전처리 파이프라인**을 Tkinter GUI로 구현함. ACDC 데이터셋을 사용하고 PSNR / SSIM / 에지 / 대비 / SNR 지표로 정량 평가함.

### [2026_산업컴퓨터비전실제_중간프로젝트_발표](2026_산업컴퓨터비전실제_중간프로젝트_발표)
위 날씨 분류기 프로젝트의 중간 발표 자료(PDF)임.

---

## 2025년

### [2025_어프렌티스프로젝트_실습](2025_어프렌티스프로젝트_실습)
NumPy 기초부터 경사하강법, 단·다변량 선형회귀, 결정계수(R²), Ridge/Lasso 정규화까지 머신러닝의 수학적 기반을 직접 구현하며 학습한 실습 노트북 모음임. scikit-learn / scipy 결과와 자체 구현을 교차 검증하여 알고리즘 동작을 체계적으로 이해하는 데 중점을 둠.

### [2025_어프렌티스프로젝트_미세먼지모델_프로젝트](2025_어프렌티스프로젝트_미세먼지모델_프로젝트)
어프렌티스 프로젝트(팀 **튀김소보로**) 산출물임. 에어코리아(AirKorea)에서 전국 17개 시·도의 시간별 PM10·PM2.5 데이터를 직접 크롤링하고, lag·rolling 기반 시계열 피처 엔지니어링을 거쳐 약 **1,122만 건** 규모의 학습 데이터를 구성함. LinearRegression / Ridge / Lasso / MLP를 비교하여 미세먼지 농도 예측 모델을 구축한 회귀 분석 프로젝트임.

### [2025_딥러닝실제_실습](2025_딥러닝실제_실습)
딥러닝실제 강의 실습 모음임. OpenCV 기반 영상 입출력·히스토그램·모폴로지·에지/영역 검출 같은 고전 영상 처리에서 시작해 CNN 학습, ResNet50 / DenseNet121 전이학습, PyQt5 GUI 응용까지 컴퓨터 비전 전체 파이프라인을 다룸.

### [2025_강화학습실제_실습](2025_강화학습실제_실습)
강화학습실제 강의 주차별 실습 모음임. 동적 계획법(정책/가치 반복), 몬테카를로, 시간차 학습(SARSA, Q-learning), 함수 근사(Q-Network), 정책 기반 학습(REINFORCE, Actor-Critic)까지 강화학습 핵심 알고리즘을 GridWorld·CartPole·MountainCar 환경에서 직접 구현함. 딥러닝 프레임워크는 `dezero`를 사용함.

### [2025_강화학습실제_qlearning_프로젝트](2025_강화학습실제_qlearning_프로젝트)
강화학습실제 **중간 프로젝트** 임. 4×4 GridWorld 환경에서 Q-learning을 처음부터 구현해 최적 경로를 학습시키고, 학습 종료 후 추가되는 **동적 장애물(dynamic walls)** 에 대해 학습된 Q-table만으로 우회 경로를 재탐색하도록 확장함. V-value / Q-value / Policy / 경로를 격자 위에 시각화함.

### [2025_강화학습실제_DQN구현](2025_강화학습실제_DQN구현)
강화학습실제 수업의 DQN 및 Actor-Critic 직접 구현 프로젝트임. CartPole에서 시작해 희소 보상 환경인 MountainCar로 난이도를 높여 가며, 네트워크 구조·ε-decay·reward shaping을 비교 실험함. `dezero` 기반으로 ReplayBuffer / TargetNet / ε-greedy를 구현하고, 테스트 동영상·학습 곡선·하이퍼파라미터를 자동 저장하여 실험 추적성을 확보함.

### [2025_강화학습실제_Atari game PONG_구현코드](2025_강화학습실제_Atari%20game%20PONG_구현코드)
강화학습실제 **기말 프로젝트** 의 구현 코드임. PyTorch로 PPO와 SAC를 직접 구현해 Atari Pong-v5 환경을 학습시킴. `AtariPreprocessing` + `FrameStack(4)` 위에서 16개 환경을 `SyncVectorEnv`로 병렬화하고, GAE / clipped surrogate / value clipping / advantage normalization / entropy bonus / lr decay까지 PPO 핵심 요소를 모두 갖춤. 베이스라인부터 최종 채택본까지 튜닝 흔적을 함께 보존함.

### [2025_강화학습실제_Atari game PONG_기말발표](2025_강화학습실제_Atari%20game%20PONG_기말발표)
위 PPO/SAC Pong 프로젝트의 기말 발표 자료(PDF)임.

---


## 사용 기술 요약

| 분류 | 사용 기술 |
|------|----------|
| 언어 | Python 3 |
| 머신러닝 | NumPy, pandas, scikit-learn, scipy |
| 딥러닝 | PyTorch, TensorFlow / Keras, dezero |
| 강화학습 | Gymnasium / OpenAI Gym, ALE (Atari) |
| 컴퓨터 비전 | OpenCV, scikit-image, Ultralytics YOLO |
| 시각화 / GUI | Matplotlib, TensorBoard, PyQt5, Tkinter |

---

## 저장소 구성 안내

본 저장소는 다수 프로젝트를 **git submodule** 로 관리함. 각 프로젝트의 최신 코드까지 함께 받으려면 다음과 같이 클론하면 됨.

```bash
git clone --recurse-submodules https://github.com/artelee/IndustrialAI.git
```

이미 클론한 경우에는 다음 명령으로 서브모듈을 받아올 수 있음.

```bash
git submodule update --init --recursive
```
