# reid-camera-calibration

**학습 없이 검색 단계에서 카메라별 특징 평균을 빼서 보행자 재식별(Re-ID)의 카메라 편향을 줄이고, 동일 도메인·교차 도메인에서 효과를 비교한다.**

[English README](README.md)

## 해결하는 문제

Re-ID 모델은 같은 카메라에서 찍힌 이미지에 공통된 카메라 고유 성분(조명, 시점, 배경, 색 특성)을 함께 담는다. 학습한 카메라에서는 이 성분이 대부분 학습으로 지워지지만, 학습하지 않은 카메라(교차 도메인)에서는 같은 카메라의 *다른* 사람이 다른 카메라의 *같은* 사람보다 더 가깝게 나올 수 있다. 이 저장소는 학습이나 라벨 없이 검색 단계의 간단한 보정으로 이 차이를 얼마나 줄일 수 있는지 측정한다.

## 방법

카메라 평균 차감 (Song et al., ICLR 2025), 특징 추출 후 적용:

```mermaid
flowchart LR
    A["이미지"] --> B["백본<br/>(CLIP-ReID / PersonViT)"]
    B --> C["f = N(원시 특징)"]
    C --> D["mu_c = 카메라 c의 f 평균<br/>(쿼리·갤러리 각각)"]
    D --> E["f_hat = N(f - mu_c)"]
    E --> F["코사인 순위<br/>Rank-1, mAP"]
    F --> G["쿼리 단위 대응 부트스트랩<br/>차이의 95% CI"]
```

| 단계 | 모듈 | 내용 |
|---|---|---|
| 1. 특징 추출 | `reid_calib.extract` | CLIP-ReID ViT-B/16 또는 PersonViT ViT-B/16, BNNeck 이전 전역 특징, 원시 float32, `.npz` 캐시 |
| 2. 보정 | `reid_calib.calibrate` | `f_hat = N(f - mu_c)`, `f = N(원시)`; `mu_c`는 카메라별 평균으로 쿼리 집합과 갤러리 집합에서 **각각** 계산; 카메라 번호만 사용 |
| 3. 평가 | `reid_calib.evaluate` | 표준 프로토콜: 쿼리마다 같은 인물·같은 카메라 갤러리 제외, Rank-1, mAP |
| 4. 통계 | `reid_calib.stats` | 쿼리별 AP / Rank-1 차이의 대응 부트스트랩 (1,000회, 시드 0), 95% CI, 0 미포함 시 `*` |

차감 직후(마지막 정규화 전) 카메라별 평균 특징은 정확히 0이 된다. `tests/test_calibrate.py`가 이 성질을 검사한다.

## 설치

```bash
git clone https://github.com/artelee/IndustrialAI.git
cd IndustrialAI/2026_석사논문_ReID_카메라특징보정
pip install -r requirements.txt          # numpy, pyyaml, pytest: 보정·평가·테스트에 충분
pip install -e .                         # 선택: 어디서나 reid_calib import
pytest                                   # CPU, 합성 데이터만 사용
```

특징 추출에는 PyTorch, torchvision, Pillow와 백본 원 저장소의 requirements가 추가로 필요하다 (`pip install -e ".[extract]"` 및 예: `pip install -r third_party/CLIP-ReID/requirements.txt`). 그림에는 matplotlib이 필요하다 (`pip install -e ".[plot]"`).

### 데이터·가중치 (링크만 안내, 이 저장소에는 포함하지 않음)

* **데이터셋**: [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html), DukeMTMC-reID (저자가 공개 중단; 사용 허가된 사본만 사용). 공식 출처에서 각 라이선스에 따라 받는다. `data/market1501/`, `data/dukemtmc-reid/` (각각 `query/`, `bounding_box_test/` 포함)에 두거나 `configs/{market,duke}.yaml` 수정 / `--data-root market <경로>` 사용.
* **CLIP-ReID** 코드와 ViT-B/16 가중치: <https://github.com/Syliz517/CLIP-ReID> → `third_party/CLIP-ReID`에 clone, 가중치는 `weights/clipreid/` (`configs/backbones/clipreid.yaml` 경로). 원 코드가 처음 실행 시 OpenAI CLIP ViT-B/16 가중치를 내려받는다.
* **PersonViT** 코드: <https://github.com/hustvl/PersonViT> → `third_party/PersonViT`에 clone; 사전학습 ViT-B/16 <https://huggingface.co/lakeAGI/PersonViT>, Re-ID 미세조정 가중치 <https://huggingface.co/lakeAGI/PersonViTReID> → `weights/personvit/` (`configs/backbones/personvit.yaml` 경로).

모델 코드는 사용자가 받은 원 저장소에서 import한다 (백본 설정의 `repo:` 또는 `--repo`). 이 저장소에 복사하지 않는다. CLIP-ReID 원 코드는 내부에서 모델을 CUDA로 옮기므로 CLIP-ReID 추출에는 GPU가 필요하다. PersonViT 추출은 `--device cpu`로도 동작한다(느림). 보정·평가·통계는 GPU나 torch가 필요 없다.

### 실행

```bash
# Duke로 학습한 CLIP-ReID를 Market에서 평가: Base 대 + 보정, 부트스트랩 CI.
# outputs/features/clip_duke_market_{query,gal}.npz가 있으면 사용하고, 없으면 추출 (GPU)
python scripts/run_eval.py --backbone clipreid --train duke --eval market --calib

# 기존 특징 캐시로 8개 설정 전체 (백본 2개 x 동일/교차 도메인) (CPU, 합계 1~2분)
python scripts/run_eval.py --backbone clipreid personvit --train market duke --eval market duke \
    --calib --features data/features

# 직접 실행한 결과로 막대그래프 생성
python scripts/plot_results.py --table outputs/results/summary.json --out outputs/calibration_map.png
```

출력 (`outputs/results/`, git 제외): `<백본>_<학습>_<평가>.json` (두 조건의 Rank-1·mAP, CI), `<백본>_<학습>_<평가>_per_query.npz` (쿼리별 AP·Rank-1), `summary.md`, `summary.json`.

특징 캐시 형식 (원시 특징, 분할마다 한 파일): `<접두어>_<학습>_<평가>_{query,gal}.npz`, 키 `feat (N, D)`, `pid`, `cam`, `name`; 접두어 `clip` (CLIP-ReID), `pvit` (PersonViT). 쿼리는 junk(pid -1)·distractor(pid 0) 제외, 갤러리는 junk만 제외. `reid-eval-bench`와 같은 형식이다.

## 폴더 구조

```
reid-camera-calibration/
├── configs/
│   ├── market.yaml, duke.yaml          # 데이터 경로, 분할 폴더, 파일명 파서, 학습 인물 수
│   └── backbones/{clipreid,personvit}.yaml  # 원 저장소 경로, 가중치, 입력 설정, 캐시 접두어
├── docs/
│   ├── figures/calibration_map.png     # 예시 그래프 (docs/results_table.json에서 생성)
│   └── results_table.json
├── scripts/
│   ├── run_eval.py                     # (필요 시) 추출 -> Base / + 보정 -> 부트스트랩 CI -> JSON / Markdown
│   └── plot_results.py
├── src/reid_calib/
│   ├── extract.py                      # 백본 로드 (torch·원 코드 지연 import), 특징 추출
│   ├── calibrate.py                    # 카메라별 평균 차감
│   ├── evaluate.py                     # Rank-1, AP / mAP (numpy만)
│   ├── stats.py                        # 대응 부트스트랩 CI (numpy만)
│   └── features.py                     # 특징 캐시 .npz 형식
├── tests/                              # pytest, 합성 데이터만
├── pyproject.toml, requirements.txt, LICENSE
└── outputs/                            # 결과·특징 캐시 (git 제외)
```

## 인용 / 라이선스 / 관련 저장소

* 카메라 평균 보정: Song et al. *Exploring the Camera Bias of Person Re-identification.* ICLR 2025.
* CLIP-ReID: S. Li, L. Sun, Q. Li. *CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels.* AAAI 2023. 코드: MIT License, (c) 2023 Syliz517.
* PersonViT: B. Hu et al. *PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification.* arXiv:2408.05398, 2024. 코드: Apache-2.0.
* Market-1501: L. Zheng et al. *Scalable Person Re-identification: A Benchmark.* ICCV 2015.
* DukeMTMC-reID: E. Ristani et al. ECCV Workshops 2016; Z. Zheng et al. ICCV 2017.

라이선스: [MIT](LICENSE), (c) 2026 ehlee. 백본은 각 원 라이선스를 따른다. 원 코드, 가중치, 이미지, 특징은 포함하지 않는다.

관련 저장소:
[2026_석사논문_ReID_평가벤치](../2026_석사논문_ReID_평가벤치/) ·
[2026_석사논문_ReID_카메라특징보정](../2026_석사논문_ReID_카메라특징보정/) ·
[2026_석사논문_보행방향분석](../2026_석사논문_보행방향분석/) ·
[2026_석사논문_재현가능한_Diffusion생성](../2026_석사논문_재현가능한_Diffusion생성/) ·
[2026_석사논문_카메라별_자세선정](../2026_석사논문_카메라별_자세선정/) ·
[2026_석사논문_MSMT17_변환기](../2026_석사논문_MSMT17_변환기/)
