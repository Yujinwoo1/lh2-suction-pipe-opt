# lh2-suction-pipe-opt
project

LH2 흡입배관 1D 솔버의 열침입·압력 열 성분을 근사하는 200개 파일럿 대리모델입니다.
Windows에서 작성한 Python 코드를 별도 Linux 머신의 CLI에서 실행할 수 있습니다. WSL, Windows 드라이브 마운트, GUI는 필요하지 않습니다.

## Linux에서 시작하기

Git과 Python 3.12 및 venv/pip가 설치되어 있어야 합니다. 저장소가 비공개라면 GitHub 접근 권한으로 먼저 인증하세요.

```bash
git clone https://github.com/Yujinwoo1/lh2-suction-pipe-opt.git
cd lh2-suction-pipe-opt
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements-tested.txt`는 Windows 파일럿 실행에서 확인한 버전 기록입니다. Linux에서도 같은 버전을 맞추려면 위 설치 명령의 파일명을 바꾸세요. Linux에서의 실제 실행 검증은 아직 하지 않았습니다.

## 데이터 준비

**데이터와 학습 모델은 Git에 포함하지 않습니다.** Clone만으로 코드가 준비되며, 학습에는 별도로 전달받은 원본 `dataset_ai.csv`가 필요합니다.

```bash
mkdir -p data
cp /path/to/dataset_ai.csv data/dataset_ai.csv
```

`/path/to/dataset_ai.csv`는 Linux에 확보한 실제 파일 경로로 바꾸세요. `cases_10k.csv`에는 정답 출력이 없으므로 학습에 사용할 수 없습니다. 테스트도 제공된 200개 파일럿 데이터를 요구합니다.

## CLI 실행 순서

```bash
# 입력·단위·누수 방지·외삽 차단 검증
python -m unittest discover -s tests -v

# 개발 자료에서 5종 비교, 튜닝, 학습 곡선 작성
python lh2_pipeline.py develop --learning-curves

# 모델 선정 완료 후 고정 테스트셋을 한 번 평가
python lh2_pipeline.py finalize

# 결과를 Markdown 보고서로 요약
python summarize_results.py
```

그래프는 비대화형 Matplotlib Agg 백엔드로 PNG에 저장하므로 화면 없는 서버에서도 실행할 수 있습니다. 결과는 `outputs/pilot/`에 저장합니다. 이미 완료한 개발 실험과 최종 테스트를 덮어쓰지 않습니다. 실행 오류를 확인하려면 종료 코드와 터미널 로그를 보세요.

빠른 동작 확인만 필요하면 별도 폴더에 `python lh2_pipeline.py develop --quick --out outputs/smoke`를 실행하세요. `--quick` 결과를 기본 실험과 섞지 마세요. 테스트 결과를 본 뒤 모델을 재선정하거나 튜닝하지 않습니다.

## 새 입력 예측

```bash
python predict.py /path/to/new_cases.csv --models outputs/pilot --output outputs/predictions.csv
```

필수 입력 열은 `D_mm`, `t_ins_mm`, `f_Hz`, `L_m`, `mdot_kg_h`, `k_MLI_W_mK`입니다. 지정 설계 범위를 벗어난 입력은 중단합니다. 제약 만족 여부는 대리모델의 예측이며 실제 솔버 검증을 대체하지 않습니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `lh2_pipeline.py` | 데이터 감사, 고정 분할, 모델 비교, 최종 평가 |
| `predict.py` | 저장된 선정 모델로 새 설계 조건 예측 |
| `summarize_results.py` | 기본 실험 결과를 `outputs/pilot/RESULTS.md`로 요약 |
| `tests/test_pipeline.py` | 파일럿 데이터 기반 검증 5개 |
| `LH2_pilot.ipynb` | 선택적으로 사용하는 설명·실행 노트북 |
| `make_notebook.py` | 노트북 재생성 |
| `requirements.txt` | 설치용 의존성 범위 |
| `requirements-tested.txt` | Windows 실행 당시 버전 기록 |
| `docs/PILOT.md` | 실험 설계, 평가 기준, 발표 시 해석과 한계 |

## 현재 범위와 한계

- 개발 160개 / 테스트 40개, seed 2026, 개발 CV 5-fold × 3회입니다.
- 선형회귀, 다항 Ridge, RF, GPR, MLP를 비교하며 두 출력을 각각 log10 공간에서 학습합니다.
- 튜닝 모델당 6개 후보를 사용한 파일럿입니다. 원 지시서의 50회 튜닝 전체를 수행한 것으로 표기하지 않습니다.
- 최적화, 신규 1D 솔버 재검증, 참고논문 ANN 재현, 열침입 물리 기준선은 아직 포함하지 않았습니다.
- 200개는 실험 측정값이 아닌 1D 솔버 출력입니다. 수력 근사오차를 AI 오차와 분리해야 합니다.
- 정확 GPR은 개발 자료 3,000개 초과 시 중단합니다. 1만 개 결과를 받으면 비교 설계를 확장해야 합니다.

세부 방법은 [파일럿 설명](docs/PILOT.md)을 참고하세요. 기존 Windows 전용 `run_local.ps1`과 로컬 Python 패키지는 이 저장소에 포함하지 않았습니다.
