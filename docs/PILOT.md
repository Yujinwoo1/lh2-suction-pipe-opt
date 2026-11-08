# LH2 흡입배관 대리모델: 200개 파일럿

이 프로젝트는 1D 솔버 출력 200개로 **재현 가능한 모델 비교와 최종 평가 절차**를 검증합니다. 실험 데이터에 대한 정확도, 최종 설계 안전성, 1만 개 학습 결과를 주장하지 않습니다.

## 먼저 열 파일

`LH2_pilot.ipynb`를 Jupyter/VS Code에서 열면 연구 목적, 실행 순서, 결과표 확인을 순서대로 볼 수 있습니다. 실제 계산은 `lh2_pipeline.py`에 모아 노트북과 명령줄이 동일한 코드를 실행하도록 했습니다.

## 실행

일반 Python 3.11 이상 환경:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python lh2_pipeline.py develop --learning-curves
.venv\Scripts\python lh2_pipeline.py finalize
```

검증한 버전을 그대로 설치하려면 `requirements.txt` 대신 `requirements-tested.txt`를 사용합니다. 노트북 UI는 Jupyter 또는 VS Code의 Python/Jupyter 확장이 별도로 필요하며 명령줄 실행에는 필요하지 않습니다.

데이터는 저장소에 포함하지 않습니다. 별도 전달받은 원본의 복사본을 `data/dataset_ai.csv`로 준비하세요. 원본을 수정하지 않습니다. `cases_10k.csv`는 정답 없는 입력 목록이므로 학습에 합치지 않습니다. 다른 파일은 `--data 경로`, 새 실험은 `--out outputs/새이름`으로 지정합니다. `--quick`은 실행 확인용 축소 실험이며 발표용 기본 설정이 아닙니다.

위 PowerShell 명령은 Windows용 참고입니다. 별도 Linux 머신에서는 저장소 루트 README의 CLI 명령을 사용하세요. 원래 Windows 로컬 환경의 `.local_deps`와 `run_local.ps1`은 저장소에 포함하지 않습니다.

새 설계 조건의 예측은 `python predict.py 입력.csv --output outputs/predictions.csv`로 실행합니다. 입력 CSV에는 여섯 원변수(D, 단열 두께, 주파수, 길이, 질량유량, MLI 열전도율)가 필요합니다. 지정 범위를 벗어나면 중단하며, 압력 제약 판정은 확정된 안전성 판단이 아닌 파일럿 예측입니다.

## 사전에 고정한 연구 설계

1. 입력: `D_mm`, `t_ins_mm`, `f_Hz`, `L_m`, `mdot_kg_h`, `log10_k_MLI` 6개만 사용합니다. ID, 압력 결과, 기화 라벨, 열침입 정답을 입력에 넣지 않습니다.
2. 출력: `Q_leak_W`, `dp_thermal_Pa`를 각각 독립적으로 모델링합니다. `TransformedTargetRegressor`에서 log10 변환하고, 예측 시 원단위로 역변환합니다. 역변환값은 log 공간의 중심 예측이며 산술평균의 불편 추정이라고 주장하지 않습니다.
3. 시드 2026, 기화 라벨로 층화하여 개발 160개 / 최종 테스트 40개를 고정합니다. 라벨은 분할용이며 입력 특성이 아닙니다. 데이터 감사는 전체의 스키마와 물리 일관성을 확인하는 절차이고 모델 선정에는 개발 자료만 씁니다.
4. 개발 데이터에서 동일한 5-fold × 3회 반복 분할을 모든 모델에 적용합니다. 스케일러와 다항 변환은 각 fold 학습 부분에서만 fit합니다.
5. 선형, 다항 Ridge, RF, GPR, MLP 5종을 비교합니다. 튜닝 모델당 6개 후보를 같은 seed로 사전 추출하고 같은 CV에서 평가합니다. 선형회귀는 튜닝 변수가 없습니다. 이는 동일 후보 수 비교이며 동일 실행시간 비교는 아닙니다. 파일럿 계산 예산으로 6개를 선택했으며, 지시서의 50회 튜닝을 완료했다고 표시하지 않습니다.
6. 각 출력에서 평균 CV log10 RMSE가 가장 낮은 모델을 선택합니다. 원단위 R²/RMSE/MAPE와 log 지표, fold 표준편차, 학습시간, 1만 행 예측시간도 저장합니다. 반복 fold는 독립 표본이 아니므로 표준편차를 신뢰구간으로 해석하지 않습니다.
7. **튜닝에 사용한 CV 점수는 선택 편향을 포함합니다.** 별도 nested CV를 수행한 결과로 표기하지 않습니다. 일반화 성능의 최종 근거는 마지막에 한 번 여는 고정 테스트셋입니다. 40개의 결과이므로 작은 차이로 모델 우열을 단정하지 않습니다.
8. 학습 곡선은 개발 데이터 안에서 별도 5-fold로 계산합니다. 25/50/100/128개를 사용하며 테스트셋은 쓰지 않습니다. 개발 자료에서 정한 하이퍼파라미터를 고정하므로 설명용 학습 곡선입니다. 각 크기에서 5개 fold의 분산을 보여주며, 독립 재표집 5회 실험과는 다릅니다.
9. 최종 테스트는 선택 모델과 사전에 정한 비교 대상 MLP만 보고합니다. 결과를 보고 재튜닝하지 않습니다. 재실행 방지 파일을 저장하지만, 다른 폴더를 만들어 반복 평가하는 것까지 기술적으로 막는 것은 아닙니다.

## 물리식과 압력 제약

강체식에서 D는 mm→m, 질량유량은 kg/h→kg/s로 변환합니다. 전체 압력 예측은 `AI(dp_thermal) + 1.8 * dp_rigid`입니다. 수력 근사식 자체의 오차는 **실제 열 성분**을 사용한 별도 `physics_audit.csv`로 분리합니다. case 110에서 AI 오차 없이도 48,300 Pa 판정이 바뀌므로, 이 오차를 AI 탓으로 돌리지 않습니다.

최종 테스트에서는 전체 압력 오차, 38,300–58,300 Pa 경계 구간 오차와 표본 수, 위험 조건을 만족으로 예측한 수(false safe), 그 반대 수를 저장합니다. `vaporized`는 제공된 압력 임계값으로 정의된 라벨이며 관측된 기화 실험값이 아닙니다.

열침입 물리 기준선과 [8,9] 재현 ANN은 아직 구현하지 않았습니다. NSGA-II 최적화와 솔버 결과 비교 기능은 아래 명령으로 제공합니다. 실제 솔버 재검증은 결과를 받은 뒤 수행합니다.

## 대리모델 최적화 (P0-1)

`python optimize.py run --models outputs/pilot`은 기존 모델과 테스트셋을 변경하지 않고 `outputs/opt_pilot/`에 결과를 생성합니다. `pymoo` 설치가 필요합니다. 시나리오는 `configs/scenarios.json`으로 관리하며 범위 밖 조건은 거부합니다. 설계변수는 D=12–50 mm, 단열=2–25 mm, 최소화 목적은 Q_leak, dp_thermal+1.8×dp_rigid, D_o이며 압력 한계는 48,300 Pa입니다.

NSGA-II(100개체, 200세대, seed 2026)와 200×200 격자를 비교합니다. 격자는 대리모델 기준의 정답지이며 솔버 정답이 아닙니다. HV는 격자 Pareto ideal/nadir로 두 전선을 공통 정규화하고 기준점 (1.1, 1.1, 1.1)을 사용합니다. `hv_summary.csv`에 정규화 상수와 비율을 기록합니다.

합친 비지배 전선에서 A=Q_leak 최소, B=각 목적을 [0,1] 정규화한 뒤 ideal(0,0,0)까지 유클리드 거리 최소, C=D_o 최소를 고릅니다. 추가 6개는 정규화 목적공간의 farthest-point sampling이며 거리 0.01 이내 후보는 높은 dp_thermal GPR σ, 38.3–48.3 kPa 구간과의 가까운 거리를 우선합니다. 전선이 작으면 추가 개수도 줄어듭니다. A/B/C가 동일 설계일 수 있습니다. GPR σ는 역변환한 Pa가 아닌 log10 표준편차이며 비GPR은 NaN입니다.

전선 CSV, `T3_candidates.csv`, 입력 6열의 `verification_inputs.csv`, 예측·ID·n*를 포함한 `verification_points.csv`, 300 dpi 흰 배경 `F4.png`를 생성합니다. n*=1040/(4Lf)가 20 미만이고 정수와 차이가 0.1 미만이면 경고합니다. 이 규칙은 검증된 공진 판정이 아닙니다.

`python optimize.py compare --points outputs/opt_pilot/verification_points.csv --solver solver_results.csv`는 ID가 일대일 대응하는 솔버의 `point_id,Q_leak_W,dp_design_Pa`를 받아 T4를 생성합니다. 오차는 솔버 값을 분모로 계산하고 제약 판정 일치 여부도 기록합니다. 가짜 솔버 데이터는 단위 테스트 전용이며 연구 결과로 사용하지 않습니다.

## P0-1 실행 확인 (2026-11-08)

`.venv`의 pymoo 0.6.2에서 기본 명령으로 NSGA-II(개체 100, 200세대, seed 2026)와 시나리오별 40,000개 격자 평가를 완료했습니다. 모델 재학습과 `finalize`는 실행하지 않았습니다.

| 시나리오 | 격자 Pareto 수 | NSGA-II Pareto 수 | NSGA-II HV / 격자 HV | 재검증 후보 수 |
|---|---:|---:|---:|---:|
| S1 | 35,800 | 100 | 0.935528 | 9 |
| S2 | 29,252 | 100 | 0.921909 | 9 |
| S3 | 29,595 | 100 | 0.948267 | 9 |

총 27개 후보, 입력 전용 CSV, T3, 전선·HV 표와 F4를 `outputs/opt_pilot/`에 생성했습니다. 이 수치는 대리모델 평가이며 실제 솔버 재검증 결과는 없습니다. T4 비교는 합성 데이터 단위 테스트에서만 확인했습니다. `.venv/bin/python -m unittest discover -s tests -v` 결과는 13개 통과, 실패 0개, 건너뜀 0개입니다. P0-2 이후 항목은 수행하지 않았습니다.

## 결과 읽기

| 결과 | 의미 |
|---|---|
| `data_audit.json`, `physics_audit.csv` | 데이터·물리 일관성과 수력 근사오차 |
| `split.csv`, `cv_indices.npz` | 고정 분할의 재현 근거 |
| `manifest.json` | 시드, 버전, 데이터 해시, 설정, 선정 모델, 수렴 경고 |
| `cv_summary.csv`, `cv_fold_metrics.csv` | 개발 CV 비교 및 fold별 수치 |
| `*__search.csv` | 모든 튜닝 후보의 기록 |
| `learning_curve.csv`, `*__learning.png` | 개발 자료 내부 학습 곡선 |
| `*__mlp_history.csv/png` | 학습 손실과 내부 검증 R². 검증 손실로 잘못 표기하지 않음 |
| `final_test_metrics.csv`, `*__parity.png` | 선정 완료 후 테스트 결과 |
| `final_pressure_metrics.json` | 하이브리드 압력 오차와 제약 판정 |
| `*.joblib` | 개발 데이터로 학습한 모델. 신뢰하는 파일만 로드 |

GPR의 커널 경계 도달, MLP의 반복 한도 등 경고는 숨기지 않고 manifest에 남깁니다. 테스트 점수를 보고 경고를 수정하거나 모델을 다시 고르지 않습니다. MLP 조기종료 검증은 각 학습 fold 내부에서만 수행하며, scikit-learn MLP는 마지막이 아닌 최고 내부 검증 점수의 가중치를 복원합니다.

## 교수님께 설명할 핵심

“모델의 복잡성보다 동일한 분할과 평가 기준을 우선했습니다. 열 성분을 log 공간에서 학습하고 수력 성분은 물리식으로 계산했습니다. CV는 모델 선택에, 별도 테스트는 최종 평가에 사용했습니다. 수력 근사오차와 대리모델 오차를 분리했고, 전체 점수뿐 아니라 압력 제약 경계의 오판정도 확인했습니다. 현재 결과는 200개 파일럿이므로 최종 최적설계의 검증으로 해석하지 않습니다.”

1만 개 결과를 받으면 데이터 버전과 분할을 새로 고정해야 합니다. 정확 GPR은 개발 표본 3,000개 초과 시 코드가 중단됩니다. 동일 부분표본 비교와 전체 데이터 활용 비교를 구분하도록 실험 설계를 확장한 후 진행합니다.
