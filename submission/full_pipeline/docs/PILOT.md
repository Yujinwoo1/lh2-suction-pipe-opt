# LH2 흡입배관 대리모델: 200개 파일럿

파일럿 실험과 이후 실행 준비를 날짜순으로 남긴 기록임. 각 절의 수치는 당시 설정에 해당하며 제출본의 최종 결과와 구분해 읽어야 함.

이 프로젝트는 1D 솔버 출력 200개로 재현 가능한 모델 비교와 최종 평가 절차를 검증함. 실험 데이터에 대한 정확도, 최종 설계 안전성, 1만 개 학습 결과를 주장하지 않음.

## 먼저 열 파일

`LH2_pilot.ipynb`를 Jupyter/VS Code에서 열면 연구 목적, 실행 순서, 결과표 확인을 순서대로 볼 수 있음. 실제 계산은 `lh2_pipeline.py`에 모아 노트북과 명령줄이 동일한 코드를 실행하도록 했음.

## 실행

일반 Python 3.11 이상 환경:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python lh2_pipeline.py develop --learning-curves
.venv\Scripts\python lh2_pipeline.py finalize
```

검증한 버전을 그대로 설치하려면 `requirements.txt` 대신 `requirements-tested.txt`를 사용함. 노트북 UI는 Jupyter 또는 VS Code의 Python/Jupyter 확장이 별도로 필요하며 명령줄 실행에는 필요하지 않음.

데이터는 저장소에 포함하지 않음. 별도 전달받은 원본의 복사본을 `data/dataset_ai.csv`로 준비함. 원본을 수정하지 않음. `cases_10k.csv`는 정답 없는 입력 목록이므로 학습에 합치지 않음. 다른 파일은 `--data 경로`, 새 실험은 `--out outputs/새이름`으로 지정함. `--quick`은 실행 확인용 축소 실험이며 발표용 기본 설정이 아님.

위 PowerShell 명령은 Windows용 참고임. 별도 Linux 머신에서는 저장소 루트 README의 CLI 명령을 사용함. 원래 Windows 로컬 환경의 `.local_deps`와 `run_local.ps1`은 저장소에 포함하지 않음.

새 설계 조건의 예측은 `python predict.py 입력.csv --output outputs/predictions.csv`로 실행함. 입력 CSV에는 여섯 원변수(D, 단열 두께, 주파수, 길이, 질량유량, MLI 열전도율)가 필요함. 지정 범위를 벗어나면 중단하며, 압력 제약 판정은 확정된 안전성 판단이 아닌 파일럿 예측임.

## 사전에 고정한 연구 설계

1. 입력: `D_mm`, `t_ins_mm`, `f_Hz`, `L_m`, `mdot_kg_h`, `log10_k_MLI` 6개만 사용함. ID, 압력 결과, 기화 라벨, 열침입 정답을 입력에 넣지 않음.
2. 출력: `Q_leak_W`, `dp_thermal_Pa`를 각각 독립적으로 모델링함. `TransformedTargetRegressor`에서 log10 변환하고, 예측 시 원단위로 역변환함. 역변환값은 log 공간의 중심 예측이며 산술평균의 불편 추정이라고 주장하지 않음. P0-2의 물리식 기준선은 학습 없이 원단위 W로 직접 계산함.
3. 시드 2026, 기화 라벨로 층화하여 개발 160개 / 최종 테스트 40개를 고정함. 라벨은 분할용이며 입력 특성이 아님. 데이터 감사는 전체의 스키마와 물리 일관성을 확인하는 절차이고 모델 선정에는 개발 자료만 사용함.
4. 개발 데이터에서 동일한 5-fold × 3회 반복 분할을 모든 모델에 적용함. 스케일러와 다항 변환은 각 fold 학습 부분에서만 fit함.
5. 선형, 다항 Ridge, RF, GPR, MLP 5종을 비교함. 튜닝 모델당 6개 후보를 같은 seed로 사전 추출하고 같은 CV에서 평가함. 선형회귀는 튜닝 변수가 없음. 이는 동일 후보 수 비교이며 동일 실행시간 비교는 아님. 파일럿 계산 예산으로 6개를 선택했으며, 지시서의 50회 튜닝을 완료했다고 표시하지 않음.
6. 각 출력에서 평균 CV log10 RMSE가 가장 낮은 모델을 선택함. 원단위 R²/RMSE/MAPE와 log 지표, fold 표준편차, 학습시간, 1만 행 예측시간도 저장함. 반복 fold는 독립 표본이 아니므로 표준편차를 신뢰구간으로 해석하지 않음.
7. 튜닝에 사용한 CV 점수는 선택 편향을 포함함. 별도 nested CV를 수행한 결과로 표기하지 않음. 일반화 성능의 최종 근거는 마지막에 한 번 여는 고정 테스트셋임. 40개의 결과이므로 작은 차이로 모델 우열을 단정하지 않음.
8. 기존 pilot의 학습 곡선은 개발 데이터 안에서 별도 5-fold로 계산했음. 25/50/100/128개를 사용했고 테스트셋은 쓰지 않았음. P1-1 이후 새 실행은 아래의 고정 검증 분할·5회 재표집 방식으로 바뀜. 개발 자료에서 정한 하이퍼파라미터를 고정하므로 설명용 학습 곡선임. 각 크기에서 5개 fold의 분산을 보여주며, 독립 재표집 5회 실험과는 다름.
9. 최종 테스트는 선택 모델과 사전에 정한 비교 대상 MLP만 보고함. 결과를 보고 재튜닝하지 않음. 재실행 방지 파일을 저장하지만, 다른 폴더를 만들어 반복 평가하는 것까지 기술적으로 막는 것은 아님.

## 물리식과 압력 제약

강체식에서 D는 mm→m, 질량유량은 kg/h→kg/s로 변환함. 전체 압력 예측은 `AI(dp_thermal) + 1.8 * dp_rigid`임. 수력 근사식 자체의 오차는 실제 열 성분을 사용한 별도 `physics_audit.csv`로 분리함. case 110에서 AI 오차 없이도 48,300 Pa 판정이 바뀌므로, 이 오차를 AI 탓으로 돌리지 않음.

최종 테스트에서는 전체 압력 오차, 38,300–58,300 Pa 경계 구간 오차와 표본 수, 위험 조건을 만족으로 예측한 수(false safe), 그 반대 수를 저장함. `vaporized`는 제공된 압력 임계값으로 정의된 라벨이며 관측된 기화 실험값이 아님.

열침입 물리 기준선은 P0-2에서 추가했으며, [8,9] 재현 ANN은 아직 구현하지 않았음. NSGA-II 최적화와 솔버 결과 비교 기능은 아래 명령으로 제공함. 실제 솔버 재검증은 결과를 받은 뒤 수행함.

## 대리모델 최적화 (P0-1)

`python optimize.py run --models outputs/pilot`은 기존 모델과 테스트셋을 변경하지 않고 `outputs/opt_pilot/`에 결과를 생성함. `pymoo` 설치가 필요함. 시나리오는 `configs/scenarios.json`으로 관리하며 범위 밖 조건은 거부함. 설계변수는 D=12–50 mm, 단열=2–25 mm, 최소화 목적은 Q_leak, dp_thermal+1.8×dp_rigid, D_o이며 압력 한계는 48,300 Pa임.

NSGA-II(100개체, 200세대, seed 2026)와 200×200 격자를 비교함. 격자는 대리모델 기준의 정답지이며 솔버 정답이 아님. HV는 격자 Pareto ideal/nadir로 두 전선을 공통 정규화하고 기준점 (1.1, 1.1, 1.1)을 사용함. `hv_summary.csv`에 정규화 상수와 비율을 기록함.

합친 비지배 전선에서 A=Q_leak 최소, B=각 목적을 [0,1] 정규화한 뒤 ideal(0,0,0)까지 유클리드 거리 최소, C=D_o 최소를 선택함. 추가 6개는 정규화 목적공간의 farthest-point sampling이며 거리 0.01 이내 후보는 높은 dp_thermal GPR σ, 38.3–48.3 kPa 구간과의 가까운 거리를 우선함. 전선이 작으면 추가 개수도 줄어듦. A/B/C가 동일 설계일 수 있음. GPR σ는 역변환한 Pa가 아닌 log10 표준편차이며 비GPR은 NaN임.

전선 CSV, `T3_candidates.csv`, 입력 6열의 `verification_inputs.csv`, 예측·ID·n*를 포함한 `verification_points.csv`, 300 dpi 흰 배경 `F4.png`를 생성함. n*=1040/(4Lf)가 20 미만이고 정수와 차이가 0.1 미만이면 경고함. 이 규칙은 검증된 공진 판정이 아님.

`python optimize.py compare --points outputs/opt_pilot/verification_points.csv --solver solver_results.csv`는 ID가 일대일 대응하는 솔버의 `point_id,Q_leak_W,dp_design_Pa`를 받아 T4를 생성함. 오차는 솔버 값을 분모로 계산하고 제약 판정 일치 여부도 기록함. 가짜 솔버 데이터는 단위 테스트 전용이며 연구 결과로 사용하지 않음.

## P0-1 실행 확인 (2026-11-08)

`.venv`의 pymoo 0.6.2에서 기본 명령으로 NSGA-II(개체 100, 200세대, seed 2026)와 시나리오별 40,000개 격자 평가를 완료했음. 모델 재학습과 `finalize`는 실행하지 않았음.

| 시나리오 | 격자 Pareto 수 | NSGA-II Pareto 수 | NSGA-II HV / 격자 HV | 재검증 후보 수 |
|---|---:|---:|---:|---:|
| S1 | 35,800 | 100 | 0.935528 | 9 |
| S2 | 29,252 | 100 | 0.921909 | 9 |
| S3 | 29,595 | 100 | 0.948267 | 9 |

총 27개 후보, 입력 전용 CSV, T3, 전선·HV 표와 F4를 `outputs/opt_pilot/`에 생성했음. 이 수치는 대리모델 평가이며 실제 솔버 재검증 결과는 없음. T4 비교는 합성 데이터 단위 테스트에서만 확인했음. `.venv/bin/python -m unittest discover -s tests -v` 결과는 13개 통과, 실패 0개, 건너뜀 0개임. P0-2 이후 항목은 수행하지 않았음.

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

GPR의 커널 경계 도달, MLP의 반복 한도 등 경고는 숨기지 않고 manifest에 남김. 테스트 점수를 보고 경고를 수정하거나 모델을 다시 고르지 않음. MLP 조기종료 검증은 각 학습 fold 내부에서만 수행하며, scikit-learn MLP는 마지막이 아닌 최고 내부 검증 점수의 가중치를 복원함.

## 교수님께 설명할 핵심

“모델의 복잡성보다 동일한 분할과 평가 기준을 우선했음. 열 성분을 log 공간에서 학습하고 수력 성분은 물리식으로 계산했음. CV는 모델 선택에, 별도 테스트는 최종 평가에 사용했음. 수력 근사오차와 대리모델 오차를 분리했고, 전체 점수뿐 아니라 압력 제약 경계의 오판정도 확인했음. 현재 결과는 200개 파일럿이므로 최종 최적설계의 검증으로 해석하지 않음.”

1만 개 결과를 받으면 데이터 버전과 분할을 새로 고정해야 함. P1-1 이후 GPR은 각 학습 fold와 최종 개발 학습에서 최대 3,000개만 사용함. n³ 비용을 위한 seed 2026 부분표본이며, 다른 모델이 해당 학습 자료 전체를 사용하는 점을 비교 시 명시함.

## 열침입 물리식 기준선 (P0-2)

`physics_baseline.py`는 `docs/codex_tasks.md` P0-2의 전도·대류 저항식을 그대로 사용함. 벽 두께 0.0015 m, h=5 W/m²K, 온도 300 K·23.6 K는 고정이며 벽 저항은 무시함. 학습 파라미터와 튜닝은 없음. `fit`은 입력이나 정답으로 상수를 변경하지 않음. Q_leak에만 추가하며 dp_thermal의 모델 구성은 그대로 유지함.

`.venv/bin/python lh2_pipeline.py develop --out outputs/pilot_v2`로 새 실험을 실행함. 기존 pilot과 같은 seed 2026의 개발 160개/최종 테스트 40개 분할을 유지하고, 개발 자료 안의 공통 5-fold × 3회 검증 fold에서 6개 모델을 비교함. `cv_summary.csv`는 fold별 지표의 평균과 표준편차이며, 최종 테스트 성능표가 아님. `finalize`는 실행하지 않음. 튜닝 모델당 후보는 기존과 같은 6개이고 물리식은 0개임. 고정식의 1회 설정 평가 기록은 `Q_leak_W__physics_baseline__search.csv`에 남음.

전체 200개의 물리식 절대 상대오차는 평균 0.059998780%, 최대 0.397697538%로, 요청한 평균 허용 구간 0.05–0.07%를 통과했음. 이는 데이터 감사 수치이며 CV 모델 비교와 구별함. 감사 파일 `data_audit.json`에 `physics_baseline_mean_relative_error_pct`, `physics_baseline_max_relative_error_pct`로 저장함.

이 파일럿의 발표 메시지는 “Q_leak는 물리식으로 충분하다 → AI의 실질 가치는 dp_thermal에 있다”임. 전체 자료 감사와 개발 CV는 최종 홀드아웃 또는 추가 솔버 검증을 대신하지 않음.


최종 실행 확인: 기존 pilot의 34개 파일 해시, 개발/최종 테스트 분할 및 15개 CV fold 인덱스가 모두 보존됐음. 최종 선정은 Q_leak=physics_baseline, dp_thermal=gpr임. `finalize`는 실행하지 않았음. 저장 모델의 모듈 분리 전후로 개발 실행을 총 2회 수행했고, 두 실행의 CV 지표는 일치했음. 각 실행에서 튜닝 모델당 동일한 6개 후보를 평가했으며 물리식 튜닝은 0회임. 첫 실행은 `outputs/pilot_v2_module_check`에 보존했음.

| 모델 | CV R² 평균 | CV RMSE 평균 [W] | CV MAPE 평균 [%] | CV log10 RMSE 평균 |
|---|---:|---:|---:|---:|
| physics_baseline | 0.999991410 | 0.043122867 | 0.058345393 | 0.000411382 |
| gpr | 0.999784025 | 0.189049411 | 0.758381201 | 0.005374708 |
| linear | 0.936874598 | 3.380770758 | 14.978113305 | 0.088075729 |
| polynomial_ridge | 0.988021955 | 1.462222738 | 6.622317473 | 0.037926384 |
| random_forest | 0.768221039 | 7.050524268 | 35.438644576 | 0.183060777 |
| mlp | 0.900256560 | 4.129202870 | 17.917675262 | 0.103286729 |

최종 실행의 중복 제거 경고 기록은 43개이며 `manifest.json`에 보존했음. 이는 경고 발생 총횟수가 아닌 고유 문구·모델/fold 조합 수임. 주로 GPR 길이척도 상한 도달이며 경고에 따라 상수나 하이퍼파라미터를 변경하지 않았음. 전체 unittest는 16개 통과, 실패 0개, 건너뜀 0개임.

## 물리 파생 특성 실험 (pilot_v3)

고정 P0-2 물리식으로 구한 Q_phys를 이용해 `log10(Q_phys/mdot_kg_h)`를 추가함. 원본 FEATURES는 6개로 유지하며 `PhysicsFeatureTransformer`가 각 학습 모델의 Pipeline 안에서 7번째 특성을 만듦. 정답 열을 참조하지 않고, 외부에서 같은 이름의 파생값을 넣어도 원본 입력으로 다시 계산함. Q_phys의 상수와 모델 튜닝 예산은 바꾸지 않음. 고정 physics_baseline에는 파생 특성을 추가하지 않음.

실행 명령은 `.venv/bin/python lh2_pipeline.py develop --physics-feature --out outputs/pilot_v3`임. 비교 대상은 physics-feature가 없는 `outputs/pilot_v2`의 dp_thermal 5개 모델임. 동일한 개발 160개/최종 홀드아웃 40개 분할과 동일한 15개 CV 검증 fold를 확인한 뒤 비교함. 각 튜닝 모델은 기존과 같은 6개 후보이며 최종 테스트는 평가하지 않음. CV는 모델 선정에 사용한 점수로 선택 편향을 포함함.

사용자의 기대값(약 0.015)에 대한 점검 구간을 실행 전에 GPR dp_thermal 평균 CV log10 RMSE 0.010–0.020으로 정했음. 구간 밖이면 추가 튜닝이나 최적화로 진행하지 않고 보고함. 기대값을 맞추기 위한 하이퍼파라미터 변경은 하지 않음.


개발 실행을 1회 완료했으며, 데이터 해시·분할·15개 CV 인덱스가 pilot_v2와 모두 같았음. 튜닝 모델당 후보는 6개이며 추가 튜닝은 하지 않았음. 아래 값은 공통 CV 검증 fold별 log10 RMSE의 평균임. 전체 원단위·log 지표의 평균과 표준편차는 `outputs/pilot_v3/dp_thermal_comparison.csv`에 나란히 저장했음.

| dp_thermal 모델 | 기존 6입력 | 물리 특성 추가 | 감소율 [%] |
|---|---:|---:|---:|
| linear | 0.143353 | 0.124368 | 13.24 |
| polynomial_ridge | 0.065607 | 0.048764 | 25.67 |
| random_forest | 0.163926 | 0.104614 | 36.18 |
| gpr | 0.034960 | 0.014549 | 58.38 |
| mlp | 0.157024 | 0.131391 | 16.32 |

GPR은 0.014549로 사전 점검 구간을 통과했음. 두 출력 모두 GPR이 선정됐음. manifest에는 고유 경고 기록 154개를 보존했음. 최종 홀드아웃 평가와 finalize는 수행하지 않았음.


`optimize.py run --models outputs/pilot_v3`가 NSGA-II(개체 100, 200세대, seed 2026)와 200×200 격자로 완료돼 27개 재검증 후보를 생성했음. 저장된 v3 모델의 일반 예측, 최적화 캐시 예측, 최적화 출력값 및 GPR sigma도 일치했음. 기존 pilot/pilot_v2/opt_pilot의 79개 파일 해시가 보존됐음.

기존 `outputs/opt_pilot` 대비 A/B/C는 같은 역할로 대응시켰음. 추가 6개는 시나리오별로 (D/38, t_ins/23) 공간의 유클리드 거리 합이 최소가 되는 일대일 대응을 사용했음. 아래는 시나리오별 9개 대응점의 절대 설계변수 차이임. 추가점은 동일한 물리적 후보의 추적이 아닌 집합 간 기하학적 대응임. Q_leak 모델도 원래 pilot과 달라졌으므로 차이를 dp_thermal 특성 하나의 순수 효과로 해석하지 않음.

| 시나리오 | 평균 ΔD [mm] | 최대 ΔD [mm] | 평균 Δt_ins [mm] | 최대 Δt_ins [mm] |
|---|---:|---:|---:|---:|
| S1 | 0.260318 | 0.954774 | 0.025684 | 0.231156 |
| S2 | 0.103561 | 0.381910 | 0.218321 | 0.693467 |
| S3 | 0.173103 | 0.954774 | 0.077052 | 0.693467 |

상세 대응과 부호 있는 차이는 `outputs/opt_pilot_v3/verification_design_changes.csv`, 요약은 `verification_design_change_summary.csv`에 저장했음. 대리모델 재검증 후보이며 실제 솔버 검증 결과는 아님. 전체 unittest는 19개 통과, 실패 0개, 건너뜀 0개임. 후속 작업으로 넘어가지 않고 여기서 마무리했음.

## 1만 개 대응 (P1-1)

합본이 없는 상태에서 `configs/full10k.json`을 2026-11-08에 사전 확정했음. 물리 특성 사용의 근거는 pilot_v3 개발 CV에서 dp_thermal 전 모델이 개선된 점이며, 파일럿 테스트 결과는 근거에 포함하지 않음. seed 2026, 층화 테스트 20%, GPR 최대 3,000개, 후보 예산 6, CV 5-fold × 3회, n_jobs=1임. 원본 설정 전체와 quick/CLI를 반영한 유효 설정을 manifest에 함께 기록함. 데이터 원본 SHA-256은 failed 행까지 포함한 파일 전체의 해시임.

GPR은 `SubsampledTargetRegressor`에서 전처리 전에 각 학습 자료만 부분표본 추출함. CV 검증 자료는 줄이지 않음. 모든 튜닝 후보의 공통 fold 부분표본과 최종 개발 부분표본의 case_id를 `gpr_subsamples.csv`에 기록함. GPR은 n³ 계산량 때문에 부분표본을 쓰며 다른 모델은 각 학습 자료 전체를 사용함. 최종 저장 모델의 regressor_는 기존과 같은 Pipeline이므로 예측과 최적화 및 log10 sigma 추출이 호환됩니다.

후보 수를 유한 탐색 공간보다 크게 요청하면 같은 범위의 alpha/학습률에는 log-uniform, RF 나무 수에는 정수 분포를 사용해 요청 수만큼 서로 다른 후보를 추출함. 이 확장 규칙도 합본 실행 전에 정한 규칙이며, 파일럿 결과를 보고 탐색 범위를 바꾸지 않음. 실제 후보는 모델별 search CSV에 전부 저장함. 예산 50의 고유 후보 추출은 테스트했으나 실제 모델을 50회 튜닝했다고 주장하지 않음.

status가 있으면 converged만 남기고 데이터 검증을 수행함. failed 행의 결측 정답은 회귀에 들어가지 않음. 원본 수·제외 수·제외 case_id는 data_audit.json에 기록함. 혼합 case_id에 연속성 가정은 없음. 원본 8,000–12,000행에서 남은 자료의 기화 비율이 파일럿 6%에서 3%p 넘게 달라지면 경고만 출력함.

학습 곡선은 개발 자료의 20%를 seed 2026 층화 검증 분할로 고정하고, 나머지 학습 풀에서 크기별 5회 독립 추출함. 모든 무작위 생성기의 seed는 2026이며 반복 추출은 같은 생성기의 연속 draw임. n은 25, 50, 100, 200, 400, 800, 1600, 3200, 개발 전체 중 개발 크기 이하이며, 학습 풀 전체 크기도 추가함. 검증 행을 제외하고 확보할 수 없는 n과 GPR 한도 초과 n은 표에 생략 사유를 남김. 모델 선정에 쓴 CV에서 하이퍼파라미터를 고정한 설명용 곡선으로, 독립적인 모델 선정 성능은 아님.

`--estimate-time`은 모델·출력별 후보 하나를 개발 fold 하나에서만 학습함. 탐색은 지정한 n_jobs, refit·보고 CV·학습 곡선은 순차 실행인 점을 추정식에 반영함. 학습 행 수에 따른 외삽은 GPR 3차, 나머지 1차로 근사하며, 1만 행 예측 벤치마크도 포함함. 후보별 비용, 메모리 경합, 그림·파일 비용은 정확히 예측하지 못함. 출력 폴더를 만들거나 저장 모델을 바꾸지 않고 종료함. 이번 파일럿 추정치는 실제 1만 개 실행시간이 아님.


P1-1 확인 명령 (실제 합본이 아닌 파일럿 200개):

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python lh2_pipeline.py develop --config configs/full10k.json --data data/dataset_ai.csv --out outputs/smoke_full10k --quick --gpr-max-rows 100 --learning-curves --estimate-time
.venv/bin/python lh2_pipeline.py develop --config configs/full10k.json --data data/dataset_ai.csv --out outputs/smoke_full10k --quick --gpr-max-rows 100 --learning-curves
```

모델·출력별 1-fold/1-candidate 측정으로 산출한 예상 시간은 아래 표에 정리함. 표의 시간은 최종 실제 소요시간이 아니라 검색·refit·보고 CV·학습 곡선·벤치마크를 합산한 추정임.

| 모델 | Q_leak 1-fold fit [s] | dp_thermal 1-fold fit [s] | Q_leak develop 예상 [s] | dp_thermal develop 예상 [s] |
|---|---:|---:|---:|---:|
| linear | 0.006 | 0.005 | 0.243 | 0.228 |
| polynomial_ridge | 0.007 | 0.007 | 0.346 | 0.330 |
| random_forest | 0.143 | 0.143 | 4.030 | 4.044 |
| gpr | 0.185 | 0.216 | 3.144 | 3.646 |
| mlp | 0.076 | 0.479 | 2.102 | 12.030 |
| physics_baseline | 0.000 | 해당 없음 | 0.016 | 해당 없음 |

추정 합계는 30.2초였으며, 경고를 포함한 전체 출력은 `outputs/smoke_full10k/runtime_estimate.txt`에 보존했음. 1만 개 실행시간으로 외삽해 주장하지 않음. 추정 실행은 출력 폴더 생성 없이 종료한 것을 확인했고, 그 뒤 smoke 개발 실행은 정확히 1회 수행했음.

smoke 유효 설정은 physics_feature=true, seed 2026, 층화 테스트 20%, gpr_max_rows=100, budget=2, CV 3-fold × 1회, n_jobs=1임. 원본 full10k 설정의 gpr_max_rows=3000, budget=6, CV 5-fold × 3회는 그대로 manifest에 보존했음. 각 튜닝 모델은 실제로 2개 후보를 평가했음. 두 출력 GPR의 최종 개발 학습과 모든 CV 학습에서 100개를 사용했고, `gpr_subsamples.csv`는 두 출력×(최종 1+CV 3)×100 = 800행임.

학습 곡선은 n=25/50/100/128에서 총 210행을 실측했으며, GPR 한도 초과 생략 12행과 검증 행 보존을 위한 생략 9행을 남겼음. GPR은 n=128에서 학습하지 않았고 다른 모델은 n=128까지 평가했음. 160개 전체를 학습시켜 검증 행이 섞이는 평가를 하지 않았음. `learning_curve_samples.csv`와 고정 검증 분할의 ID도 확인했음.

전체 unittest 25개 통과, 실패 0개, 건너뜀 0개임. status 가짜 파일에서 failed 3개 제외·정답 결측 허용·혼합 ID를 확인했고, 8,000행 가짜 파일의 기화율 경고 경로도 확인했음. 기존 출력 172개 파일 해시는 모두 유지됐음. 저장된 smoke 모델의 개발 입력 예측과 최적화 캐시 예측도 일치함. 고유 경고 기록은 159개를 manifest에 보존했음. 이번 작업에서 finalize와 git add/commit은 실행하지 않았음. 실제 합본 자료와 3,000행 GPR 학습 성능·메모리·실행시간은 아직 검증하지 않았음. P1-1에서 마무리함.

## GPR 3,000행 측정, P1-2·P1-3

3,000행 벤치마크는 data/cases_10k.csv의 원본 입력만 사용함. 합성 log10 응답은 log10(Q_phys)+N(0,0.01²)이며, TTR 입력에는 이를 10의 거듭제곱으로 복원해 기존 GPR 학습 경로를 그대로 측정함. 연구용 정답 또는 학습 모델을 생성하지 않음. 커널별 독립 프로세스, seed 2026, ARD 7차원, alpha=1e-10, BLAS 1 thread 조건임. kernel optimizer의 내부 반복은 fit 1회의 일부임. 최대 RSS는 Linux 프로세스 절대 peak로, 라이브러리 메모리를 포함함. 평균 예측만 측정하며 sigma 계산은 포함하지 않음.

15-fold 예산 6/20/50 시간표는 세 커널의 실측 fit·10만 행 예측 시간, 다른 모델의 합성 응답 1-fold/1-candidate 측정을 이용한 proxy임. 독립 출력 2개, n_jobs=1, 탐색 CV와 보고 CV 및 refit·3회 1만 행 예측 벤치마크를 포함하고, 학습 곡선과 그림/I/O 비용은 제외함. full10k의 수력 사전 비교용 GPR 15 CV fit+1 refit도 주 GPR 시간으로 근사함. 후보 alpha와 실제 solver 응답에 따라 optimizer 시간이 달라질 수 있음. 이 proxy로 full10k 예산을 결정하지 않음.

full10k budget_rule은 2026-11-08에 고정했음. 실제 합본의 budget-50 기준 총 예상시간 <=4h이면 50, <=8h이면 20, 나머지는 6임. 기준 예산을 50으로 고정해 순환 판단을 방지함. 자동 적용한 값·기준 추정시간·이유를 manifest의 budget_decision과 effective_config에 기록함. quick과 명시적 CLI override도 이유를 기록함.

수력 GPR은 StandardScaler + ConstantKernel*RBF(6 ARD)+WhiteKernel, alpha=1e-7, normalize_y=True, seed 2026의 고정 구조임. ratio=dp_hyd/dp_rigid의 원단위 비율을 학습하며 별도 grid 탐색은 없음. 실제 solver 출력은 회귀 목표에만 사용하고 입력은 FEATURES 6개임. 각 fold와 개발 refit 전에 최대 gpr_max_rows개를 부분표본으로 뽑고 ID를 hydraulic_subsamples.csv에 남김. 향후 합본 finalize에서 const/GPR 양쪽의 경계 MAPE, 판정 불일치, false safe/unsafe를 final_pressure_comparison.csv로 보고하도록 사전 지정했음. 이번에는 finalize를 실행하지 않았음.

pilot_v3의 개발 분할·선택된 thermal 하이퍼파라미터·15개 fold를 재사용한 수력 비교는 개발 160행의 반복 CV 관측 480개임. 경계 38.3–58.3 kPa는 개발 설계점 16개×3회=48개 관측임.

| 수력 방식 | 비율 평균 절대 상대오차 [%] | 전체 압력 MAPE [%] | 경계 MAPE [%] | 경계 판정 불일치 / 48 | 불일치 고유 설계점 |
|---|---:|---:|---:|---:|---:|
| const 1.8 | 0.956516 | 0.835825 | 1.199382 | 3 | 1 |
| GPR | 0.196808 | 0.326287 | 0.435328 | 2 | 1 |

두 방법 모두 이 CV에서는 false safe 0건이며 false unsafe는 각각 3건·2건임. 열 성분을 실제값으로 고정해 수력 효과만 본 보조 계산의 경계 불일치는 3건·1건임. 주 표는 thermal AI의 검증 fold 예측을 사용함. 표는 튜닝에 사용된 개발 자료를 재사용하므로 독립 일반화 성능으로 해석하지 않음. 파일럿 테스트셋이나 기존 모델 폴더는 변경하지 않았음. 최종 비율 목표는 제공된 dp_hyd_Pa/dp_rigid_Pa 열로 계산했고, 압력 조합에만 rigid_pressure()를 사용함. 최종 결과는 outputs/hydraulic_pilot_cv_exact_ratio에 별도 저장했음. 초기 계산(반올림 전 강체식 분모) 결과는 outputs/hydraulic_pilot_cv에 보존했음.

configs/ann_ref.json은 hidden_layer_sizes, activation, solver, learning_rate, max_iter, n_train, source를 모두 null로 만들었음. 미기입 상태는 로그를 남기고 건너뜀. 채워진 설정은 원본 6입력의 고정 구조 MLP로 두 출력 비교표에 들어가고 튜닝 후보 수·시간은 0으로 기록함. n_train은 해당 학습 fold에서만 seed 2026으로 적용하며 가용 행이 적으면 실제 행 수를 기록함. 논문 미기재 항목은 사용자 확인 후 명시적인 미기재 마커로 sklearn 기본값을 사용하며, 현재 파일은 값이나 출처를 추측해서 채우지 않았음. 단위 테스트의 임시 ANN은 재현 실험이나 논문 설정이 아님.


GPR 실측 결과 (2026-11-08~09, 커널별 fit 1회, 총 3회):

| 커널 | fit [s] | 10만 행 예측 [s] | 최대 RSS [MiB] |
|---|---:|---:|---:|
| RBF | 207.587 | 37.178 | 2031.336 |
| Matern_1.5 | 206.167 | 39.441 | 2031.098 |
| Matern_2.5 | 188.967 | 44.841 | 2031.238 |

예산별 시간 proxy는 아래 표에 정리함. GPR 열은 주 출력 2개의 GPR 합계이며 전체 열에는 나머지 모델과 수력 GPR 비교 비용 proxy를 더했음. 범위는 세 커널 비용의 최소·최대이며 신뢰구간이 아님. 학습 곡선·그림·I/O는 제외했음.

| budget | 주 GPR 평균 [h] | 전체 평균 [h] | 전체 커널별 범위 [h] |
|---|---:|---:|---:|
| 6 | 11.876 | 13.379 | 12.680–13.768 |
| 20 | 35.390 | 38.092 | 36.008–39.254 |
| 50 | 85.779 | 91.048 | 85.997–93.868 |

실측 gpr_timing.csv, 다른 모델 1-fold proxy 측정, 파생 develop_estimates.csv, 입력 해시·환경·가정 benchmark.json, 경고 로그 benchmark.log를 outputs/gpr_3000_benchmark에 보존했음. 벤치마크에는 joblib 모델이나 합성 정답 CSV가 없음. 입력 파일 해시는 보존됐음. 예산 규칙은 합본이 아직 없으므로 실제 적용하지 않았음.

전체 unittest 33개 통과, 실패 0개, 건너뜀 0개임. 테스트에는 예산 4/8시간 경계, 자동 예산 적용·manifest 기록, 수력 목표 열 정의·경계 false safe/unsafe, cache의 1회 모델 로드, predict/objective의 const/GPR 일치, ANN 미기입 생략·고정 구조·튜닝 0회를 포함했음. 별도 quick 개발 smoke는 gpr_max_rows=100, 후보 2개, CV 3-fold ×1회로 실행했고 ANN 생략 이유와 예산 적용 이유를 manifest에 남겼음. 이 smoke의 비율 오차는 0.956508%→0.234640%, 경계 불일치 1/16→1/16으로, 위 15-fold 표와 별개임. 해당 smoke는 초기 강체식 분모 정의의 호환성 확인이며 최종 비율 목표 정의는 위 exact_ratio CV와 단위 테스트에서 확인했음. 저장 모델로 predict 및 optimize --hyd-model gpr 실행을 완료해 27개 후보를 생성했음. 원래 출력 212개 파일은 모두 보존됐으며 finalize와 커밋은 실행하지 않았음. 여기서 마무리함.

## 2026-11-09 합본 수령 전 비용 기반 실행 계획 재고정

이번 변경의 근거는 합성 응답을 사용한 시간·메모리 측정뿐임. solver 정답, 파일럿 테스트셋 결과, 새 모델의 예측 성능을 보고 바꾼 것이 아님. configs/full10k.json의 execution_revision에 날짜와 이유를 고정했고, 원본 설정 및 변경 이유를 manifest에 함께 기록함. 이전 절의 CV 15-fold·동일 budget·8시간 규칙은 당시 실행 기록이며 새 합본에는 아래 계획을 적용함.

native-BLAS 측정 명령은 `python benchmark_native_gpr.py --inputs data/cases_10k.csv --out outputs/gpr_native_benchmark`임. Matérn 2.5, alpha=1e-10, 물리 특성을 포함한 7차원, seed 2026과 같은 합성 응답 정의를 사용했음. 스레드 제한 없이 NumPy/SciPy OpenBLAS 모두 18스레드였고, fit 호출은 3,000행·2,000행 각각 1회임. 가중치나 합성 정답을 저장하지 않았음.

| 행 수 | native fit [s] | 최대 RSS [MiB] | fit 전 가용 RAM [MiB] |
|---|---:|---:|---:|
| 3000 | 134.141 | 2051.500 | 8015.160 |
| 2000 | 79.966 | 1065.664 | 7978.418 |

가용 RAM 중 작은 측정값 약 7,978 MiB에서 시스템 예약 2,048 MiB를 뺀 5,930 MiB를 사용 상한으로 정했음. worker당 예약량은 최대 RSS×1.25로, 3,000행 2,564.375 MiB·2,000행 1,332.080 MiB임. RAM 상한은 각각 동시 2개·4개임. BLAS 18스레드로 동시 fit하면 CPU 18개 기준 worker 1개로 제한해야 함.

선택한 계획은 BLAS 1스레드와 RAM/CPU 제한 fold 병렬임. 3,000행·budget 50의 비용 proxy는 이 계획에서 3.096시간, native 18스레드 직렬에서 4.390시간이었음. native 조건의 다른 모델 비용은 직접 재측정하지 않았고 기존 1스레드 proxy를 재사용한 제한이 있음. 3,000행 GPR의 1스레드 시간은 이전 세 커널 측정의 평균을 사용했음. 2,000행의 1스레드 시간은 새 native 측정의 행 수별 비율로 옮긴 추정이며 직접 실측하지 않았음.

| GPR 행 수 | worker | budget 6 [h] | budget 20 [h] | budget 50 [h] |
|---|---:|---:|---:|---:|
| 3000 | 2 | 2.468 | 2.668 | 3.096 |
| 2000 | 4 | 0.904 | 1.001 | 1.218 |

전체 develop proxy는 독립 출력 2개의 탐색 CV·보고 CV·refit·예측 벤치마크와 수력 GPR 비교를 포함함. 학습 곡선·프로세스 시작/종료·그림/I/O·병렬 경합은 제외했음. 후보별 수렴 비용을 정확히 예측하지 못하므로 합본의 실제 실행시간이나 모델 성능을 보장하는 표가 아님.

CV는 5-fold×1회임. 약 1만 행에서는 검증 fold당 약 1,600행이어서 반복으로 분산을 줄일 필요가 작다는 사전 판단과 비용 절감이 이유임. GPR은 고정 3 kernel×2 alpha=6개를 전수 탐색하고 다른 튜닝 모델만 budget개를 평가함. 비교표 actual_candidates·manifest tuning_candidates는 실제 수임. 고정 선형·물리식·참고 ANN의 튜닝 수는 0임.

각 단계에서 현재 MemAvailable과 설정 상한을 재확인해 worker 수를 RAM·CPU 상한 중 작은 값으로 제한함. 탐색·보고 CV·수력 CV는 같은 스레드 제어를 사용함. 탐색 후 worker를 종료하고 메모리를 다시 확인한 뒤 parent refit 1회를 수행함. GridSearchCV를 refit=False로 두고 같은 best_params를 parent에서 한 번 학습하므로 후보 수나 fit 횟수를 늘리지 않았음.

budget 50의 벽시계 추정이 4시간 이하면 50, 아니면 20이 4시간 이하면 20, 나머지는 6임. 3,000행·6도 4시간 초과면 2,000행 GPR만 다시 실측하고 다른 모델은 기존 1-fold 측정을 새 worker 수로 재계산해 같은 규칙을 적용함. 2,000행·6도 초과면 target_met=false로 기록함. 원본·유효 설정, 예산·행 수 판단, 단계별 resource_plans를 manifest에 저장하며 quick·명시 CLI override의 이유도 남김.

전체 unittest 38개 통과, 실패 0개, 건너뜀 0개임. 6개 조합·GPR 고정 후보 수·동시 fit 메모리/CPU 제한·5-fold·2,000행 재측정과 모델/manifest 적용·목표 초과 표시를 확인했음. 실제 합본 develop, finalize, git add/commit은 실행하지 않았음. 여기서 마무리함.

## 마지막 준비: 무인 실행·재개·보고서 확인

`./run_full10k.sh /path/to/merged.csv` 한 줄로 nohup 백그라운드 실행함. 검증, 실제 데이터 1-fold 시간 추정과 계획 고정, learning-curves develop, finalize 1회, const/GPR 각각 S1–S3 최적화, 보고서를 순서대로 생성함. 기본 경로는 outputs/full10k이며 models/와 report/, optimization/const 및 optimization/gpr로 나눕니다. 시작 RAM 확인과 각 fit 단계의 기존 자원 제한을 적용함.

실행 잠금과 데이터/설정/시나리오/ANN 해시를 저장함. 비용 추정은 원본 설정과 실제 적용 예산·GPR 행 수를 담은 execution_plan.json으로 develop에 전달함. 완료 영수증은 산출물 해시를 포함하며, 완료된 단계는 다시 실행하지 않음. 일반 실패는 새 staging 폴더로 재개함. finalize는 child 시작 전 durable 상태를 남기며, 시작 후 실패하거나 중단되면 완료 영수증이 없는 한 절대로 재시도하지 않음. 정상 완료 파일이 있으면 최종 테스트를 다시 열지 않고 이후 단계만 이어갑니다.

T1은 두 출력의 모든 비교 모델 CV R²/RMSE/MAPE 평균·표준편차, log 지표, CV fit 시간, 최종 refit 시간, 탐색 시간, 1만 건 예측 시간, 실제 후보 수를 포함함. T2는 저장된 모델에서 최종 파라미터와 GPR fitted_kernel_를 읽음. F1은 최종 테스트 저장 예측의 선정 모델·MLP 패리티, F2는 개발 검증 RMSE 곡선(두 출력 나란히, x/y log, 반복 표준편차), F3는 MLP 학습 손실임. F4/T3는 수력 모드별 최적화 원본과 동일한 파일을 수집함. 표는 CSV, PNG는 300 dpi·흰 배경·축 단위임.

파일럿/현재 모델 오차 비교는 공통 모델의 저장된 개발 CV이며 데이터·특성·fold 수 차이를 순수한 데이터 증대 효과로 해석하지 않음. 물리 특성 효과는 pilot_v2/pilot_v3의 동일 데이터·분할·CV 인덱스를 확인한 기존 비교임. 합본에서 물리 특성 없이 재학습한 실험은 추가하지 않았음. 재검증점은 같은 A/B/C 역할끼리, extra는 설계 범위로 정규화한 거리 합 최소의 일대일 대응으로 비교함. 수력 효과는 현재 개발 OOF와 사전에 정한 최종 const/GPR 비교를 구분함. 모든 요약은 저장된 결과를 읽기만 함.

실제 확인 명령은 `./run_full10k.sh data/dataset_ai.csv --quick`임. 도구 샌드박스가 첫 백그라운드 프로세스를 Python 초기화 전에 정리해, 모니터가 열린 동일 도구 세션에서 nohup 스크립트를 실행해 검증했음. 실제 workflow 체크포인트 생성 후에는 모든 단계가 1회씩만 실행됐음. 기존 파일럿 결과 폴더를 덮어쓰지 않았음.

- 전체 unittest: 43개 통과, 실패 0개, 건너뜀 0개.
- quick 전체 소요시간: 123.216초 (약 2분 3.2초).
- 단계별 시간: 검증 1.721초, 추정 3.495초, develop 58.415초, finalize 3.168초, const 최적화 24.172초, GPR 최적화 26.435초, 요약 4.794초.
- finalize 시도 1회, 각 모드 재검증점 27개, 병합 T3 54개.
- report/에 PNG 6개·CSV 21개·Markdown 1개·JSON 2개, 총 30개 파일.
- 모든 PNG의 DPI와 흰 모서리 배경 확인, F4/T3 원본 해시 일치 확인.
- 원래 outputs의 285개 파일 해시 보존.

기본 실행에서 사용하는 실제 합본은 아직 없으므로 quick 결과를 합본 성능·실제 전체 실행시간으로 주장하지 않음. 이 확인 실행에서 최종 홀드아웃은 모델 선정 뒤 한 번만 평가했고, 이후 모델/하이퍼파라미터를 변경하지 않았음. 커밋은 실행하지 않았음.
