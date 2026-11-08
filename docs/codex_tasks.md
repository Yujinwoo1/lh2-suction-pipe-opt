# Codex 작업 요청: LH2 흡입배관 대리모델 → 최적화·1만 개 대응으로 확장

## 0. 배경 (먼저 읽을 것)

이 저장소는 LH2 흡입배관 1D 솔버 출력 200개(파일럿)로 대리모델 5종(선형, 다항 Ridge, RF, GPR, MLP)을 비교하는 파이프라인이다. `README.md`, `docs/PILOT.md`, `lh2_pipeline.py`, `predict.py`, `tests/test_pipeline.py`를 먼저 읽고 기존 설계 원칙을 그대로 따른다.

현재 상태 (Linux, Python 3.12, sklearn 1.9에서 확인함):
- `python -m unittest discover -s tests -v` → 5개 통과
- `python lh2_pipeline.py develop --learning-curves` → 약 1분 30초, 선정 모델은 두 출력 모두 GPR
  - CV log10 RMSE: Q_leak GPR 0.0054, dp_thermal GPR 0.035. MLP는 각각 0.103, 0.157
- 경고 51개는 대부분 Q_leak GPR의 f_Hz, mdot_kg_h 길이척도가 상한(1e5)에 닿은 것이다. 열침입은 전도 문제라 주파수와 유량에 무관하므로 물리적으로 맞는 결과다. 경고를 숨기지 말고, 해석 자료(가산점)로 쓸 수 있게 남겨 둔다.

데이터:
- `data/dataset_ai.csv`: 파일럿 200개, 16열. Git에 포함하지 않으며 로컬에 직접 복사해 둔다.
- `cases_10k.csv`: 입력 7열만 있고 정답이 없다. **학습에 쓰지 않는다.**
- 1만 개 솔버 결과 합본(약 10,200행, 같은 16열 예정)은 아직 받지 못했다. 이 파일이 오면 같은 코드로 바로 돌릴 수 있게 준비하는 것이 목표다.

일정: AI 결과 마감 11/10(화), 발표 11/13(금). **재검증 점 목록(1D 솔버에 다시 넣을 설계점)이 가장 급하다.**

## 1. 절대 규칙 (모든 작업에 적용)

1. seed는 모든 무작위 과정에서 `2026`을 쓴다.
2. 고정 테스트셋은 모델 선정이 끝난 뒤 `finalize`에서 한 번만 연다. 테스트 결과를 보고 모델, 하이퍼파라미터, 식을 바꾸지 않는다. 기존 `outputs/pilot/`의 결과는 덮어쓰지 않는다.
3. 입력은 `FEATURES` 6개만 쓴다. `case_id`, `Q_leak_W`, `dp_design_Pa`, `dp_hyd_Pa`, `vaporized`, `dp_rigid_Pa`를 입력에 넣지 않는다.
4. 스케일러와 다항 변환은 Pipeline 안에서 학습 fold에만 fit한다.
5. `data/`와 `outputs/`, `*.joblib`은 커밋하지 않는다.
6. 200개 모델의 예측값으로 1만 개의 정답을 채우지 않는다(의사 라벨 금지).
7. 구한 숫자를 실제보다 좋게 표기하지 않는다. 예를 들어 50회 튜닝을 하지 않았으면 50회라고 쓰지 않고, 대리모델 Pareto를 솔버 검증 결과라고 쓰지 않는다.
8. 작업 단위마다 테스트를 추가하거나 갱신하고, `unittest`가 통과한 뒤 커밋한다. 커밋 메시지는 한국어로 쓴다.

**작업 1에서 위 규칙을 저장소 루트의 `AGENTS.md`로 만든다.** 실행 명령과 "테스트셋 1회 원칙"도 함께 적는다.

## 2. 작업 목록 (우선순위 순)

### P0-1. `optimize.py`: NSGA-II + 격자 탐색 + 재검증 점 목록 ★가장 급함

선정된 파일럿 모델(`outputs/pilot`의 `manifest.json` → `selected`)을 불러 최적화한다. 모델 디렉터리는 인자로 받아서, 나중에 1만 개 모델 폴더로 바꿔 끼울 수 있게 한다.

**문제 정의**
- 설계변수: `D_mm` 12–50, `t_ins_mm` 2–25
- 목적 3개(모두 최소화): `Q_leak` (AI), `dp_design` = AI(dp_thermal) + 1.8 × `rigid_pressure` (`lh2_pipeline.rigid_pressure`를 재사용), `D_o` = D + 3 + 2·t_ins
- 제약: `dp_design ≤ 48,300 Pa`
- 시나리오(설정 파일 `configs/scenarios.json`으로 분리, 기본값은 아래와 같음):

| id | f_Hz | L_m | mdot_kg_h | k_MLI_W_mK |
|---|---|---|---|---|
| S1 기준 | 6 | 2 | 80 | 8e-5 |
| S2 진공 열화 | 6 | 2 | 80 | 2.7e-3 |
| S3 고부하 | 9 | 2 | 100 | 8e-5 |

- 시나리오마다 n* = 1040/(4·L·f)를 계산해 표에 넣는다. n* < 20이면서 정수와의 차이가 0.1 미만이면 경고를 출력한다.

**구현 요구사항**
- 모델은 시나리오마다가 아니라 **한 번만 로드**한다. 목적함수는 벡터화한다(pymoo `ElementwiseProblem`이 아니라 `Problem`). `predict.predict()`는 호출할 때마다 joblib을 다시 읽으므로 최적화 루프에서 직접 쓰지 말고, 모델을 캐시하는 헬퍼를 만든다. 범위 검사 로직은 재사용한다.
- NSGA-II: pymoo, 개체 100, 200세대, seed 2026. `requirements.txt`에 `pymoo`를 추가한다.
- **격자 탐색(정답지)**: D × t_ins를 200 × 200으로 전수 평가하고 제약을 만족하는 점에서 비지배 해를 구한다. 이건 "대리모델 기준의 정답"이지 솔버 정답이 아니라는 점을 출력과 주석에 명시한다.
- **hypervolume 비교**: 격자 Pareto의 ideal/nadir로 두 해집합을 같은 방식으로 정규화하고, 기준점은 (1.1, 1.1, 1.1)로 둔다. NSGA-II HV / 격자 HV 비율을 보고한다.
- **GPR 불확실성**: `TransformedTargetRegressor`는 `return_std`를 지원하지 않으므로, `model.regressor_`(Pipeline)에서 `predict(X, return_std=True)`로 log10 공간의 σ를 구하는 헬퍼를 만든다. GPR이 아닌 모델이 선정됐으면 σ는 NaN으로 둔다.
- **후보 선택**(시나리오마다):
  - A: Q_leak 최소
  - B: 무릎점. Pareto 해의 목적값을 [0,1]로 정규화한 뒤 ideal(0,0,0)과의 유클리드 거리가 최소인 해. 정의를 출력 파일과 주석에 적는다.
  - C: D_o 최소
  - 추가 6개: 정규화 목적공간에서 farthest-point sampling으로 전선을 고르게 덮는다. 후보가 비슷하면 dp_thermal의 GPR σ가 큰 점과 제약 경계(dp_design 38.3–48.3 kPa)에 가까운 점을 우선한다.
- **출력** (`outputs/opt_<모델폴더이름>/`):
  - `verification_points.csv`: `point_id, scenario, role(A/B/C/extra), D_mm, t_ins_mm, f_Hz, L_m, mdot_kg_h, k_MLI_W_mK, Q_leak_pred_W, dp_thermal_pred_Pa, dp_design_pred_Pa, D_o_mm, sigma_log10_dp_thermal, n_star`. 앞의 입력 6열만 잘라서 1D 솔버 담당에게 보낼 수 있게 `verification_inputs.csv`도 따로 저장한다.
  - `pareto_<S>.csv`, `grid_pareto_<S>.csv`, `hv_summary.csv`, `T3_candidates.csv`
  - 그림 F4 (PNG, 300 dpi, 흰 배경, 축 단위 표기): 시나리오마다 Q_leak vs dp_design 산점도에 D_o를 색으로 표시하고, 격자 Pareto를 회색으로 겹치고, A·B·C를 표시하고, 48.3 kPa 제약선을 그린다.
- **재검증 결과 비교 기능**: `optimize.py compare --points verification_points.csv --solver solver_results.csv`로 1D 결과가 오면 Q_leak와 dp_design의 오차(%)와 제약 판정 일치 여부를 표(T4)로 만든다. 지금은 솔버 결과가 없으니 가짜 결과 파일로 테스트만 만든다.
- **테스트**: 목적함수의 벡터 결과와 `predict()`의 결과가 일치하는지, 격자 Pareto가 비지배 집합인지, 설계 범위 밖 시나리오는 거부하는지 확인한다.

### P0-2. 열침입 물리식 기준선

이미 200개로 검증한 식이다(평균 오차 0.06%, 최대 0.40%). 상수를 튜닝하지 말고 아래 식 그대로 구현한다.

```
r_i  = D_mm/2000                      # 배관 내반경 [m]
r_w  = r_i + 0.0015                   # SS316 벽 1.5 mm (벽 저항은 무시)
r_o  = r_w + t_ins_mm/1000            # MLI 외반경 = D_o/2
R    = ln(r_o/r_w)/(2π·k_MLI·L) + 1/(h·2π·r_o·L),   h = 5 W/m²K
Q_leak = (300 − 23.6) / R             # [W]
```

- 비교표에 `physics_baseline` 모델로 넣는다. Q_leak에만 해당하고, 튜닝은 없으며 CV fold에서도 같은 식을 쓴다.
- 데이터 감사(`audit`)에 물리식 오차의 평균과 최대를 추가한다.
- 발표 메시지: "Q_leak는 물리식으로 충분하다 → AI의 실질 가치는 dp_thermal에 있다." 결과표에서 이 메시지가 드러나게 정리한다.

### P1-1. 1만 개 데이터 대응 (합본이 오면 바로 돌릴 수 있게 지금 준비)

- **GPR 3,000개 제한**: 지금은 3,000개를 넘으면 중단한다. 이걸 "개발 데이터에서 seed 2026으로 3,000개를 부분표본으로 뽑아 학습"으로 바꾼다. 부분표본 크기는 인자로 받고, manifest에 기록한다. 다른 모델은 전체 개발 데이터를 쓴다(공정성 주석: "GPR은 n³ 계산량 때문에 부분표본 사용").
- **튜닝 예산**: `--budget`(기본 6, 발표용 목표 50)와 `--cv-repeats` 인자를 추가한다. 모든 튜닝 모델에 같은 후보 수를 쓴다. 1만 개에서 50 × 15 fold는 오래 걸릴 수 있으니, **실행 전에 모델별 1-fold 시간을 재서 전체 예상 시간을 출력하는 `--estimate-time`** 옵션을 만든다. 시간이 부족하면 budget이나 repeat를 줄이고 그 값을 그대로 기록한다.
- **학습 곡선**: n = 25, 50, 100, 200, 400, 800, 1600, 3200, 전체 중 개발 데이터 크기 이하의 값만 쓴다. n마다 무작위 추출을 5회 하고, 평가는 개발 데이터 안의 고정 검증 분할에서 한다(테스트셋 사용 금지). GPR은 n > 3000이면 생략하고 표에 표시한다.
- **데이터 로딩**: `status` 열이 있으면 `converged` 행만 회귀에 쓰고, 제외한 개수와 case_id를 감사 파일에 기록한다. `case_id`가 1–200과 100001–110000으로 섞여 있어도 동작해야 한다.
- 새 출력 폴더(`outputs/full10k`)에서 새 분할을 고정하고, 데이터 해시를 기록한다. 파일럿 결과와 섞지 않는다.

### P1-2. 수력 계수 보정 (실험 옵션, 기본값은 1.8 유지)

파일럿 분석에서 `dp_hyd/dp_rigid` 비율이 1.77–1.84로 움직이고, 입력 6개로 GPR 학습했을 때 R² 0.93, 평균 오차가 0.95% → 0.18%로 줄었다. 경계 근처는 수력 성분이 대부분이라 판정에 영향이 크다(case 110이 그 예).
- `--hyd-model {const,gpr}` 옵션을 추가한다. `gpr`이면 개발 데이터로 비율을 학습하고, CV로 평가하고, dp_design = AI(dp_thermal) + ratio_AI × dp_rigid로 계산한다.
- 이 옵션은 파일럿 테스트 결과를 보고 정한 것이 아니지만, 이미 테스트를 연 파일럿 폴더에는 적용하지 않는다. **1만 개 새 분할에서 사전에 정한 비교 항목**으로만 쓴다.
- `finalize`에서 const와 gpr 두 방식의 경계 구간 오차, false safe, false unsafe를 나란히 보고한다.

### P1-3. 참고문헌 [8,9] 재현 ANN 자리 만들기

- `configs/ann_ref.json`에 `hidden_layer_sizes, activation, solver, learning_rate, max_iter, n_train, source` 칸을 만든다. 값은 **비워 두고** 추측해서 채우지 않는다. 논문에 없는 항목은 `"논문 미기재, 기본값 사용"`으로 표기한다.
- 설정이 채워지면 `ann_ref` 모델이 튜닝 없이 고정 구조로 비교표에 들어가게 한다. 비어 있으면 건너뛰고 로그에 남긴다.

### P2. 가산점 (시간이 남을 때)

1. 외삽 시험: D < 40으로 학습하고 D ≥ 40을 예측한다. k_MLI 상위 20%에도 같은 시험을 한다. 테스트셋과 무관하게 개발 데이터 안에서만 한다.
2. 해석: GPR ARD 길이척도 표, RF permutation importance, PDP(t_ins↑ → Q_leak↓, k_MLI↑ → Q_leak↑ 확인).
3. Pareto 그림에 GPR ±2σ 오차 막대 넣기, 보수 판정(예측 + 2σ ≤ 48.3 kPa) 추가.

## 3. 완료 기준

- `python -m unittest discover -s tests -v` 전부 통과
- `python optimize.py run --models outputs/pilot` 한 번으로 S1–S3의 Pareto, 격자 비교, HV, A·B·C·추가 점, `verification_inputs.csv`, F4 그림이 생성됨
- 1만 개 합본을 `--data`로 넣고 `develop --estimate-time` → `develop` → `finalize` → `optimize.py run`이 코드 수정 없이 돌아감
- README와 PILOT.md에 새 명령, 새 출력, 한계(대리모델 Pareto ≠ 솔버 검증, GPR 부분표본, 실제 튜닝 횟수)를 반영함

## 4. 진행 방식

- P0-1 → P0-2 → P1-1 → P1-2 → P1-3 → P2 순서로, 작업마다 별도 커밋으로 진행한다.
- 시작 전에 계획을 5줄 이내로 보여 주고, 각 작업이 끝나면 바뀐 파일, 실행 명령, 생성된 결과 요약(숫자 포함)을 보고한다.
- 위 규칙과 충돌하는 판단이 필요하면 임의로 정하지 말고 멈춰서 질문한다.
