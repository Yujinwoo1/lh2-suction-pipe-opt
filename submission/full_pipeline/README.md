# LH2 흡입배관 실행 코드

이 문서는 파일럿부터 합본 준비까지의 실행 기록을 담고 있음. 제출본의 최종 결과는 상위 README와 핵심 노트북에서 확인 가능함.

LH2 흡입배관 1D 솔버의 열침입·압력 열 성분을 근사하는 200개 파일럿 대리모델임.
Windows에서 작성한 Python 코드를 별도 Linux 머신의 CLI에서 실행 가능함. WSL, Windows 드라이브 마운트, GUI는 필요하지 않음.

## Linux에서 시작하기

Git, Python 3.12, venv/pip 설치가 필요함. 비공개 저장소는 GitHub 접근 권한으로 먼저 인증해야 함.

```bash
git clone https://github.com/Yujinwoo1/lh2-suction-pipe-opt.git
cd lh2-suction-pipe-opt
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements-tested.txt`는 Windows 파일럿 실행에서 확인한 버전 기록임. Linux에서도 같은 버전을 맞추려면 위 설치 명령의 파일명을 바꿔 사용함. 이 환경 설명은 파일럿 준비 당시의 기록임. 제출본 실행 환경은 상위 README에 정리돼 있음.

## 데이터 준비

데이터와 학습 모델은 Git에 포함하지 않음. Clone만으로 코드가 준비되며, 학습에는 별도로 전달받은 원본 `dataset_ai.csv`가 필요함.

```bash
mkdir -p data
cp /path/to/dataset_ai.csv data/dataset_ai.csv
```

`/path/to/dataset_ai.csv`는 Linux에 확보한 실제 파일 경로로 바꿔 사용함. `cases_10k.csv`에는 정답 출력이 없어 학습에 사용하지 않음. 테스트도 제공된 200개 파일럿 데이터가 필요함.

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

그래프는 비대화형 Matplotlib Agg 백엔드로 PNG에 저장하므로 화면 없는 서버에서도 실행 가능함. 결과는 `outputs/pilot/`에 저장함. 이미 완료한 개발 실험과 최종 테스트를 덮어쓰지 않음. 실행 오류는 종료 코드와 터미널 로그로 확인함.

빠른 동작 확인만 필요하면 별도 폴더에 `python lh2_pipeline.py develop --quick --out outputs/smoke`를 실행함. `--quick` 결과를 기본 실험과 구분해 보관함. 테스트 결과를 본 뒤 모델을 재선정하거나 튜닝하지 않음.

## 새 입력 예측

```bash
python predict.py /path/to/new_cases.csv --models outputs/pilot --output outputs/predictions.csv
```

필수 입력 열은 `D_mm`, `t_ins_mm`, `f_Hz`, `L_m`, `mdot_kg_h`, `k_MLI_W_mK`임. 지정 설계 범위를 벗어난 입력은 중단함. 제약 만족 여부는 대리모델의 예측이며 실제 솔버 검증을 대체하지 않음.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `lh2_pipeline.py` | 데이터 감사, 고정 분할, 모델 비교, 최종 평가 |
| `physics_baseline.py` | 상수 튜닝 없는 열침입 물리식 기준선 |
| `predict.py` | 저장된 선정 모델로 새 설계 조건 예측 |
| `summarize_results.py` | 기본 실험 결과를 `outputs/pilot/RESULTS.md`로 요약 |
| `tests/test_pipeline.py` | 파일럿 데이터 기반 검증 5개 |
| `LH2_pilot.ipynb` | 선택적으로 사용하는 설명·실행 노트북 |
| `make_notebook.py` | 노트북 재생성 |
| `requirements.txt` | 설치용 의존성 범위 |
| `requirements-tested.txt` | Windows 실행 당시 버전 기록 |
| `docs/PILOT.md` | 실험 설계, 평가 기준, 발표 시 해석과 한계 |

## 현재 범위와 한계

- 개발 160개 / 테스트 40개, seed 2026, 개발 CV 5-fold × 3회임.
- 선형회귀, 다항 Ridge, RF, GPR, MLP를 비교하며 두 출력을 각각 log10 공간에서 학습함.
- 튜닝 모델당 6개 후보를 사용한 파일럿임. 원 지시서의 50회 튜닝 전체를 수행한 것으로 표기하지 않음.
- NSGA-II와 격자 기반 대리모델 최적화를 제공함. Q_leak에는 튜닝 없는 열침입 물리식 기준선을 함께 비교함. 신규 1D 솔버 검증과 참고논문 ANN 재현은 아직 포함하지 않았음.
- 200개는 실험 측정값이 아닌 1D 솔버 출력임. 수력 근사오차를 AI 오차와 분리해야 함.
- GPR은 n³ 비용 때문에 각 학습 fold와 최종 개발 학습에서 최대 3,000개를 seed 2026으로 부분표본 추출함. 다른 모델은 해당 학습 자료 전체를 사용함.

세부 방법은 [파일럿 설명](docs/PILOT.md)을 참고. 기존 Windows 전용 `run_local.ps1`과 로컬 Python 패키지는 이 저장소에 포함하지 않았음.

## 최적화와 솔버 재검증 전달

```bash
python -m pip install -r requirements.txt
python optimize.py run --models outputs/pilot
python optimize.py compare --points outputs/opt_pilot/verification_points.csv --solver solver_results.csv
```

`configs/scenarios.json`의 S1–S3를 대상으로 seed 2026, 개체 100, 200세대 NSGA-II와 200×200 격자를 평가함. 모델은 한 번만 로드함. 새 모델 폴더는 `--models`, 시나리오는 `--scenarios`, 새 출력은 `--out`으로 지정함. 기존 출력 폴더는 덮어쓰지 않음.

`outputs/opt_pilot/`에 `pareto_S*.csv`, `grid_pareto_S*.csv`, `hv_summary.csv`, `T3_candidates.csv`, `verification_points.csv`, 여섯 입력 열만 담은 `verification_inputs.csv`, `F4.png`, 설정과 후보 정의를 기록한 `optimization.json`을 저장함. HV는 격자 ideal/nadir로 공통 정규화하고 기준점 (1.1, 1.1, 1.1)을 사용함. 격자와 NSGA-II를 합친 비지배 전선에서 A(열침입 최소), B(정규화 ideal 거리 최소), C(외경 최소), 추가 최대 6개를 선택함. 극점이 겹치면 A/B/C가 같은 설계일 수 있음.

솔버 결과는 `verification_points.csv`의 `point_id`를 그대로 보존한 `point_id,Q_leak_W,dp_design_Pa` 열로 전달받음. 입력 전용 파일에는 ID가 없으므로 원래 행 순서와 점 목록의 대응을 유지해야 함. `compare`는 ID로 결합하여 부호 있는 오차율 100×(예측/솔버−1), 절대 오차율, 제약 일치 여부를 `T4_comparison.csv`로 저장함. 대리모델 Pareto와 HV는 솔버 검증 결과가 아님. 비GPR의 불확실성은 NaN이며 GPR σ는 log10 공간 값임.

## 열침입 물리식 기준선 (P0-2)

```bash
.venv/bin/python lh2_pipeline.py develop --out outputs/pilot_v2
```

기존 `outputs/pilot`을 보존하고, Q_leak 비교에 `physics_baseline`을 추가함. 식과 상수는 `docs/codex_tasks.md` P0-2 그대로이며 튜닝하지 않음. 입력은 기존 FEATURES 6개이고 전도율은 `10**log10_k_MLI`로 복원함. 다른 모델과 동일한 개발/최종 테스트 분할 및 CV 검증 fold에서 평가함. 이 명령은 최종 홀드아웃 성능을 계산하지 않음.

`cv_summary.csv`와 `cv_fold_metrics.csv`에 동일 지표로 기록하고, `data_audit.json`에는 전체 자료의 물리식 평균·최대 절대 상대오차(%)를 별도 저장함. 전체 200개 감사 수치와 개발 CV 수치를 혼합하지 않음. Q_leak는 이 파일럿에서 물리식으로 충분하며 AI 비교의 실질적인 초점은 dp_thermal임. 최종 일반화 성능은 아직 평가하지 않았음.

## 물리 파생 특성 실험

```bash
.venv/bin/python lh2_pipeline.py develop --physics-feature --out outputs/pilot_v3
.venv/bin/python optimize.py run --models outputs/pilot_v3
```

`--physics-feature`는 학습 모델의 Pipeline 첫 단계에서 원본 FEATURES 6개만 선택한 뒤 `log10(Q_phys/mdot_kg_h)`를 추가함. Q_phys는 `physics_baseline.py`의 고정식을 그대로 사용하며, 유량 단위는 kg/h임. 정답 열·ID·외부에서 채운 파생값은 사용하지 않음. 스케일러와 다항 변환은 이 단계 이후 각 학습 fold에서 fit함. GPR ARD 길이척도는 7개로 맞춥니다. 옵션이 없으면 기존 6입력 동작을 유지함.

옵션은 두 출력의 학습 모델에 적용하며 고정 `physics_baseline`에는 적용하지 않음. 저장된 Pipeline이 파생값을 계산하므로 `predict.py`와 `optimize.py`에 별도 전처리나 옵션이 필요하지 않음. 최종 홀드아웃은 평가하지 않음. 실험의 상세 비교 기준과 결과는 `docs/PILOT.md`를 참고.

## 1만 개 합본 준비 (P1-1)

`configs/full10k.json`은 합본 수령 전인 2026-11-08에 고정한 설정임. 물리 특성은 파일럿 개발 CV만을 근거로 켰으며, 파일럿 테스트셋은 근거로 쓰지 않았음. 현재 실행 계획은 CV 5-fold × 1회, GPR 최대 3,000개와 후보 6개 전수 탐색임. 다른 튜닝 모델의 budget은 벽시계 규칙으로 정하며, fold worker는 RAM/CPU 한도에서 자동 제한하고 BLAS는 1스레드로 고정함.

```bash
# 합본 수령 후: 먼저 비용 확인 (모델 저장 및 출력 폴더 변경 없음)
.venv/bin/python lh2_pipeline.py develop --config configs/full10k.json --data /path/to/merged.csv --out outputs/full10k --learning-curves --estimate-time
# 예상 시간을 확인한 뒤 개발 실행
.venv/bin/python lh2_pipeline.py develop --config configs/full10k.json --data /path/to/merged.csv --out outputs/full10k --learning-curves
# 모델 선정 완료 후 사용자가 별도로 승인한 최종 평가에서만, 한 번 실행
.venv/bin/python lh2_pipeline.py finalize --config configs/full10k.json --data /path/to/merged.csv --out outputs/full10k
```

설정 우선순위는 기본값 → JSON → quick 축소값 → 명시한 CLI 인자임. `--budget`, `--cv-repeats`, `--n-jobs`, `--gpr-max-rows`, `--physics-feature/--no-physics-feature`, `--blas-threads`로 조정한 실제 값은 원본 JSON 전체와 함께 manifest의 `effective_config`에 기록함. quick은 다른 튜닝 모델 후보 2개, 3-fold × 1회이며 발표용 결과가 아님. full10k의 GPR 전수 탐색 설정은 quick에서도 유지해 6개를 평가함. finalize는 저장된 유효 설정과 모델을 사용하고 다른 JSON은 거부함. 이번 P1-1 작업에서 finalize는 실행하지 않았음.

GPR 전처리도 부분표본에만 fit하며 `gpr_subsamples.csv`에 최종 개발 학습과 각 CV 학습 fold의 case_id를 저장함. 공통 CV 검증 행은 모두 평가함. `status`가 있으면 정확히 `converged`인 행만 사용하고 제외 수·ID를 감사에 기록함. 8,000–12,000행 자료에서 회귀 대상의 기화 비율이 파일럿 6%와 3%p 초과 차이면 경고하며 비율 자체를 이유로 거부하지 않음.

학습 곡선은 개발 자료 안의 고정 20% 검증 분할에서 평가함. 요청한 25/50/100/200/400/800/1600/3200/개발 전체 크기 중 개발 행 수 이하만 표에 넣으며, 크기마다 학습 풀에서 5회 무작위 추출함. 검증 행 때문에 학습할 수 없는 크기는 `skipped_validation_holdout`, GPR 한도 초과는 `skipped_gpr_max_rows`로 표시함. 사용 가능한 학습 풀 전체 크기도 별도 실측점으로 넣음. 분할과 표본 ID는 `learning_curve_split.csv`, `learning_curve_samples.csv`에 저장함. 최종 홀드아웃은 사용하지 않음.

시간 추정은 출력별·모델별 후보 하나를 개발 fold 하나에서 학습한 실측값으로 탐색, refit, 보고 CV, 선택한 학습 곡선, 1만 행 예측 벤치마크 시간을 추정함. 후보마다 비용이 다르고 병렬 메모리 경합·I/O·그림 비용은 반영하지 못하므로 대략적인 예산 확인용임. 합본이 오면 합본 자체로 다시 추정해야 함.

## GPR 비용 측정·수력 보정·참고 ANN 설정

```bash
# 합성 응답을 쓰는 시간/메모리 측정 전용; 모델이나 합성 정답은 저장하지 않음
.venv/bin/python benchmark_gpr.py --inputs data/cases_10k.csv --out outputs/gpr_3000_benchmark --threads 1
# 기존 파일럿 개발 분할과 선택된 thermal 설정을 재사용하는 수력 CV 비교
.venv/bin/python hydraulics.py --models outputs/pilot_v3 --data data/dataset_ai.csv --out outputs/hydraulic_pilot_cv
# 새 실험에서 수력 GPR 사용 (기본 const=1.8)
.venv/bin/python lh2_pipeline.py develop --config configs/full10k.json --data /path/to/merged.csv --out outputs/full10k --hyd-model gpr
```

GPR 벤치마크는 입력 6개만 읽고 Pipeline 안에서 물리 특성을 추가함. seed 2026으로 선택한 3,000행, log10(Q_phys)에 표준편차 0.01 잡음을 더한 합성 응답을 세 후보 커널의 fit/예측 비용 측정에만 사용함. 커널마다 새 프로세스를 사용하며 Linux 최대 RSS에는 Python과 라이브러리 메모리도 포함됩니다. 10만 행 예측은 원래 입력을 반복하고 batch 1,000으로 실행함. alpha는 1e-10이며 커널별 fit 호출은 1회임. 이는 실제 dp_thermal 학습시간이나 성능이 아님.

2026-11-09 full10k 예산 규칙은 실제 합본의 `--estimate-time` 벽시계 추정에서 budget 50이 4시간 이하면 50, 아니면 budget 20이 4시간 이하면 20, 나머지는 6임. 3,000행·budget 6도 4시간을 넘으면 GPR 2,000행을 다시 측정하고 같은 순서로 선택함. 그래도 초과하면 target_met=false로 보고함. 이 변경은 합본 수령 전 비용 측정에만 근거하며 solver 성능이나 파일럿 테스트 결과에 따른 변경이 아님. 일반 develop도 이 규칙을 적용하고 예산·행 수·이유를 manifest에 기록함. quick은 축소 설정을 유지하고, 명시적 `--budget`은 CLI override로 기록함. 합성 벤치마크로 예산을 확정하지 않음.

`--hyd-model {const,gpr}`는 develop·predict·optimize에서 지원함. develop 기본값은 const이며, predict/optimize는 별도 지정이 없으면 저장 manifest의 선택을 따릅니다. GPR 보정은 원본 입력 6개로 dp_hyd/dp_rigid 비율을 예측함. 스케일러와 GPR은 학습 fold에만 fit하고 최대 행 수를 동일하게 적용함. `dp_design=AI(dp_thermal)+ratio_AI*dp_rigid`이며, 고정 구조 GPR의 커널은 학습 자료 안에서 최적화하고 별도 후보 탐색은 하지 않음. `configs/full10k.json`은 향후 단 한 번의 finalize에서 const/GPR 양쪽의 경계 오차와 false safe/unsafe를 보고하도록 사전 지정함. 기본 const에서도 이 비교용 GPR을 개발 자료로 준비함. 이번 작업에서 finalize는 실행하지 않았음.

`configs/ann_ref.json`의 7개 값은 모두 null임. 항목이 비어 있으면 ann_ref를 건너뛰고 로그와 manifest에 이유를 남김. 출처와 값이 모두 채워지면 각 출력에 튜닝 없는 고정 MLP를 추가함. 논문에 없는 항목은 명시적으로 `논문 미기재, 기본값 사용`이라 적을 수 있음. 참고 ANN은 원본 6입력만 사용하며 n_train은 학습 fold 안에서 적용하고 실제 학습 수를 기록함. `--ann-ref-config`로 별도 설정 파일을 지정할 수 있음. 현재 논문 재현 결과를 주장하지 않음.

## 2026-11-09 비용 기반 실행 계획

```bash
.venv/bin/python benchmark_native_gpr.py --inputs data/cases_10k.csv --out outputs/gpr_native_benchmark
.venv/bin/python plan_walltime.py --out outputs/full10k_execution_plan
```

추가 native-BLAS 벤치마크는 스레드 제한 없이 18스레드에서 Matérn 2.5의 3,000행·2,000행 fit을 각각 1회 측정했음. 모델이나 합성 정답은 저장하지 않음. 기존 1스레드 벤치마크와 새 native 측정만으로 실행 비용을 비교했으며, 학습 성능을 근거로 설정을 바꾸지 않았음.

설정은 fold/candidate 병렬과 BLAS 1스레드로 고정했음. 측정 RAM 기준으로 GPR 3,000행은 2 worker, 2,000행은 4 worker임. 시작 시 가용 메모리에서 2,048 MiB를 예약하고, 최대 fit RSS ×1.25를 worker당 예약함. 사용 한도는 5,930 MiB와 현재 가용량 중 작은 값이며 CPU/BLAS 상한도 적용함. 각 단계의 실제 worker 수와 메모리 계산을 manifest에 남김. 탐색·보고 CV·수력 CV를 병렬화하며, refit 전에 worker를 종료해 잔류 RSS와 parent fit이 겹치지 않게 함.

GPR은 커널 3개×alpha 2개를 전수 탐색하고 다른 튜닝 모델만 선택 budget을 적용함. `cv_summary.csv`의 actual_candidates와 manifest 모델별 tuning_candidates가 실제 수임. 기존 “모든 튜닝 모델에 같은 후보 수” 원칙을 이 비용 기반 계획에서 명시적으로 바꿨음. CV 반복은 1회로 줄였으며 fold당 약 1,600개 검증 표본을 사용할 예정이라는 이유를 config/manifest의 execution_revision에 기록했음.

합성 비용 proxy에서 전체 develop 예상은 3,000행의 budget 6/20/50에서 2.47/2.67/3.10시간, 2,000행에서 0.90/1.00/1.22시간임. 2,000행의 1스레드 fit은 직접 측정하지 않았고 native 2,000/3,000행 시간 비율로 환산했음. 병렬 경합·프로세스 시작/종료·학습 곡선·그림/I/O 비용은 제외했음. 실제 합본의 시간 추정으로 규칙을 다시 적용해야 하며, 이 표로 실행 예산을 확정한 것은 아님.

## 합본 무인 실행과 재개

```bash
./run_full10k.sh /path/to/merged.csv
```

스크립트가 nohup으로 백그라운드 실행하고 PID를 출력함. 기본 실행은 데이터 검증 → 학습 곡선을 포함한 시간 추정 → 결정된 예산·GPR 행 수로 develop → finalize 1회 → S1–S3 const/GPR 최적화 → 요약 순서임. 이 명령은 최종 홀드아웃 평가까지 자동 수행함. 모델은 `outputs/full10k/models`, 보고서는 `outputs/full10k/report`, 최적화는 `outputs/full10k/optimization/{const,gpr}`에 저장함.

`outputs/full10k/run.log`에는 단계별 시작·종료 시각과 출력, `logs/`에는 시도별 로그, `run_state.json`에는 체크포인트와 총 소요시간을 남김. 시작 시 가용 RAM을 확인해 worker 수를 제한하고 경고함. 실제 fit 전에도 기존 메모리/CPU 제한을 다시 적용함. `--out 새폴더`로 다른 실행 위치를 지정할 수 있음.

실패하면 중단하며 같은 명령을 다시 실행하면 동일 데이터·설정인지 확인하고 완료 단계는 해시 검증 후 건너뜀. 실패한 일반 단계는 새 임시 폴더에서 실행하고 부분 결과는 보존함. finalize는 시작 기록부터 재시도를 금지함. 정상 완료 영수증이 있으면 건너뛰고 다음 단계로 이어가지만, 실패·중단됐고 완료 영수증이 없으면 자동 재개를 차단함. 다른 데이터나 설정은 새 출력 폴더를 사용해야 함. 동시에 같은 폴더에서 두 작업이 실행되지 않도록 잠금을 사용함.

시간 추정 결과는 `estimate/execution_plan.json`에 저장하고 develop이 그대로 소비함. 같은 비용 측정을 다시 실행해 예산을 바꾸지 않으며, 원본 설정·유효 설정·데이터 해시·ANN 설정을 확인함. 원래 실험 출력은 덮어쓰지 않음.

검증용 명령은 `./run_full10k.sh data/dataset_ai.csv --quick`이며 기본 출력은 `outputs/dryrun_full10k`임. quick에서도 GPR 후보 6개는 전수 탐색하고, 다른 튜닝 모델은 2개 후보와 3-fold×1회를 사용함. 이 확인 실행의 결과는 1만 개 성능 결과가 아님.

보고서는 T1 모델별 개발 CV 평균±표준편차·학습/1만 건 예측 시간·실제 후보 수, T2 최종 모델 파라미터, F1 저장된 최종 테스트 예측 패리티, F2 개발 검증 학습 곡선, F3 MLP 손실을 생성함. F4/T3와 검증 입력·예측·HV는 수력 모드별 원본을 복사하며, T3 병합표에도 모드를 기록함. 그림은 300 dpi·흰 배경·축 단위를 사용하고 표는 CSV임. 파일럿 대비 CV 오차와 재검증점 이동량, 기존 동일 분할 파일럿의 물리 특성 효과, 현재 CV/최종 수력 보정 효과를 별도 표로 생성함. 학습 곡선은 사용자 확인대로 최종 홀드아웃이 아닌 개발 고정 검증 RMSE임. 요약 단계는 새 모델을 fit하거나 테스트 예측을 다시 계산하지 않음.

개별 요약은 `summarize_results.py --models ... --out 새폴더 --opt-const ... --opt-gpr ...`로도 실행 가능함. 참고 파일럿 결과가 없으면 효과·비교 표에 unavailable을 표시하며 과거 모델을 재학습하거나 수치를 추측하지 않음.
