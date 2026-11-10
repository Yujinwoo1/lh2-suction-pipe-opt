먼저 LH2_core.ipynb를 열어 보세요

LH2 흡입배관의 열침입·압력 손실·외경을 비교하고 설계 후보를 찾은 연구입니다. 노트북에는 설명, 표, 그림과 실제 실행 출력이 저장되어 있어 실행하지 않고 읽을 수 있습니다.

| 경로 | 내용 |
|---|---|
| `LH2_core.ipynb` | 물리식, 고정 설정 재학습, 최종 테스트 비교, S1 최적화 |
| `data/dataset_ai_10k.csv` | 정답이 있는 1D 솔버 출력 10,000개 |
| `results/` | 원본 `outputs/full10k/report/`의 표·그림과 결과 설명 |
| `results/reproduction/` | 원본 manifest·분할·부분표본 ID, 정확한 커널, 실행 환경, 재현 비교표 |
| `full_pipeline/` | 원본 코드·설정·문서·테스트의 사본과 제출본 검증 테스트 |

외경과 강체 압력은 식으로 계산합니다. 수력 압력은 강체 압력의 1.8배를 기본으로 쓰고 GPR 비율 보정과 비교합니다. AI의 핵심 역할은 열적 압력 손실 `dp_thermal` 예측입니다. 열침입은 물리식과 비교합니다. 원본이 두 출력 모두 GPR을 선정했으므로 노트북의 재현 평가와 최적화도 같은 모델 구성을 유지합니다.

노트북 실행은 제출 폴더에서 다음 순서로 준비합니다. JupyterLab을 열고 `LH2_core.ipynb`에서 커널을 선택한 뒤 **Restart Kernel and Run All Cells**를 실행하세요. 패키지 설치 시간은 노트북 측정 시간에 포함하지 않습니다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r results/reproduction/requirements-notebook.txt
pip install jupyterlab
python -m ipykernel install --prefix .venv --name lh2 --display-name 'LH2 Python'
jupyter lab LH2_core.ipynb
```

노트북 실행 시간과 원본 대비 차이는 아래 실측 표에 기록했습니다. 학습된 커널을 전체 정밀도로 고정하고 GPR 3,000개 부분표본의 ID·순서를 재사용합니다. MLP는 원본 선정 설정으로 개발 데이터 8,000개를 학습합니다. 튜닝과 커널 최적화를 생략하며, 원본 `finalize`는 다시 실행하지 않습니다. 최종 테스트는 고정 설정의 재현 비교에만 사용합니다.

실측 노트북 실행 시간은 **24.26초**입니다(커널 시작을 포함한 시간은 `results/reproduction/package_validation.json` 참조). 최종 테스트 2,000개에서의 차이는 다음과 같습니다. 차이는 노트북 − 원본이며 MAPE 차이 단위는 %p입니다.

| 출력 | 모델 | ΔR² | ΔRMSE | ΔMAPE (%p) |
|---|---|---:|---:|---:|
| 열침입 [W] | GPR | 4.330e-15 | -4.565e-10 | 1.417e-09 |
| 열침입 [W] | MLP | 0.000e+00 | 5.551e-17 | 0.000e+00 |
| 열적 압력 [Pa] | GPR | 1.917e-13 | -1.185e-07 | -3.094e-10 |
| 열적 압력 [Pa] | MLP | -1.110e-16 | 0.000e+00 | 0.000e+00 |

GPR의 미세한 차이는 커널 복원과 부동소수점 연산 경로의 차이로 추정합니다. MLP는 표시 정밀도에서 일치합니다. 수력 보정의 false safe/unsafe 개수와 S1 A·B·C의 설계값도 원본과 일치하며, A·C 단열 두께의 차이는 4×10⁻¹⁵ mm 미만입니다.

원본 코드는 수정하지 않고 복사했습니다. 제출본의 파일 무결성과 실행 출력을 확인하는 테스트만 추가했습니다. 노트북용 데이터는 `data/`에, 원본 코드의 테스트·벤치마크용 입력 사본은 `full_pipeline/data/`에 담았습니다. 가상환경·캐시·기존 outputs·Git 및 개인 설정 폴더는 제외했습니다.

전체 재현은 약 3시간이며 아래 명령으로 실행합니다. `run_full10k.sh`는 `full_pipeline/.venv`를 사용하므로 별도 환경을 만듭니다. 실행은 백그라운드로 진행되며 `run.log`에서 확인할 수 있습니다. 아래 명령은 새 출력 폴더를 대상으로 하며 제출된 결과를 덮어쓰지 않습니다.

```bash
cd full_pipeline
python3.10 -m venv .venv
.venv/bin/pip install -r ../results/reproduction/requirements-notebook.txt
.venv/bin/python -m unittest discover -s tests -v
bash run_full10k.sh ../data/dataset_ai_10k.csv --out outputs/reproduction
# 진행 확인
tail -f outputs/reproduction/run.log
```

전체 재현에는 개발 데이터의 모델 비교·튜닝·학습 곡선, 선정 완료 후 최종 테스트 1회, S1–S3 최적화가 포함됩니다. `finalize` 시작 기록이 있으면 중단·실패했어도 다시 실행하지 않습니다. 재개할 때는 같은 명령과 같은 출력 폴더를 사용합니다. 다항 Ridge·RF·MLP는 각각 출력별 50후보, GPR은 각각 6후보 전수 탐색을 했습니다. 선형과 물리식은 튜닝하지 않았습니다. F2는 개발 데이터 안의 고정 검증 RMSE입니다. 원본의 전체 실행은 약 2.87시간이었습니다. 파일럿 비교 보조표는 `results/`의 사본으로 제공하며, 파일럿 outputs를 제외한 전체 재현에서는 해당 보조표가 미제공으로 표시됩니다.

복사본의 `requirements-tested.txt`는 원본 문서에 남아 있는 별도 환경 기록입니다. 이 제출본의 실측 환경은 `results/reproduction/environment.json`과 `requirements-notebook.txt`를 기준으로 합니다.

| 실행 환경 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS, Linux x86_64 |
| Python | 3.10.12 |
| NumPy / pandas / SciPy | 2.2.6 / 2.3.3 / 1.15.3 |
| scikit-learn / matplotlib | 1.7.2 / 3.10.9 |
| joblib / pymoo | 1.6.0 / 0.6.2 |
| RAM | 15.21 GiB, swap 없음 |
| 무작위 seed / BLAS | 2026 / 1 thread |

최종 테스트 2,000개의 원본 수치입니다. RMSE 단위는 열침입 W, 열적 압력 Pa입니다.

| 출력 | 모델 | R² | RMSE | MAPE (%) |
|---|---|---:|---:|---:|
| 열침입 | GPR | 0.999999991 | 0.001965605 | 0.001087416 |
| 열침입 | MLP | 0.999490291 | 0.467578881 | 0.871363695 |
| 열적 압력 | GPR | 0.999981371 | 23.050065902 | 0.209912607 |
| 열적 압력 | MLP | 0.995340812 | 364.533115757 | 1.921484304 |

수력 보정은 같은 최종 테스트에서 전체 압력 오차와 제약 판정을 개선했습니다. 경계 구간은 실제 전체 압력 38.3–58.3 kPa의 134개이며, 안전 제약은 48.3 kPa 이하입니다. false safe는 실제 초과를 안전으로, false unsafe는 실제 안전을 초과로 예측한 개수입니다.

| 수력 보정 | 전체 압력 MAPE (%) | 경계 MAPE (%) | false safe | false unsafe |
|---|---:|---:|---:|---:|
| 1.8 고정 | 0.687533271 | 0.951264697 | 1 | 4 |
| GPR 비율 | 0.073793844 | 0.089401457 | 0 | 0 |

원본 수치는 `results/T_final_test_metrics.csv`, `results/T_hydraulic_effect.csv`에 있습니다. 재현 차이는 `results/reproduction/notebook_*comparison.csv`에 따로 저장했습니다. 원본 경고는 `results/T_warnings.csv`에 보존했습니다.

데이터는 1D 솔버 출력이며 실험값이 아닙니다. GPR은 계산량 때문에 개발 데이터의 3,000개 부분표본을 사용했습니다. 대리모델 Pareto와 A·B·C는 설계 후보이며, 1D 재검증 결과는 별도입니다. 물리 특성 유무 비교표는 파일럿 개발 데이터의 결과입니다. 1만 개 데이터에서 특성을 제거한 모델을 다시 학습한 비교가 아닙니다.

문제 정의, 실험 설계(데이터 분할·평가 기준·비교 모델·하이브리드 구조·사전 설정), 결과 검증과 해석은 팀이 수행했습니다.일부 파일의 구현에는 AI 코딩 도구(Codex)의 도움을 받았습니다: [artifact_state.py, full10k_runner.py(실행상태 관리 파트)]
