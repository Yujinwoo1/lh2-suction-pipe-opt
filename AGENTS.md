# 작업 규칙

- 모든 무작위 과정의 seed는 2026이다.
- 테스트셋은 모델 선정 후 `finalize`에서 한 번만 연다. 결과를 보고 모델, 하이퍼파라미터, 식을 바꾸지 않는다. 기존 `outputs/pilot/`을 덮어쓰지 않는다.
- 입력은 `lh2_pipeline.FEATURES` 6개만 사용한다. ID, 정답 출력, 기화 라벨은 입력에서 제외한다.
- 스케일러와 다항 변환은 Pipeline 내부에서 학습 fold에만 fit한다.
- `data/`, `outputs/`, `*.joblib`은 커밋하지 않는다.
- 정답 없는 `cases_10k.csv`는 학습에 사용하지 않으며 예측으로 정답을 채우지 않는다.
- 실제 튜닝 횟수와 결과를 그대로 보고한다. 대리모델 Pareto는 솔버 검증 결과가 아니다. 수렴 경고를 숨기지 않는다.
- 작업마다 테스트를 추가하거나 갱신하고 전체 unittest 통과 후 한국어 메시지로 별도 커밋한다.
- `docs/codex_tasks.md`의 우선순위를 따른다. 현재 요청은 P0-1 완료 후 멈추고 보고하는 것이다.

## 실행 명령

```bash
python -m unittest discover -s tests -v
python lh2_pipeline.py develop --learning-curves
python lh2_pipeline.py finalize
python optimize.py run --models outputs/pilot
python optimize.py compare --points outputs/opt_pilot/verification_points.csv --solver solver_results.csv
```

`develop`는 개발 자료만 사용한다. `finalize`는 선정 완료 후 단 한 번 실행하며 최적화 작업에서 다시 실행하지 않는다.
