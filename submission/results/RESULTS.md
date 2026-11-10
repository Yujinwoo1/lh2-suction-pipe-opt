# LH2 결과 요약

실행 구분: 합본 실행
개발 8000행 / 최종 테스트 2000행, CV 5개 fold.
선정 모델: {'Q_leak_W': 'gpr', 'dp_thermal_Pa': 'gpr'}
저장된 고유 경고 기록: 286개 (manifest 및 T_warnings.csv 참조).

T1은 개발 CV 평균·표준편차, 실제 후보 수, 학습·예측 시간을 담습니다. 최종 테스트 성능은 별도 T_final_test_metrics.csv입니다.
F2는 개발 검증 RMSE이며 최종 테스트셋을 학습 곡선에 사용하지 않았습니다.
물리 특성 효과는 기존 파일럿의 동일 분할 비교이며, 합본에서 특성을 뺀 모델을 새로 학습한 결과가 아닙니다.
대리모델 Pareto와 재검증 후보는 솔버 검증 결과가 아닙니다.

## 생성 파일

- F1_parity.png
- F2_learning_curve_records.csv
- F2_learning_curve_statistics.csv
- F2_learning_curves.png
- F3_mlp_history.csv
- F3_mlp_loss.png
- F4.png
- F4_const.png
- F4_gpr.png
- T1_model_comparison.csv
- T2_hyperparameters.csv
- T3_candidates.csv
- T3_const.csv
- T3_gpr.csv
- T_final_test_metrics.csv
- T_hydraulic_effect.csv
- T_physics_feature_effect.csv
- T_pilot_full_model_errors.csv
- T_pilot_full_point_shift_summary.csv
- T_pilot_full_point_shifts.csv
- T_warnings.csv
- const_hv_summary.csv
- const_verification_inputs.csv
- const_verification_points.csv
- gpr_hv_summary.csv
- gpr_verification_inputs.csv
- gpr_verification_points.csv
- report_sources.json
