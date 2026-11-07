"""Write a concise Korean report from completed experiment outputs."""
import json
from pathlib import Path
import pandas as pd


def summarize(out=Path('outputs/pilot')):
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((out/'data_audit.json').read_text(encoding='utf-8'))
    pressure = json.loads((out/'final_pressure_metrics.json').read_text(encoding='utf-8'))
    test = pd.read_csv(out/'final_test_metrics.csv')
    cv = pd.read_csv(out/'cv_summary.csv')
    lines = ['# LH2 200개 파일럿 실행 결과', '',
        '## 실험 설계', '',
        f"- 개발 {manifest['train_rows']}개 / 최종 테스트 {manifest['test_rows']}개, seed {manifest['seed']}.",
        '- 5-fold × 3회 반복 CV, 튜닝 모델당 6개 후보. 모델 선택 기준은 log10 RMSE.',
        '- 선형회귀, 다항 Ridge, RF, GPR, MLP 비교. 각 출력에 별도 모델 학습.',
        '- 학습 곡선은 개발 데이터만 사용. 최종 테스트는 모델 선택 이후 1회 평가.', '',
        '## 최종 테스트 결과', '',
        '| 출력 | 모델 | 원단위 R² | RMSE | MAPE (%) |',
        '|---|---|---:|---:|---:|']
    for _, row in test.iterrows():
        selected = ' (선정)' if manifest['selected'][row.target] == row.model else ' (사전 비교 대상)'
        lines.append(f'| {row.target} | {row.model}{selected} | {row.r2:.6f} | {row.rmse:.6g} | {row.mape_pct:.3f} |')
    lines += ['', 'RMSE 단위는 열침입 W, 압력 열 성분 Pa입니다. 테스트 성능으로 모델을 다시 고르지 않았습니다.', '',
        '## 하이브리드 총압력 평가', '',
        f"- 전체 R²: {pressure['r2']:.6f}, RMSE: {pressure['rmse']:.3f} Pa, MAPE: {pressure['mape_pct']:.3f}%.",
        f"- 실제 제약 위반 {pressure['actual_violations']}개 중 false safe {pressure['false_safe_count']}개.",
        f"- false unsafe {pressure['false_unsafe_count']}개, 전체 판정 일치율 {100*pressure['constraint_agreement']:.1f}%.",
        f"- 48.3 kPa ± 10 kPa 경계 구간: {pressure['boundary_n']}개."]
    if pressure['boundary_metrics']:
        b = pressure['boundary_metrics']
        lines.append(f"- 경계 RMSE: {b['rmse']:.3f} Pa, MAPE: {b['mape_pct']:.3f}%.")
    lines += ['', '## 개발 CV 비교', '',
        '아래 값은 튜닝과 선택에 사용한 CV 값입니다. 독립적인 일반화 성능 추정으로 해석하지 않습니다.', '',
        '| 출력 | 모델 | log10 RMSE 평균 ± 표준편차 | 원단위 MAPE 평균 (%) |',
        '|---|---|---:|---:|']
    for _, row in cv.iterrows():
        lines.append(f'| {row.target} | {row.model} | {row.log10_rmse_mean:.5f} ± {row.log10_rmse_std:.5f} | {row.mape_pct_mean:.3f} |')
    lines += ['', '## 해석과 남은 과제', '',
        f"- 전체 200개 감사에서 수력 근사만으로 판정이 바뀐 사례는 {audit['hydraulic_only_constraint_mismatches']}개입니다. AI와 수력 근사오차를 구분해야 합니다.",
        '- 테스트 40개와 적은 경계 표본으로 안전성·외삽 성능을 보증할 수 없습니다.',
        '- 데이터는 실험값이 아닌 1D 솔버 출력입니다.',
        '- 물리 열침입 기준선, 참고논문 재현 ANN, NSGA-II 및 신규 솔버 재검증은 아직 수행하지 않았습니다.',
        f"- 저장된 고유 경고 기록: {len(manifest['warnings'])}개. 상세 내용은 manifest.json을 확인하세요.",
        '- 1만 개 솔버 결과 확보 후 데이터 규모에 맞는 비교 실험을 별도로 설계해야 합니다.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    summarize()
