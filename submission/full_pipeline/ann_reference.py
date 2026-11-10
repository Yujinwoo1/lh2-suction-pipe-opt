"""깊은 MLP를 개발 CV 비교군에 추가한다. [8,9]의 논문 구조를 그대로 재현한 것은 아니다."""
import json
from pathlib import Path
import numpy as np
from scipy.special import exp10
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from training_support import SubsampledTargetRegressor

FIELDS=['hidden_layer_sizes','activation','solver','learning_rate','max_iter','n_train','source']
UNREPORTED='논문 미기재, 기본값 사용'


def load_ann_reference(path):
    spec=json.loads(Path(path).read_text(encoding='utf-8'))
    missing=[key for key in FIELDS if spec.get(key) is None or spec.get(key)=='' or spec.get(key)==[]]
    if missing:
        return spec,None,'ann_ref skipped: unfilled fields '+', '.join(missing)
    kwargs={key:value for key,value in spec.items() if key in FIELDS[:5] and value!=UNREPORTED}
    if 'hidden_layer_sizes' in kwargs:
        sizes=kwargs['hidden_layer_sizes']
        if not isinstance(sizes,list) or not sizes or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in sizes):
            raise ValueError('ann_ref hidden_layer_sizes must be a nonempty list of positive integers.')
        kwargs['hidden_layer_sizes']=tuple(sizes)
    n=spec['n_train']
    if n!=UNREPORTED and (isinstance(n,bool) or not isinstance(n,int) or n<2):
        raise ValueError('ann_ref n_train must be integer >=2 or the explicit unreported marker.')
    # Exactly the paper-provided settings; no guessed early stopping or target
    # labels as inputs. Six original features, no optional physics feature.
    model=SubsampledTargetRegressor(regressor=Pipeline([
        ('scale',StandardScaler()),('model',MLPRegressor(random_state=2026,**kwargs))]),
        func=np.log10,inverse_func=exp10,max_rows=n if n!=UNREPORTED else 2**31-1)
    return spec,model,'ann_ref enabled: fixed structure, zero tuning; original six inputs. n_train is capped to available fold training rows and actual counts recorded.'


def dnn_estimator(parameters):
    """기존 MLP의 입력 변환을 유지해 깊이에 따른 차이를 비교한다.
    논문 구조의 재현이 아니라 깊은 MLP를 추가한 비교군이다.
    """
    from sklearn.base import clone
    from lh2_pipeline import candidates
    estimator = clone(candidates(quick=False, physics_feature=True)['mlp'][0])
    return estimator.set_params(regressor__model__activation='relu',
        regressor__model__solver='adam', **parameters)


def dnn_candidates():
    """성능을 보기 전에 후보를 고정해 결과에 맞춘 탐색 변경을 막는다.
    비용 측정에는 가장 넓은 구성을 먼저 사용한다.
    """
    from training_support import sampled_parameters
    prefix = 'regressor__model__'
    space = {
        prefix+'hidden_layer_sizes': [(64, 64, 64), (128, 128, 128),
            (256, 256, 256), (256, 128, 64), (64, 64, 64, 64),
            (128, 128, 128, 128), (256, 256, 256, 256), (256, 128, 128, 64)],
        prefix+'learning_rate_init': [.0003, .001, .003],
        prefix+'alpha': [.0001, .001, .01],
        prefix+'batch_size': [64, 128, 256]}
    probe = {prefix+'hidden_layer_sizes': (256, 256, 256, 256),
             prefix+'learning_rate_init': .001, prefix+'alpha': .0001,
             prefix+'batch_size': 64}
    sampled = sampled_parameters(space, 50)
    return [probe] + [item for item in sampled if item != probe][:49]


def development_only_data(data_path, reference, fold_path):
    """분할 복원에는 ID와 기화 라벨만 읽고, 정답은 개발 행만 읽는다.
    기존 분할 함수로 복원한 순서와 저장된 fold 인덱스를 모두 대조한다.
    """
    import hashlib
    import pandas as pd
    from lh2_pipeline import FEATURES, TARGETS, train_test_split, RepeatedKFold
    from training_support import sample_indices
    manifest = json.loads((reference/'manifest.json').read_text())
    cfg = manifest['effective_config']
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Data hash differs from the original T1 run.')
    if (manifest['seed'], cfg['cv_splits'], cfg['cv_repeats'],
            manifest['train_rows'], manifest['test_rows']) != (2026, 5, 1, 8000, 2000):
        raise ValueError('Expected the original 8000/2000 split and five CV folds.')
    metadata = pd.read_csv(data_path, usecols=['case_id', 'vaporized'])
    train, test = train_test_split(np.arange(len(metadata)), test_size=cfg['test_size'],
        random_state=2026, stratify=metadata[cfg['stratify']])
    split = pd.read_csv(reference/'split.csv')
    development_ids = metadata.iloc[train].case_id.to_numpy()
    test_ids = set(split.loc[split.split == 'test', 'case_id'])
    if (set(development_ids) != set(split.loc[split.split == 'development', 'case_id'])
            or set(metadata.iloc[test].case_id) != test_ids
            or set(development_ids) & test_ids):
        raise ValueError('Saved development and test IDs do not match.')
    # skiprows로 테스트 행을 파싱 전에 제외한다. 타깃 배열에는 개발 정답만 들어간다.
    positions = set(train.tolist())
    columns = ['case_id', *[name for name in FEATURES if name != 'log10_k_MLI'],
               'k_MLI_W_mK', *TARGETS]
    development = pd.read_csv(data_path, usecols=columns,
        skiprows=lambda row: row > 0 and row-1 not in positions)
    development = development.set_index('case_id').loc[development_ids].reset_index()
    development['log10_k_MLI'] = np.log10(development.k_MLI_W_mK)
    if len(development) != 8000 or not development.case_id.is_unique:
        raise ValueError('Expected 8000 unique development rows.')
    if not np.isfinite(development[FEATURES+TARGETS].to_numpy()).all():
        raise ValueError('Development data must be finite.')
    if (development[TARGETS] <= 0).any().any():
        raise ValueError('Targets must be positive for log10 training.')
    # 부분표본 ID도 대조해 원래 개발 행 순서까지 복원했는지 확인한다.
    samples = pd.read_csv(reference/'gpr_subsamples.csv')
    expected_sample = development_ids[sample_indices(len(train), cfg['gpr_max_rows'])]
    for target in TARGETS:
        saved = samples.loc[(samples.stage == 'refit') & (samples.target == target), 'case_id']
        np.testing.assert_array_equal(expected_sample, saved.to_numpy())
    folds = list(RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'],
        random_state=2026).split(development[FEATURES]))
    with np.load(fold_path) as saved:
        if len(saved.files) != 2*len(folds):
            raise ValueError('Saved CV fold count differs.')
        for index, (a, b) in enumerate(folds):
            np.testing.assert_array_equal(a, saved[f'{index}_train'])
            np.testing.assert_array_equal(b, saved[f'{index}_val'])
            if set(development.iloc[np.r_[a, b]].case_id) & test_ids:
                raise ValueError('Holdout ID entered a development fold.')
    return development, folds


def dnn_budget(probe_seconds, maximum_seconds=5400, requested=50):
    """두 타깃의 실측 비용으로 같은 후보 수를 정해 탐색 기회를 맞춘다.
    후보별 수렴 시간 차이를 고려해 25%의 시간 여유를 둔다.
    """
    if not np.isfinite(probe_seconds) or probe_seconds <= 0:
        raise ValueError('Probe duration must be finite and positive.')
    return max(1, min(requested, int(maximum_seconds/(1.25*probe_seconds))))


def evaluate_dnn_candidate(X, y, folds, parameters, target, candidate):
    """각 fold에서 Pipeline을 새로 학습해 검증 자료의 전처리 누출을 막는다.
    선정 기준과 원단위 평가 지표는 기존 T1 함수를 그대로 사용한다.
    """
    import time
    import warnings
    from sklearn.base import clone
    from lh2_pipeline import metrics, log_rmse_score
    estimator = dnn_estimator(parameters)
    rows = []
    for fold, (a, b) in enumerate(folds):
        fitted = clone(estimator)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fitted.fit(X.iloc[a], y[a])
        fit_seconds = time.perf_counter()-started
        # 경고를 파일과 터미널에 함께 남겨 미수렴을 결과에서 숨기지 않는다.
        for warning in captured:
            warnings.warn_explicit(str(warning.message), warning.category,
                filename=Path(warning.filename).name, lineno=warning.lineno)
        started = time.perf_counter()
        prediction = fitted.predict(X.iloc[b])
        predict_seconds = time.perf_counter()-started
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError('DNN prediction must be finite and positive.')
        score = -log_rmse_score(fitted, X.iloc[b], y[b])
        row = dict(target=target, model='dnn', candidate=candidate, fold=fold,
            fit_rows=len(a), validation_rows=len(b), fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            iterations=fitted.regressor_.named_steps['model'].n_iter_,
            selection_log10_rmse=score,
            warnings=json.dumps([str(w.message) for w in captured], ensure_ascii=False),
            **metrics(y[b], prediction))
        rows.append(row)
        print(f'DNN {target} candidate {candidate+1} fold {fold+1}/5: '
              f'{fit_seconds:.2f}s, MAPE={row["mape_pct"]:.6f}%', flush=True)
    return rows


def dnn_summary(rows, parameters, target, candidate_count, elapsed_seconds):
    """CV로 고른 설정의 fold 평균·표준편차를 기존 T1 형식에 맞춘다.
    같은 CV로 튜닝과 비교를 하므로 독립적인 일반화 성능 추정은 아니다.
    """
    import pandas as pd
    table = pd.DataFrame(rows)
    scores = table.groupby('candidate').selection_log10_rmse.mean()
    winner = int(scores.sort_index().idxmin())
    group = table.loc[table.candidate == winner]
    summary = dict(target=target, model='dnn', actual_candidates=candidate_count,
        selected_candidate=winner, selected_hyperparameters=json.dumps(parameters[winner],
            ensure_ascii=False, sort_keys=True), execution_seconds=elapsed_seconds,
        tuning_seconds=elapsed_seconds, predict_10k_seconds_median=np.nan,
        final_refit_seconds=np.nan, unit='W' if target == 'Q_leak_W' else 'Pa',
        evaluation_scope='development CV, used for tuning and comparison; no final holdout',
        seed=2026, cv_folds=5, development_rows=8000, final_holdout_used=False,
        activation='relu', solver='adam', max_iter=3000, early_stopping=True,
        physics_feature=True, blas_threads=1)
    for metric in ['r2', 'rmse', 'mape_pct', 'log10_rmse', 'log10_r2', 'fit_seconds']:
        summary[metric+'_mean'] = group[metric].mean()
        summary[metric+'_std'] = group[metric].std(ddof=1)
    for metric in ['r2', 'rmse', 'mape_pct']:
        summary[metric+'_mean_pm_std'] = f'{summary[metric+"_mean"]:.6g} ± {summary[metric+"_std"]:.6g}'
    return summary


def append_dnn_table(original_path, output_path, rows):
    """기존 행을 다시 계산하거나 반올림하지 않고 CSV 끝에 DNN 행만 추가한다.
    원본 바이트를 보존해 기존 수치가 바뀌는 것을 막는다.
    """
    import csv
    import io
    original = original_path.read_bytes()
    columns = next(csv.reader(io.StringIO(original.decode('utf-8'))))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    for row in rows:
        writer.writerow(row)
    with output_path.open('xb') as handle:
        handle.write(original)
        if original and not original.endswith(b'\n'):
            handle.write(b'\n')
        handle.write(buffer.getvalue().encode('utf-8'))


def run_dnn_comparison(submission):
    """최종 테스트를 열지 않고 기존 개발 분할에 DNN 비교 행을 추가한다.
    첫 후보의 두 타깃 5-fold 측정 후 시간 예산으로 후보 수를 확정한다.
    """
    import time
    import hashlib
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from lh2_pipeline import FEATURES, TARGETS
    results = submission/'results'
    reference = results/'reproduction'
    audit = results/'dnn_audit'
    audit.mkdir(exist_ok=True)
    detailed = results/'T1b_dnn_cv.csv'
    combined = results/'T1_with_dnn.csv'
    plan_path = audit/'dnn_plan.json'
    if any(path.exists() for path in [detailed, combined, plan_path]):
        raise FileExistsError('DNN run exists; results must not be overwritten.')
    development, folds = development_only_data(submission/'data/dataset_ai_10k.csv',
        reference, audit/'reference_cv_indices.npz')
    parameters = dnn_candidates()
    plan = dict(seed=2026, requested_candidates=50, maximum_seconds=5400,
        timing_safety_factor=1.25, selection_metric='minimum mean CV log10 RMSE',
        interpretation='DNN model-family comparison inspired by [9], not paper reproduction',
        candidates=parameters, final_holdout_used=False,
        data_sha256=hashlib.sha256((submission/'data/dataset_ai_10k.csv').read_bytes()).hexdigest(),
        fold_sha256=hashlib.sha256((audit/'reference_cv_indices.npz').read_bytes()).hexdigest())
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    development[['case_id']].to_csv(audit/'development_ids.csv', index=False)
    fold_ids = [dict(fold=index, part=part, case_id=case_id)
        for index, pair in enumerate(folds) for part, positions in zip(['train', 'validation'], pair)
        for case_id in development.iloc[positions].case_id]
    pd.DataFrame(fold_ids).to_csv(audit/'cv_case_ids.csv', index=False)
    X = development[FEATURES]
    records = {target: [] for target in TARGETS}
    durations = {target: 0.0 for target in TARGETS}
    started_all = time.perf_counter()
    with threadpool_limits(limits=1):
        for target in TARGETS:
            started = time.perf_counter()
            records[target].extend(evaluate_dnn_candidate(X, development[target].to_numpy(),
                folds, parameters[0], target, 0))
            durations[target] += time.perf_counter()-started
        probe_seconds = time.perf_counter()-started_all
        budget = dnn_budget(probe_seconds)
        plan.update(probe_seconds_two_targets_5fold=probe_seconds,
            probe_seconds_per_target=dict(durations), actual_candidates_per_target=budget,
            estimated_50_candidates_seconds=50*probe_seconds,
            guarded_estimated_50_candidates_seconds=1.25*50*probe_seconds,
            budget_reason='same prefix of preregistered candidates, reduced only by measured runtime')
        (audit/'dnn_timing_decision.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
        print(f'Probe {probe_seconds:.2f}s; 50-candidate estimate {50*probe_seconds/60:.2f} min; '
              f'budget {budget} per target', flush=True)
        pd.DataFrame([row for rows in records.values() for row in rows]).to_csv(audit/'dnn_cv_folds.csv', index=False)
        for candidate in range(1, budget):
            for target in TARGETS:
                started = time.perf_counter()
                records[target].extend(evaluate_dnn_candidate(X, development[target].to_numpy(),
                    folds, parameters[candidate], target, candidate))
                durations[target] += time.perf_counter()-started
                pd.DataFrame([row for rows in records.values() for row in rows]).to_csv(audit/'dnn_cv_folds.csv', index=False)
    summaries = [dnn_summary(records[target], parameters, target, budget, durations[target]) for target in TARGETS]
    for row in summaries:
        row.update(requested_candidates=50, probe_seconds_two_targets_5fold=probe_seconds,
            estimated_50_candidates_seconds=50*probe_seconds,
            guarded_estimated_50_candidates_seconds=1.25*50*probe_seconds,
            total_execution_seconds=time.perf_counter()-started_all,
            budget_limit_seconds=5400, time_budget_met=time.perf_counter()-started_all <= 5400)
    pd.DataFrame(summaries).to_csv(detailed, index=False, mode='x')
    append_dnn_table(results/'T1_model_comparison.csv', combined, summaries)
    print(pd.DataFrame(summaries)[['target', 'actual_candidates', 'mape_pct_mean', 'rmse_mean']].to_string(index=False), flush=True)
    return summaries


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Development-only DNN comparison; no final holdout.')
    parser.add_argument('--submission', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_dnn_comparison(args.submission)
