"""테스트셋을 남겨 두고 개발 자료로 학습 곡선과 예상 실행 시간을 확인함."""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from training_support import SEED, sampled_parameters


def batched_predict(model, X, batch_size=1000):
    return np.concatenate([model.predict(X.iloc[start:start+batch_size])
                           for start in range(0, len(X), batch_size)])


def curve_plan(n_rows, labels, gpr_max_rows):
    pool, validation = train_test_split(np.arange(n_rows), test_size=.2,
                                        stratify=labels, random_state=SEED)
    sizes = sorted(set([n for n in [25, 50, 100, 200, 400, 800, 1600, 3200, n_rows]
                        if n <= n_rows] + [len(pool)]))
    rng = np.random.default_rng(SEED)
    plan = []
    for n in sizes:
        if n > len(pool):
            plan.append((n, None, None, 'skipped_validation_holdout'))
        else:
            for repeat in range(5):
                plan.append((n, repeat, rng.choice(pool, n, replace=False), 'evaluated'))
    return pool, validation, plan


def learning_rows(best, X, y, target, name, cap, validation, plan, metric_function, warning_records):
    rows = []
    for n, repeat, ids, status in plan:
        if name == 'gpr' and n > cap:
            status = 'skipped_gpr_max_rows'
        if name == 'ann_ref' and n > best.max_rows:
            status = 'skipped_ann_n_train'
        row = dict(target=target, model=name, n=n, repeat=repeat,
                   validation_n=len(validation), status=status)
        if status == 'evaluated':
            model = clone(best)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(X.iloc[ids], y[ids])
            warning_records.extend(f'{target}/{name}/learning/n{n}/repeat{repeat}: {w.message}' for w in captured)
            row.update(metric_function(y[validation], model.predict(X.iloc[validation])))
        rows.append(row)
    return rows


def fit_validation(estimator,X,y,a,b):
    model=clone(estimator)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(X.iloc[a],y[a])
    fit_seconds=time.perf_counter()-started
    return dict(prediction=model.predict(X.iloc[b]),fit_seconds=fit_seconds,
                fit_rows=len(model.sample_indices_) if hasattr(model,'sample_indices_') else len(a),
                warnings=[str(w.message) for w in captured])


def estimate_runtime(models_by_target, X, targets, folds, cfg, learning=False, labels=None):
    """후보 하나와 fold 하나의 실측 시간으로 전체 실행 시간을 추정함.
    단독 학습 기준이라 동시 실행 시 CPU·메모리 경합은 반영하지 못함.
    """
    import math
    from execution_plan import resource_plan,candidate_parameters,search_wall_seconds
    a,b=folds[0]
    rows=[]
    for target,models in models_by_target.items():
        y=targets[target]
        for name,(est,space) in models.items():
            sampled=candidate_parameters(name,space,cfg['budget'],cfg.get('exhaustive_gpr',False))
            evaluations=len(sampled)
            plan=resource_plan(cfg,max_tasks=evaluations*len(folds))
            model=clone(est).set_params(**sampled[0])
            started=time.perf_counter();model.fit(X.iloc[a],y[a]);fit_seconds=time.perf_counter()-started
            started=time.perf_counter();model.predict(X.iloc[b]);validation_seconds=time.perf_counter()-started
            probe=pd.concat([X.iloc[b]]*int(np.ceil(1000/len(b))),ignore_index=True).iloc[:1000]
            started=time.perf_counter();model.predict(probe);prediction_1k_seconds=time.perf_counter()-started
            n_fold=min(len(a),cfg['gpr_max_rows']) if name=='gpr' else len(a)
            n_refit=min(len(X),cfg['gpr_max_rows']) if name=='gpr' else len(X)
            exponent=3 if name=='gpr' else 1
            jobs=plan['effective_jobs']
            tuning_seconds=0.0 if target=='hyd_ratio' else search_wall_seconds(fit_seconds,validation_seconds,evaluations,len(folds),jobs)
            refit_seconds=fit_seconds*(n_refit/n_fold)**exponent
            reporting_seconds=math.ceil(len(folds)/jobs)*(fit_seconds+validation_seconds)
            curve_seconds=0.0
            if learning and target!='hyd_ratio':
                _,validation,curve=curve_plan(len(X),labels,cfg['gpr_max_rows'])
                for n,_,_,status in curve:
                    if status=='evaluated' and not (name=='gpr' and n>cfg['gpr_max_rows']) and not (name=='ann_ref' and n>est.max_rows):
                        curve_seconds+=fit_seconds*(n/n_fold)**exponent+validation_seconds*len(validation)/len(b)
            predict_seconds=0.0 if target=='hyd_ratio' else prediction_1k_seconds*30
            total=tuning_seconds+refit_seconds+reporting_seconds+curve_seconds+predict_seconds
            rows.append(dict(target=target,model=name,fit_rows=n_fold,
                one_fold_fit_seconds=fit_seconds,one_fold_predict_seconds=validation_seconds,
                candidates=evaluations,tunable=bool(space),
                budget_sensitive=bool(space) and not (name=='gpr' and cfg.get('exhaustive_gpr',False)),
                cv_folds=len(folds),search_workers=jobs,blas_threads=plan['blas_threads'],
                estimated_search_seconds=tuning_seconds,estimated_refit_seconds=refit_seconds,
                estimated_reporting_cv_seconds=reporting_seconds,estimated_learning_seconds=curve_seconds,
                estimated_benchmark_seconds=predict_seconds,estimated_develop_seconds=total))
    result=pd.DataFrame(rows)
    print(result.to_string(index=False,float_format=lambda value:f'{value:.3f}'),flush=True)
    print(f'Estimated total develop: {result.estimated_develop_seconds.sum():.1f} seconds '
          '(wall-clock task waves; solo-fit timing, concurrency contention and I/O/plots excluded).',flush=True)
    return result
