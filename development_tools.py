"""Development-only learning curves and approximate runtime estimates."""
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


def estimate_runtime(models_by_target, X, targets, folds, cfg, learning=False, labels=None):
    """One candidate / one development fold fit per target-model, no holdout use.

    First-candidate timings extrapolate linearly in n except cubic GPR fitting.
    Search alone is parallel; repeated reporting CV/curves/refit are serial.
    This cannot model different candidate costs, convergence, RAM or contention.
    """
    a, b = folds[0]
    rows = []
    for target, models in models_by_target.items():
        y = targets[target]
        for name, (est, space) in models.items():
            params = sampled_parameters(space, cfg['budget'])[0]
            model = clone(est).set_params(**params)
            started = time.perf_counter()
            # Warnings stay visible in this read-only estimation mode.
            model.fit(X.iloc[a], y[a])
            fit_seconds = time.perf_counter()-started
            started = time.perf_counter()
            model.predict(X.iloc[b])
            validation_seconds = time.perf_counter()-started
            probe = pd.concat([X.iloc[b]]*int(np.ceil(1000/len(b))), ignore_index=True).iloc[:1000]
            started = time.perf_counter()
            model.predict(probe)
            prediction_1k_seconds = time.perf_counter()-started
            n_fold = min(len(a), cfg['gpr_max_rows']) if name == 'gpr' else len(a)
            n_refit = min(len(X), cfg['gpr_max_rows']) if name == 'gpr' else len(X)
            exponent = 3 if name == 'gpr' else 1
            evaluations = cfg['budget'] if space else 1
            work = evaluations*len(folds)
            workers = min(joblib.effective_n_jobs(cfg['n_jobs']), work)
            tuning_seconds = (fit_seconds+validation_seconds)*work/workers
            refit_seconds = fit_seconds*(n_refit/n_fold)**exponent
            reporting_seconds = (fit_seconds+validation_seconds)*len(folds)
            curve_seconds = 0.0
            if learning:
                _, validation, plan = curve_plan(len(X), labels, cfg['gpr_max_rows'])
                for n, _, _, status in plan:
                    if status == 'evaluated' and not (name == 'gpr' and n > cfg['gpr_max_rows']):
                        curve_seconds += fit_seconds*(n/n_fold)**exponent + validation_seconds*len(validation)/len(b)
            predict_seconds = prediction_1k_seconds*30
            total = tuning_seconds+refit_seconds+reporting_seconds+curve_seconds+predict_seconds
            rows.append(dict(target=target, model=name, fit_rows=n_fold,
                             one_fold_fit_seconds=fit_seconds, one_fold_predict_seconds=validation_seconds,
                             candidates=evaluations, cv_folds=len(folds), search_workers=workers,
                             estimated_search_seconds=tuning_seconds,
                             estimated_refit_seconds=refit_seconds,
                             estimated_reporting_cv_seconds=reporting_seconds,
                             estimated_learning_seconds=curve_seconds,
                             estimated_benchmark_seconds=predict_seconds,
                             estimated_develop_seconds=total))
    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda value: f'{value:.3f}'), flush=True)
    print(f'Estimated total develop: {result.estimated_develop_seconds.sum():.1f} seconds '
          '(rough first-candidate estimate; I/O/plots, varying candidate costs and memory contention excluded).', flush=True)
    return result
