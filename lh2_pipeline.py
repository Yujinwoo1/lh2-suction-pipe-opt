"""Reproducible pilot surrogate study. Run development before final evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import sklearn
from scipy.special import exp10
from sklearn.base import clone
from physics_baseline import (PhysicsBaseline, physics_heat_leak,
                              PhysicsFeatureTransformer, PHYSICS_FEATURE_NAME)
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from training_support import (SubsampledTargetRegressor, sample_indices, sampled_parameters,
                              read_config, settings, budget_decision)
from development_tools import batched_predict, curve_plan, learning_rows, estimate_runtime
from ann_reference import load_ann_reference
from hydraulics import hydraulic_cv, ratio_estimator, load_ratio_model, predicted_ratio, pressure_metrics

SEED = 2026
FEATURES = ['D_mm', 't_ins_mm', 'f_Hz', 'L_m', 'mdot_kg_h', 'log10_k_MLI']
TARGETS = ['Q_leak_W', 'dp_thermal_Pa']
BOUNDS = {'D_mm': (12, 50), 't_ins_mm': (2, 25), 'f_Hz': (2, 10),
          'L_m': (.5, 3), 'mdot_kg_h': (50, 100), 'k_MLI_W_mK': (10**-4.1, .01)}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def rigid_pressure(df):
    # D: mm -> m; mass flow: kg/h -> kg/s. rho cancels algebraically.
    area = np.pi * (df.D_mm.to_numpy() / 1000)**2 / 4
    return df.L_m.to_numpy() * (np.pi * df.mdot_kg_h.to_numpy() / 3600) * (2*np.pi*df.f_Hz.to_numpy()) / area

def load_data(path):
    df = pd.read_csv(path)
    required = ['case_id', *BOUNDS, *TARGETS, 'dp_hyd_Pa', 'dp_design_Pa', 'dp_rigid_Pa', 'D_o_mm', 'vaporized']
    missing = sorted(set(required) - set(df))
    if missing:
        raise ValueError(f'Missing solver-output columns: {missing}. Input-only cases_10k.csv cannot train a model.')
    source_rows = len(df)
    excluded = df.loc[df.status != 'converged', 'case_id'].tolist() if 'status' in df else []
    if 'status' in df:
        df = df.loc[df.status == 'converged'].copy().reset_index(drop=True)
    df.attrs['source_rows'] = source_rows
    df.attrs['excluded_status_count'] = len(excluded)
    df.attrs['excluded_status_case_ids'] = excluded
    if 8000 <= source_rows <= 12000 and len(df) and abs(df.vaporized.mean()-.06) > .03:
        warnings.warn(f'Large dataset vaporized rate {df.vaporized.mean():.2%} differs from pilot 6% by more than 3 percentage points.')
    if df[required].isna().any().any() or not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite values found.')
    if df.case_id.duplicated().any() or df.duplicated(list(BOUNDS)).any():
        raise ValueError('Duplicate IDs or input conditions: resolve before splitting.')
    if len(df) < 50 or (df[TARGETS] <= 0).any().any():
        raise ValueError('At least 50 rows and strictly positive targets are required.')
    for col, (lo, hi) in BOUNDS.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f'{col} is outside the declared domain.')
    for col, raw in [('log10_k_MLI', 'k_MLI_W_mK'), ('log10_Q_leak', 'Q_leak_W')]:
        computed = np.log10(df[raw])
        if col in df and not np.allclose(df[col], computed, atol=1e-6, rtol=0):
            raise ValueError(f'{col} is inconsistent with {raw}.')
        df[col] = computed
    if not np.allclose(df.D_o_mm, df.D_mm+3+2*df.t_ins_mm, atol=1e-4):
        raise ValueError('Outer diameter identity failed.')
    if not np.allclose(df.dp_design_Pa, df.dp_hyd_Pa+df.dp_thermal_Pa, atol=1, rtol=1e-5):
        raise ValueError('Pressure component identity failed.')
    if not np.array_equal(df.vaporized.to_numpy(), (df.dp_design_Pa > 48300).astype(int).to_numpy()):
        raise ValueError('vaporized labels do not match the supplied threshold definition.')
    if not np.allclose(rigid_pressure(df), df.dp_rigid_Pa, rtol=1e-4, atol=.01):
        raise ValueError('Rigid-pressure formula check failed.')
    return df


def audit(df, out):
    p = df.dp_thermal_Pa.to_numpy() + 1.8*rigid_pressure(df)
    rows = pd.DataFrame({'case_id': df.case_id, 'solver_pressure_Pa': df.dp_design_Pa,
                         'reassembled_with_true_thermal_Pa': p,
                         'relative_error_pct': 100*abs(p/df.dp_design_Pa-1),
                         'constraint_mismatch': (p > 48300) != (df.vaporized == 1)})
    rows.to_csv(out/'physics_audit.csv', index=False)
    heat_error = 100*np.abs(physics_heat_leak(df)/df.Q_leak_W.to_numpy()-1)
    save_json(out/'data_audit.json', {'physics_baseline_mean_relative_error_pct': float(heat_error.mean()),
        'physics_baseline_max_relative_error_pct': float(heat_error.max()),
        'source_rows': df.attrs.get('source_rows', len(df)),
        'excluded_status_count': df.attrs.get('excluded_status_count', 0),
        'excluded_status_case_ids': df.attrs.get('excluded_status_case_ids', []),
        'vaporized_rate': float(df.vaporized.mean()), 'rows': len(df), 'vaporized': int(df.vaporized.sum()),
        'boundary_rows': int(df.dp_design_Pa.between(38300, 58300).sum()),
        'hydraulic_only_mean_error_pct': float(rows.relative_error_pct.mean()),
        'hydraulic_only_max_error_pct': float(rows.relative_error_pct.max()),
        'hydraulic_only_constraint_mismatches': int(rows.constraint_mismatch.sum())})


def candidates(quick=False, target=None, physics_feature=False):
    def wrapped(steps):
        if physics_feature:
            steps = [('physics_feature', PhysicsFeatureTransformer(tuple(FEATURES))), *steps]
        return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log10, inverse_func=exp10)
    dimensions = len(FEATURES) + int(physics_feature)
    prefix = 'regressor__model__'
    models = {
        'linear': (wrapped([('scale', StandardScaler()), ('model', LinearRegression())]), {}),
        'polynomial_ridge': (wrapped([('scale', StandardScaler()), ('poly', PolynomialFeatures(include_bias=False)),
            ('poly_scale', StandardScaler()), ('model', Ridge())]),
            {'regressor__poly__degree': [2, 3, 4], prefix+'alpha': [.01, .1, 1, 10, 100]}),
        'random_forest': (wrapped([('model', RandomForestRegressor(random_state=SEED, n_jobs=1))]),
            {prefix+'n_estimators': [80, 150] if quick else [150, 300], prefix+'max_depth': [None, 8, 16], prefix+'min_samples_leaf': [1, 2, 4]}),
        'gpr': (wrapped([('scale', StandardScaler()), ('model', GaussianProcessRegressor(normalize_y=True, random_state=SEED))]),
            {prefix+'kernel': [ConstantKernel(1, (1e-3, 1e3))*k+WhiteKernel(1e-5, (1e-9, 1e-1))
                 for k in [RBF(np.ones(dimensions)), Matern(np.ones(dimensions), nu=1.5), Matern(np.ones(dimensions), nu=2.5)]],
             prefix+'alpha': [1e-10, 1e-7]}),
        'mlp': (wrapped([('scale', StandardScaler()), ('model', MLPRegressor(random_state=SEED,
                 max_iter=1000 if quick else 3000, early_stopping=True, validation_fraction=.2, n_iter_no_change=40))]),
            {prefix+'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 32)], prefix+'activation': ['tanh', 'relu'],
             prefix+'alpha': [.0001, .001, .01], prefix+'learning_rate_init': [.001, .003]})}
    gpr = models['gpr'][0]
    models['gpr'] = (SubsampledTargetRegressor(regressor=gpr.regressor,
                     func=np.log10, inverse_func=exp10), models['gpr'][1])
    if target == 'Q_leak_W':
        models['physics_baseline'] = (PhysicsBaseline(), {})
    return models


def lambda_power10(y):
    return np.power(10., y)


def log_rmse_score(estimator, X, y):
    prediction = estimator.predict(X)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        return -np.inf
    return -float(np.sqrt(mean_squared_error(np.log10(y), np.log10(prediction))))


def metrics(y, pred):
    return {'r2': r2_score(y, pred), 'rmse': np.sqrt(mean_squared_error(y, pred)),
            'mape_pct': 100*mean_absolute_percentage_error(y, pred),
            'log10_rmse': np.sqrt(mean_squared_error(np.log10(y), np.log10(pred))),
            'log10_r2': r2_score(np.log10(y), np.log10(pred))}


def development(data, out, quick=False, learning=False, physics_feature=None, *,
                config=None, budget=None, cv_repeats=None, n_jobs=None,
                gpr_max_rows=None, estimate_time=False, hyd_model=None, ann_ref_config=None):
    declared, cfg = settings(config, quick=quick, physics_feature=physics_feature,
                             budget=budget, cv_repeats=cv_repeats, n_jobs=n_jobs,
                             gpr_max_rows=gpr_max_rows, hyd_model=hyd_model)
    # Estimation is read-only, including when --out already contains a run.
    if not estimate_time and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory to preserve existing experiments.')
    df = load_data(data)
    train, test = train_test_split(np.arange(len(df)), test_size=cfg['test_size'],
                                   random_state=SEED, stratify=df[cfg['stratify']])
    X = df.iloc[train][FEATURES]
    labels = df.iloc[train].vaporized.to_numpy()
    folds = list(RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'],
                               random_state=SEED).split(X))
    models_by_target = {target: candidates(quick, target=target,
                                          physics_feature=cfg['physics_feature']) for target in TARGETS}
    ann_spec,ann_model,ann_note=load_ann_reference(ann_ref_config or ROOT/'configs/ann_ref.json')
    print(ann_note,flush=True)
    if ann_model is not None:
        for models in models_by_target.values():
            models['ann_ref']=(clone(ann_model),{})
    for models in models_by_target.values():
        if 'gpr' in models:
            models['gpr'][0].set_params(max_rows=cfg['gpr_max_rows'])
    needs_hyd_gpr=cfg['hyd_model']=='gpr' or 'gpr' in declared.get('finalize_pressure_comparisons',[])
    estimate_models=dict(models_by_target)
    estimate_targets={target: df.iloc[train][target].to_numpy() for target in TARGETS}
    if needs_hyd_gpr:
        estimate_models['hyd_ratio']={'gpr':(ratio_estimator(cfg['gpr_max_rows']),{})}
        estimate_targets['hyd_ratio']=df.iloc[train].dp_hyd_Pa.to_numpy()/df.iloc[train].dp_rigid_Pa.to_numpy()
    decision={'applied_budget':cfg['budget'], 'reason':'Explicit CLI override' if budget is not None else 'Quick smoke settings' if quick else 'Configured fixed budget'}
    if estimate_time or (declared.get('budget_rule') and not quick and budget is None):
        estimate=estimate_runtime(estimate_models,X,estimate_targets,folds,cfg,learning=learning,labels=labels)
        if declared.get('budget_rule') and budget is None and not quick:
            decision=budget_decision(estimate,declared['budget_rule'])
            cfg['budget']=decision['applied_budget']
        print('Budget decision: '+json.dumps(decision,ensure_ascii=False),flush=True)
        if estimate_time:
            estimate.attrs['budget_decision']=decision
            return estimate
    out.mkdir(parents=True, exist_ok=True)
    audit(df, out)
    split = pd.DataFrame({'case_id': df.case_id, 'split': np.where(np.isin(np.arange(len(df)), test), 'test', 'development')})
    split.to_csv(out/'split.csv', index=False)
    np.savez(out/'cv_indices.npz', **{f'{i}_{part}': arr for i, pair in enumerate(folds) for part, arr in zip(['train', 'val'], pair)})
    manifest = {'seed': SEED, 'data_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'config': declared, 'effective_config': cfg, 'budget_decision': decision,
        'hyd_model':cfg['hyd_model'], 'ann_ref':{'config':ann_spec,'note':ann_note},
        'features': FEATURES, 'physics_feature': cfg['physics_feature'],
        'derived_features': [PHYSICS_FEATURE_NAME] if cfg['physics_feature'] else [],
        'physics_feature_definition': 'log10(physics_heat_leak(X)/mdot_kg_h), fixed P0-2 formula, inside Pipeline' if cfg['physics_feature'] else None,
        'train_rows': len(train), 'test_rows': len(test), 'quick': quick,
        'cv_folds': len(folds), 'tuning_candidates_per_tunable_model': cfg['budget'],
        'gpr_max_rows': cfg['gpr_max_rows'],
        'gpr_note': 'GPR uses a seed-2026 training-only subsample because of O(n^3) cost; other models use all training rows. Search folds use the same subsets as reporting CV.',
        'selection_metric': 'minimum mean CV log10 RMSE, separately per target',
        'cv_note': 'CV used for tuning and selection; these are not unbiased generalization estimates. Final test is separate.',
        'learning_curves': learning,
        'warnings_note': 'Unique parent-process warning records below; parallel search workers also emit warnings to stderr.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'models': {}, 'warnings': []}
    subsamples = []
    for target, models in models_by_target.items():
        if 'gpr' not in models:
            continue
        for stage, fold, positions in [('refit', -1, np.arange(len(X))),
                                        *[('cv_train', i, a) for i, (a, _) in enumerate(folds)]]:
            selected = positions[sample_indices(len(positions), cfg['gpr_max_rows'])]
            subsamples.extend(dict(target=target, stage=stage, fold=fold, case_id=case_id)
                              for case_id in df.iloc[train[selected]].case_id.tolist())
    pd.DataFrame(subsamples, columns=['target', 'stage', 'fold', 'case_id']).to_csv(out/'gpr_subsamples.csv', index=False)
    curve_validation, plan = None, None
    if learning:
        pool, curve_validation, plan = curve_plan(len(X), labels, cfg['gpr_max_rows'])
        pd.DataFrame({'case_id': df.iloc[train].case_id.to_numpy(),
                      'split': np.where(np.isin(np.arange(len(X)), curve_validation), 'validation', 'train_pool')}).to_csv(out/'learning_curve_split.csv', index=False)
        curve_samples = [dict(n=n, repeat=repeat, case_id=case_id)
                         for n, repeat, ids, status in plan if status == 'evaluated'
                         for case_id in df.iloc[train[ids]].case_id.tolist()]
        pd.DataFrame(curve_samples).to_csv(out/'learning_curve_samples.csv', index=False)
        manifest['learning_curve_note'] = ('Fixed 20% validation within development; 5 independent seed-2026 draws per n. '
            'Requested n above the remaining pool is marked skipped; full pool is an extra measured point. '
            'Hyperparameters fixed by development CV; descriptive, not independent model-selection evidence.')
    all_folds, summaries, lc_rows = [], [], []
    thermal_oof={}
    for target in TARGETS:
        y = df.iloc[train][target].to_numpy()
        for name, (est, space) in models_by_target[target].items():
            print(f"{'Evaluating fixed formula' if name == 'physics_baseline' else 'Tuning'} {target}: {name}", flush=True)
            sampled = sampled_parameters(space, cfg['budget'])
            grid = [{k: [v] for k, v in item.items()} for item in sampled]
            search = GridSearchCV(est, grid, scoring=log_rmse_score, cv=folds, refit=True,
                                  n_jobs=cfg['n_jobs'], error_score='raise')
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', ConvergenceWarning)
                search.fit(X, y)
            elapsed = time.perf_counter()-start
            manifest['warnings'].extend(sorted({f'{target}/{name}: {w.message}' for w in captured}))
            best = search.best_estimator_
            key = target+'__'+name
            joblib.dump(best, out/(key+'.joblib'))
            pd.DataFrame(search.cv_results_).to_csv(out/(key+'__search.csv'), index=False)
            manifest['models'][key] = {'parameters': search.best_params_, 'tuning_seconds': 0.0 if name in ['physics_baseline','ann_ref'] else elapsed,
                'tuning_candidates': 0 if not space else len(sampled),
                'evaluation_seconds': elapsed,
                'fit_rows': len(best.sample_indices_) if hasattr(best,'sample_indices_') else len(X),
                'note': 'Fixed P0-2 formula; no tuning or learned parameters.' if name == 'physics_baseline' else ann_note if name=='ann_ref' else 'Development CV only.'}
            row_start = len(all_folds)
            if target=='dp_thermal_Pa': thermal_oof[name]=[]
            for fold, (a, b) in enumerate(folds):
                fitted = clone(best)
                t = time.perf_counter()
                with warnings.catch_warnings(record=True) as fold_warnings:
                    warnings.simplefilter('always', ConvergenceWarning)
                    fitted.fit(X.iloc[a], y[a])
                manifest['warnings'].extend(sorted({f'{target}/{name}/fold{fold}: {w.message}' for w in fold_warnings}))
                fit_seconds = time.perf_counter()-t
                pred = fitted.predict(X.iloc[b])
                if target=='dp_thermal_Pa': thermal_oof[name].append(pred)
                all_folds.append({'target': target, 'model': name, 'fold': fold,
                    'fit_rows': len(fitted.sample_indices_) if hasattr(fitted,'sample_indices_') else len(a),
                    'validation_rows': len(b), 'fit_seconds': fit_seconds, **metrics(y[b], pred)})
            group = pd.DataFrame(all_folds[row_start:])
            timing_X = pd.concat([X]*int(np.ceil(10000/len(X))), ignore_index=True).iloc[:10000]
            best.predict(timing_X.iloc[:5])
            timing = []
            for _ in range(3):
                t = time.perf_counter()
                batched_predict(best, timing_X)
                timing.append(time.perf_counter()-t)
            summary = {'target': target, 'model': name, 'tuning_seconds': 0.0 if name in ['physics_baseline','ann_ref'] else elapsed,
                'predict_10k_seconds_median': float(np.median(timing))}
            for metric in ['r2', 'rmse', 'mape_pct', 'log10_rmse', 'log10_r2', 'fit_seconds']:
                summary[metric+'_mean'] = group[metric].mean()
                summary[metric+'_std'] = group[metric].std(ddof=1)
            summaries.append(summary)
            if name == 'mlp':
                mlp = best.regressor_.named_steps['model']
                pd.DataFrame({'iteration': np.arange(1, len(mlp.loss_curve_)+1), 'training_loss_log_target': mlp.loss_curve_,
                              'internal_validation_r2_log_target': mlp.validation_scores_}).to_csv(out/(target+'__mlp_history.csv'), index=False)
                fig, ax = plt.subplots(1, 2, figsize=(9, 3))
                ax[0].plot(mlp.loss_curve_); ax[0].set(xlabel='Iteration', ylabel='Training loss (log target)')
                ax[1].plot(mlp.validation_scores_); ax[1].set(xlabel='Iteration', ylabel='Internal validation R2 (log target)')
                fig.tight_layout(); fig.savefig(out/(target+'__mlp_history.png'), dpi=300); plt.close(fig)
            if learning:
                lc_rows.extend(learning_rows(best, X, y, target, name, cfg['gpr_max_rows'],
                                             curve_validation, plan, metrics, manifest['warnings']))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'cv_summary.csv', index=False)
    pd.DataFrame(all_folds).to_csv(out/'cv_fold_metrics.csv', index=False)
    manifest['selected'] = {target: summary[summary.target == target].sort_values(['log10_rmse_mean', 'model']).iloc[0]['model'] for target in TARGETS}
    if needs_hyd_gpr:
        manifest['hydraulic'],hyd_table=hydraulic_cv(
            df.iloc[train].reset_index(drop=True),FEATURES,folds,None,rigid_pressure(df.iloc[train]),
            out,cfg['gpr_max_rows'],thermal_oof=thermal_oof[manifest['selected']['dp_thermal_Pa']],
            warning_records=manifest['warnings'])
        print(hyd_table.to_string(index=False),flush=True)
    if lc_rows:
        curve = pd.DataFrame(lc_rows)
        curve.to_csv(out/'learning_curve.csv', index=False)
        for target in TARGETS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for name, frame in curve[(curve.target == target) & (curve.status == 'evaluated')].groupby('model'):
                agg = frame.groupby('n').rmse.agg(['mean', 'std'])
                ax.errorbar(agg.index, agg['mean'], yerr=agg['std'], label=name, marker='o')
            ax.set(xscale='log', yscale='log', xlabel='Training samples', ylabel='Validation RMSE ('+('W' if target == 'Q_leak_W' else 'Pa')+')')
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/(target+'__learning.png'), dpi=300); plt.close(fig)
    manifest['warnings'] = sorted(set(manifest['warnings']))
    save_json(out/'manifest.json', manifest)
    print('Development complete. Selected models:', manifest['selected'], flush=True)
    return manifest


def validate_finalize_config(manifest, config):
    if config is not None:
        declared, _ = settings(config)
        if declared != manifest.get('config', {}):
            raise ValueError('Config differs from development. Finalize must use the recorded configuration.')


def finalize(data, out, config=None):
    if (out/'final_test_metrics.csv').exists() or (out/'final_evaluation_started.json').exists():
        raise FileExistsError('Final test already opened. Do not repeatedly evaluate or tune against it.')
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    validate_finalize_config(manifest, config)
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Source data changed since development.')
    df = load_data(data)
    splits = pd.read_csv(out/'split.csv')
    test = df.set_index('case_id').loc[splits.loc[splits.split == 'test', 'case_id']].reset_index()
    save_json(out/'final_evaluation_started.json', {'purpose': 'final holdout opened after model selection', 'selected': manifest['selected'],
        'config': manifest.get('config', {}), 'effective_config': manifest.get('effective_config', {})})
    rows, preds = [], pd.DataFrame({'case_id': test.case_id})
    selected_predictions = {}
    for target in TARGETS:
        # MLP is a prespecified reporting comparator, never selected from test results.
        for name in sorted({manifest['selected'][target], 'mlp'}):
            model = joblib.load(out/(target+'__'+name+'.joblib'))
            pred = model.predict(test[FEATURES])
            rows.append({'target': target, 'model': name, 'n': len(test), **metrics(test[target], pred)})
            preds[target+'__actual'] = test[target]
            preds[target+'__'+name] = pred
            if name == manifest['selected'][target]:
                selected_predictions[target] = pred
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(test[target], pred, s=20)
            low, high = min(test[target].min(), pred.min()), max(test[target].max(), pred.max())
            ax.plot([low, high], [low, high], '--', color='gray')
            unit = 'W' if target == 'Q_leak_W' else 'Pa'
            ax.set(xlabel=f'Solver ({unit})', ylabel=f'Prediction ({unit})', title=f'{target}: {name}', xscale='log', yscale='log')
            fig.tight_layout(); fig.savefig(out/(target+'__'+name+'__parity.png'), dpi=300); plt.close(fig)
    mode,ratio=load_ratio_model(out,manifest)
    pressure = selected_predictions['dp_thermal_Pa']+predicted_ratio(ratio,test[FEATURES])*rigid_pressure(test)
    actual = test.dp_design_Pa.to_numpy()
    bad_actual, bad_pred = actual > 48300, pressure > 48300
    boundary = abs(actual-48300) <= 10000
    report = {'n': len(test), **metrics(actual, pressure), 'boundary_n': int(boundary.sum()),
        'boundary_metrics': metrics(actual[boundary], pressure[boundary]) if boundary.sum() >= 2 else None,
        'false_safe_count': int((bad_actual & ~bad_pred).sum()),
        'false_unsafe_count': int((~bad_actual & bad_pred).sum()),
        'actual_violations': int(bad_actual.sum()), 'constraint_agreement': float(np.mean(bad_actual == bad_pred))}
    comparisons=manifest.get('config',{}).get('finalize_pressure_comparisons',[mode])
    comparison_rows=[]
    for comparison in comparisons:
        _,ratio_model=load_ratio_model(out,manifest,comparison)
        comparison_pressure=selected_predictions['dp_thermal_Pa']+predicted_ratio(ratio_model,test[FEATURES])*rigid_pressure(test)
        comparison_rows.append({'hyd_model':comparison,**pressure_metrics(actual,comparison_pressure)})
        preds['dp_design_Pa__'+comparison]=comparison_pressure
    pd.DataFrame(comparison_rows).to_csv(out/'final_pressure_comparison.csv',index=False)
    preds['dp_design_Pa__actual'] = actual
    preds['dp_design_Pa__hybrid'] = pressure
    preds['actual_violation'] = bad_actual
    preds['predicted_violation'] = bad_pred
    pd.DataFrame(rows).to_csv(out/'final_test_metrics.csv', index=False)
    preds.to_csv(out/'final_test_predictions.csv', index=False)
    save_json(out/'final_pressure_metrics.json', report)
    print('Final test complete. Do not change models based on this holdout.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['develop', 'finalize'])
    parser.add_argument('--data', type=Path, default=ROOT/'data/dataset_ai.csv')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--quick', action='store_true', help='Smoke run only: 3 folds, 2 candidate configurations.')
    parser.add_argument('--learning-curves', action='store_true')
    parser.add_argument('--physics-feature', action=argparse.BooleanOptionalAction, default=None,
                        help='Append fixed log10(Q_phys/mdot_kg_h) inside each learned model Pipeline.')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--cv-repeats', type=int)
    parser.add_argument('--n-jobs', type=int)
    parser.add_argument('--gpr-max-rows', type=int)
    parser.add_argument('--estimate-time', action='store_true')
    parser.add_argument('--hyd-model',choices=['const','gpr'])
    parser.add_argument('--ann-ref-config',type=Path)
    args = parser.parse_args()
    out = args.out or ROOT/'outputs'/('full10k' if args.config else 'pilot')
    if args.stage == 'develop':
        development(args.data, out, args.quick, args.learning_curves, args.physics_feature,
                    config=args.config, budget=args.budget, cv_repeats=args.cv_repeats,
                    n_jobs=args.n_jobs, gpr_max_rows=args.gpr_max_rows, estimate_time=args.estimate_time,
                    hyd_model=args.hyd_model,ann_ref_config=args.ann_ref_config)
    else:
        if args.estimate_time or args.quick or args.learning_curves or any(value is not None for value in
                [args.physics_feature, args.budget, args.cv_repeats, args.n_jobs, args.gpr_max_rows,args.hyd_model,args.ann_ref_config]):
            parser.error('finalize uses saved effective settings; only --config, --data and --out are accepted.')
        finalize(args.data, out, config=args.config)


if __name__ == '__main__':
    main()
