"""Reproducible training caps and settings, shared by CLI and saved models."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler

SEED = 2026
DEFAULTS = dict(physics_feature=False, seed=SEED, test_size=.2, stratify='vaporized',
                gpr_max_rows=3000, budget=6, cv_splits=5, cv_repeats=3, n_jobs=1, hyd_model='const',blas_threads=None,exhaustive_gpr=False,
                memory_limit_mib=6144,memory_reserve_mib=2048,memory_safety_factor=1.25,
                fit_memory_mib={},other_model_memory_mib=1024)
METADATA = {'preregistered_on', 'decision_basis', 'scope', 'budget_rule',
            'finalize_pressure_comparisons', 'hydraulic_preregistration',
            'execution_revision', 'resource_evidence'}


def read_config(config):
    if config is None:
        return {}
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text(encoding='utf-8'))
    return dict(config)


def settings(config=None, quick=False, **overrides):
    declared = read_config(config)
    unknown = set(declared)-set(DEFAULTS)-METADATA
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    result = {**DEFAULTS, **{k: v for k, v in declared.items() if k in DEFAULTS}}
    if quick:
        result.update(budget=2, cv_splits=3, cv_repeats=1)
    result.update({k: v for k, v in overrides.items() if v is not None})
    if set(result) != set(DEFAULTS):
        raise ValueError('Unknown settings override.')
    if result['seed'] != SEED or result['stratify'] != 'vaporized':
        raise ValueError('Use seed 2026 and stratify=vaporized.')
    if result['blas_threads'] is not None and (isinstance(result['blas_threads'],bool) or not isinstance(result['blas_threads'],int) or result['blas_threads']<1):
        raise ValueError('blas_threads must be null (native) or a positive integer.')
    if not isinstance(result['exhaustive_gpr'],bool):
        raise ValueError('exhaustive_gpr must be boolean.')
    if result['memory_limit_mib']<=0 or result['memory_reserve_mib']<0 or result['memory_safety_factor']<1:
        raise ValueError('Invalid memory limit, reserve or safety factor.')
    if any(float(v)<=0 for v in result['fit_memory_mib'].values()):
        raise ValueError('Measured fit memory must be positive.')
    if result['hyd_model'] not in ['const','gpr']:
        raise ValueError('hyd_model must be const or gpr.')
    if not isinstance(result['physics_feature'], bool):
        raise ValueError('physics_feature must be boolean.')
    for key in ['gpr_max_rows', 'budget', 'cv_splits', 'cv_repeats', 'n_jobs']:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value == 0 or (key != 'n_jobs' and value < 1):
            raise ValueError(f'{key} must be a valid nonzero integer.')
    if result['cv_splits'] < 2 or result['gpr_max_rows'] < 2:
        raise ValueError('cv_splits and gpr_max_rows must be at least 2.')
    if isinstance(result['test_size'], bool) or not 0 < result['test_size'] < 1:
        raise ValueError('test_size must be between 0 and 1.')
    return declared, result


def sample_indices(n_rows, max_rows):
    if n_rows <= max_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(SEED).choice(n_rows, max_rows, replace=False))


class SubsampledTargetRegressor(TransformedTargetRegressor):
    """Cap GPR before any Pipeline preprocessing; validation is never sampled.

    GPR uses a subsample because exact training costs O(n³). Other models
    continue to use all available training rows. regressor_ remains a Pipeline
    so existing inference and log-space sigma extraction work unchanged.
    """
    def __init__(self, regressor=None, *, transformer=None, func=None,
                 inverse_func=None, check_inverse=True, max_rows=3000):
        super().__init__(regressor=regressor, transformer=transformer, func=func,
                         inverse_func=inverse_func, check_inverse=check_inverse)
        self.max_rows = max_rows

    def fit(self, X, y, **fit_params):
        if not isinstance(self.max_rows, int) or self.max_rows < 2:
            raise ValueError('max_rows must be an integer >= 2.')
        self.sample_indices_ = sample_indices(len(X), self.max_rows)
        subset = X.iloc[self.sample_indices_] if hasattr(X, 'iloc') else X[self.sample_indices_]
        return super().fit(subset, np.asarray(y)[self.sample_indices_], **fit_params)


def sampled_parameters(space, budget):
    """Exact equal budget without silently truncating finite search spaces.

    Preserve pilot finite candidates when sufficient. Larger budgets use
    continuous alpha/learning-rate or integer tree-count ranges with the same
    endpoints, fixed before full-data evaluation. Record every actual draw.
    """
    if not space:
        return [{}]
    distribution = dict(space)
    if budget > len(ParameterGrid(space)):
        for key in space:
            values = space[key]
            if key.endswith(('alpha', 'learning_rate_init')) and len(values) > 1:
                distribution[key] = loguniform(min(values), max(values))
                break
            if key.endswith('n_estimators') and len(values) > 1:
                distribution[key] = randint(min(values), max(values)+1)
                break
        else:
            raise ValueError('Requested budget exceeds the available unique candidates.')
    samples = list(ParameterSampler(distribution, n_iter=budget, random_state=SEED))
    # Integer distributions may repeat; refill deterministically rather than
    # claiming repeated configurations are unique candidates.
    unique = {}
    for item in samples:
        unique.setdefault(repr(sorted(item.items())), item)
    if len(unique) < budget:
        for item in ParameterSampler(distribution, n_iter=budget*100, random_state=SEED):
            unique.setdefault(repr(sorted(item.items())), item)
            if len(unique) == budget:
                break
    if len(unique) != budget:
        raise ValueError('Could not draw the requested unique tuning budget.')
    return list(unique.values())


def budget_decision(estimates, rule):
    """Apply preregistered thresholds to the reference-budget total estimate."""
    if rule.get('mode')=='wall_clock':
        from execution_plan import walltime_decision
        return walltime_decision(estimates,rule)
    reference=rule['reference_budget']
    total=0.0
    for row in estimates.to_dict('records'):
        count=row['candidates']
        # Fixed models have one evaluated candidate, not a tunable budget.
        tunable=row.get('tunable',row['model'] not in ['linear','physics_baseline','ann_ref'])
        search=row['estimated_search_seconds']*(reference/count if tunable else 1)
        total += row['estimated_develop_seconds']-row['estimated_search_seconds']+search
    budget=rule['otherwise_budget']
    for threshold in rule['thresholds']:
        limit=threshold['max_hours']*3600
        if total<=limit or np.isclose(total,limit,rtol=0,atol=1e-6):
            budget=threshold['budget']; break
    return dict(applied_budget=budget, reference_budget=reference,
                reference_estimated_seconds=total,
                reason=f'Actual --data development-fold estimate at budget {reference}: {total/3600:.6f} h; preregistered <=4h:50, <=8h:20, otherwise:6.')
