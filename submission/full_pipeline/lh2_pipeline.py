"""LH2 흡입배관의 대리모델 비교/선정과 최종 평가를 담당하는 실행 코드.

develop에서는 개발 자료로 후보 비교, 학습 곡선, 수력 보정을 준비함.
finalize에서는 이미 선정한 모델을 불러 고정 테스트셋을 한 번만 평가함.
1만 개 자료의 기본 분할은 개발 8,000개 / 테스트 2,000개이며,
기화 라벨 비율을 유지하도록 층화하고 무작위 seed는 2026으로 고정함.
테스트 결과를 보고 모델이나 설정을 바꾸지 않도록 두 단계를 분리했음.
"""
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
from development_tools import batched_predict, curve_plan, learning_rows, estimate_runtime, fit_validation
from execution_plan import resource_plan,candidate_parameters,walltime_decision,release_workers,reschedule_estimates
from joblib import Parallel,delayed,parallel_config
from threadpoolctl import threadpool_limits
from artifact_state import atomic_json,write_receipt,sha256
from ann_reference import load_ann_reference
from hydraulics import hydraulic_cv, ratio_estimator, load_ratio_model, predicted_ratio, pressure_metrics

# 1. 공통 입력과 물리식
# case_id·정답·기화 라벨은 FEATURES에 넣지 않음. 기화 라벨은 층화 분할에만 사용함.
# 열전도도는 값의 범위가 넓어 log10 값으로 입력하며, BOUNDS는 예측 허용 범위임.
SEED = 2026
FEATURES = ['D_mm', 't_ins_mm', 'f_Hz', 'L_m', 'mdot_kg_h', 'log10_k_MLI']
TARGETS = ['Q_leak_W', 'dp_thermal_Pa']
BOUNDS = {'D_mm': (12, 50), 't_ins_mm': (2, 25), 'f_Hz': (2, 10),
          'L_m': (.5, 3), 'mdot_kg_h': (50, 100), 'k_MLI_W_mK': (10**-4.1, .01)}


def save_json(path, value):
    """실행 설정과 감사 기록을 사람이 읽을 수 있는 JSON으로 남김.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def rigid_pressure(df):
    """AI 오차와 수력 근사 오차를 구분하려고 강체 압력은 식으로 계산함.
    최적화와 새 입력 예측에서도 같은 식을 재사용함.
    """
    # 내경은 mm→m, 유량은 kg/h→kg/s로 환산. 식을 정리하면 밀도 항은 소거됨.
    area = np.pi * (df.D_mm.to_numpy() / 1000)**2 / 4
    return df.L_m.to_numpy() * (np.pi * df.mdot_kg_h.to_numpy() / 3600) * (2*np.pi*df.f_Hz.to_numpy()) / area

def load_data(path):
    """단위·중복·물리 관계를 먼저 검사해 잘못된 자료가 학습에 들어가는 것을 막음.
    정답 없는 입력 목록은 거부하고, status가 있으면 수렴한 행만 사용함.
    """
    df = pd.read_csv(path)
    required = ['case_id', *BOUNDS, *TARGETS, 'dp_hyd_Pa', 'dp_design_Pa', 'dp_rigid_Pa', 'D_o_mm', 'vaporized']
    missing = sorted(set(required) - set(df))
    if missing:
        raise ValueError(f'Missing solver-output columns: {missing}. Input-only cases_10k.csv cannot train a model.')
    # 수렴 실패 행은 정답이 비어 있을 수 있어 수치 검사 전에 제외하고 ID를 보존함.
    source_rows = len(df)
    excluded = df.loc[df.status != 'converged', 'case_id'].tolist() if 'status' in df else []
    if 'status' in df:
        df = df.loc[df.status == 'converged'].copy().reset_index(drop=True)
    df.attrs['source_rows'] = source_rows
    df.attrs['excluded_status_count'] = len(excluded)
    df.attrs['excluded_status_case_ids'] = excluded
    # 기화 비율 변화는 확인할 사항이지만, 파일럿과 다르다는 이유로 자료를 버리지는 않음.
    if 8000 <= source_rows <= 12000 and len(df) and abs(df.vaporized.mean()-.06) > .03:
        warnings.warn(f'Large dataset vaporized rate {df.vaporized.mean():.2%} differs from pilot 6% by more than 3 percentage points.')
    if df[required].isna().any().any() or not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite values found.')
    # 같은 설계가 학습/검증 양쪽에 들어가 정확도가 부풀려지는 것을 막음.
    if df.case_id.duplicated().any() or df.duplicated(list(BOUNDS)).any():
        raise ValueError('Duplicate IDs or input conditions: resolve before splitting.')
    if len(df) < 50 or (df[TARGETS] <= 0).any().any():
        raise ValueError('At least 50 rows and strictly positive targets are required.')
    for col, (lo, hi) in BOUNDS.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f'{col} is outside the declared domain.')
    # CSV에 저장된 로그값도 원값과 대조해 열 이름·변환 방식의 혼동을 확인함.
    for col, raw in [('log10_k_MLI', 'k_MLI_W_mK'), ('log10_Q_leak', 'Q_leak_W')]:
        computed = np.log10(df[raw])
        if col in df and not np.allclose(df[col], computed, atol=1e-6, rtol=0):
            raise ValueError(f'{col} is inconsistent with {raw}.')
        df[col] = computed
    # 외경, 압력 합, 임계값 라벨, 강체식을 함께 대조해 단위/열 대응 오류를 찾음.
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
    """실제 열적 압력을 넣어 수력 계수 1.8 자체에서 생기는 오차를 따로 확인
    전체 자료의 일관성 감사이며, 이 수치로 모델을 선정하지 않음.
    """
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


# 2. 비교 모델과 평가 기준
def candidates(quick=False, target=None, physics_feature=False):
    """모든 모델이 같은 입력과 출력 변환으로 비교되도록 Pipeline으로 묶음.
    스케일러·다항 변환은 CV의 학습 fold에서만 fit해 정보 누출을 막음.
    """
    # 물리 특성도 Pipeline 안에서 원본 입력으로 계산함. 정답을 파생 입력으로 쓰지 않음.
    # 출력은 log10으로 학습하고 predict에서는 원단위로 복원해 같은 지표로 비교하였음
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
    # GPR은 계산량 O(n³) 때문에 기본 최대 3,000개만 사용함.
    # 각 학습 fold 안에서 seed 2026으로 추출하고 전처리도 그 부분표본에만 fit함.
    # 검증 행은 줄이지 않으며 다른 모델은 해당 학습 자료 전체를 사용
    gpr = models['gpr'][0]
    models['gpr'] = (SubsampledTargetRegressor(regressor=gpr.regressor,
                     func=np.log10, inverse_func=exp10), models['gpr'][1])
    # 열침입은 AI가 필요한지 판단할 수 있도록 튜닝 없는 물리식도 비교 대상에 포함하였음
    if target == 'Q_leak_W':
        models['physics_baseline'] = (PhysicsBaseline(), {})
    return models


def lambda_power10(y):
    """log10 공간의 값을 원단위로 되돌리는 보조 함수
    """
    return np.power(10., y)


def log_rmse_score(estimator, X, y):
    """출력 크기에 따른 상대적인 예측 차이를 비교하려고 log10 RMSE를 사용함
    GridSearchCV는 큰 점수를 고르므로 음수로 반환하고, 유효하지 않은 예측은 탈락시킴.
    """
    prediction = estimator.predict(X)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        return -np.inf
    return -float(np.sqrt(mean_squared_error(np.log10(y), np.log10(prediction))))


def metrics(y, pred):
    """원단위 오차와 log10 공간 오차를 함께 남겨 한 지표만으로 해석하지 않도록 함
    MAPE는 백분율이며 RMSE의 단위는 각 출력의 원단위와 같음.
    """
    return {'r2': r2_score(y, pred), 'rmse': np.sqrt(mean_squared_error(y, pred)),
            'mape_pct': 100*mean_absolute_percentage_error(y, pred),
            'log10_rmse': np.sqrt(mean_squared_error(np.log10(y), np.log10(pred))),
            'log10_r2': r2_score(np.log10(y), np.log10(pred))}


# 3. 개발 자료에서 모델 비교와 선정
def _development_impl(data, out, quick=False, learning=False, physics_feature=None, *,
                config=None, budget=None, cv_repeats=None, n_jobs=None,
                gpr_max_rows=None, estimate_time=False, hyd_model=None, ann_ref_config=None,blas_threads=None,execution_plan=None):
    """고정 개발 분할에서 튜닝·최종 개발 학습·보고 CV를 순서대로 수행함.
    실행 비용으로 예산을 조정할 수 있으나 테스트 성능은 판단에 사용하지 않음.
    """
    declared, cfg = settings(config, quick=quick, physics_feature=physics_feature,
                             budget=budget, cv_repeats=cv_repeats, n_jobs=n_jobs,
                             gpr_max_rows=gpr_max_rows, hyd_model=hyd_model,blas_threads=blas_threads)
    # 원본 JSON과 CLI를 반영한 유효 설정을 구분해 기록함.
    # 저장된 비용 계획을 재사용할 때는 데이터·설정·실행 모드가 같은지 먼저 확인함.
    accepted_plan=None
    if execution_plan is not None:
        if estimate_time or budget is not None or gpr_max_rows is not None:
            raise ValueError('Saved execution plan cannot be combined with estimate/budget/row overrides.')
        accepted_plan=json.loads(Path(execution_plan).read_text())
        if (accepted_plan['data_sha256']!=sha256(data) or accepted_plan['config']!=declared
                or accepted_plan['quick']!=quick or accepted_plan['learning_curves']!=learning):
            raise ValueError('Saved execution plan does not match data/config/mode.')
        effective=accepted_plan['effective_config']
        if set(effective)!=set(cfg) or any(effective[k]!=cfg[k] for k in cfg if k not in ['budget','gpr_max_rows']):
            raise ValueError('Execution plan runtime settings differ from this invocation.')
        if accepted_plan.get('ann_ref_sha256') and accepted_plan['ann_ref_sha256']!=sha256(ann_ref_config or ROOT/'configs/ann_ref.json'):
            raise ValueError('ANN reference configuration changed after estimate.')
        _,cfg=settings({**declared,**effective})
    # 기존 결과를 덮어쓰지 않음. 시간 추정만 요청한 경우에는 기존 폴더도 읽기만 함.
    if not estimate_time and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty output directory to preserve existing experiments.')
    df = load_data(data)
    # 기본 20%를 테스트로 남김. 1만 개라면 개발 8,000개 / 테스트 2,000개임.
    # vaporized로 층화해 압력 한계 초과 사례 비율을 유지하고 seed 2026으로 분할을 고정함.
    train, test = train_test_split(np.arange(len(df)), test_size=cfg['test_size'],
                                   random_state=SEED, stratify=df[cfg['stratify']])
    X = df.iloc[train][FEATURES]
    labels = df.iloc[train].vaporized.to_numpy()
    # 모든 모델에 같은 개발 CV 인덱스를 적용해 분할 차이가 비교에 섞이지 않도록 함.
    # full10k 설정은 5-fold×1회이며, 반복 횟수는 기록된 유효 설정을 따름.
    folds = list(RepeatedKFold(n_splits=cfg['cv_splits'], n_repeats=cfg['cv_repeats'],
                               random_state=SEED).split(X))
    models_by_target = {target: candidates(quick, target=target,
                                          physics_feature=cfg['physics_feature']) for target in TARGETS}
    # 참고 ANN은 논문 설정이 채워졌을 때만 추가함. 미확인 구조를 추측해 사용하지 않음.
    ann_spec,ann_model,ann_note=load_ann_reference(ann_ref_config or ROOT/'configs/ann_ref.json')
    print(ann_note,flush=True)
    if ann_model is not None:
        for models in models_by_target.values():
            models['ann_ref']=(clone(ann_model),{})
    for models in models_by_target.values():
        if 'gpr' in models:
            models['gpr'][0].set_params(max_rows=cfg['gpr_max_rows'])
    # const/GPR 수력 비교가 사전 지정돼 있으면 기본 모드가 const여도 보정 모델을 준비함.
    needs_hyd_gpr=cfg['hyd_model']=='gpr' or 'gpr' in declared.get('finalize_pressure_comparisons',[])
    estimate_models=dict(models_by_target)
    estimate_targets={target: df.iloc[train][target].to_numpy() for target in TARGETS}
    if needs_hyd_gpr:
        estimate_models['hyd_ratio']={'gpr':(ratio_estimator(cfg['gpr_max_rows']),{})}
        estimate_targets['hyd_ratio']=df.iloc[train].dp_hyd_Pa.to_numpy()/df.iloc[train].dp_rigid_Pa.to_numpy()
    decision=accepted_plan['budget_decision'] if accepted_plan else {'applied_budget':cfg['budget'], 'reason':'Explicit CLI override' if budget is not None else 'Quick smoke settings' if quick else 'Configured fixed budget'}
    # 후보 하나·fold 하나의 실측 시간으로 비용을 추정함.
    # 예산과 GPR 행 수는 사전 비용 규칙으로 정하며, 검증·테스트 정확도로 바꾸지 않음.
    if accepted_plan is None and (estimate_time or (declared.get('budget_rule') and not quick and budget is None)):
        estimate=estimate_runtime(estimate_models,X,estimate_targets,folds,cfg,learning=learning,labels=labels)
        if declared.get('budget_rule') and budget is None and not quick:
            rule=declared['budget_rule']
            decision=budget_decision(estimate,rule)
            # 메모리·시간 때문에 GPR 행 수를 줄이면 GPR만 다시 측정함.
            # 다른 모델은 기존 측정값을 새 병렬 수에 맞춰 재계산해 불필요한 학습을 피함.
            if decision.get('reduced_estimate_required'):
                reduced_cfg={**cfg,'gpr_max_rows':rule['fallback_gpr_rows']}
                capped_models={target:{name:(clone(est).set_params(max_rows=reduced_cfg['gpr_max_rows']),space)
                               for name,(est,space) in models.items() if name=='gpr'}
                               for target,models in estimate_models.items()}
                capped_models={target:models for target,models in capped_models.items() if models}
                reduced=estimate_runtime(capped_models,X,estimate_targets,folds,reduced_cfg,learning=learning,labels=labels)
                keys=set(zip(reduced.target,reduced.model))
                unchanged=estimate.loc[[(target,name) not in keys for target,name in zip(estimate.target,estimate.model)]]
                unchanged=reschedule_estimates(unchanged,reduced_cfg)
                reduced_all=pd.concat([unchanged,reduced],ignore_index=True)
                decision=walltime_decision(estimate,rule,reduced_all)
                cfg['gpr_max_rows']=decision['applied_gpr_max_rows']
                for models in models_by_target.values():
                    if 'gpr' in models:models['gpr'][0].set_params(max_rows=cfg['gpr_max_rows'])
                estimate=reduced_all
            cfg['budget']=decision['applied_budget']
        print('Budget decision: '+json.dumps(decision,ensure_ascii=False),flush=True)
        if estimate_time:
            estimate.attrs['budget_decision']=decision
            estimate.attrs['effective_config']=cfg
            return estimate
    # 시간 추정은 위에서 반환되므로 아래부터 실제 개발 산출물을 저장
    # audit는 전체 자료의 물리 일관성을 확인하며, 이후 모델 학습·선정에는 개발 행만 사용
    out.mkdir(parents=True, exist_ok=True)
    audit(df, out)
    split = pd.DataFrame({'case_id': df.case_id, 'split': np.where(np.isin(np.arange(len(df)), test), 'test', 'development')})
    split.to_csv(out/'split.csv', index=False)
    np.savez(out/'cv_indices.npz', **{f'{i}_{part}': arr for i, pair in enumerate(folds) for part, arr in zip(['train', 'val'], pair)})
    # 데이터 해시, 원본·유효 설정, 실제 후보 수, 환경을 남겨 실행 조건을 추적할 수 있게 함.
    manifest = {'seed': SEED, 'data_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'config': declared, 'effective_config': cfg, 'budget_decision': decision,
        'execution_revision':declared.get('execution_revision'), 'resource_plans':[],
        'saved_execution_plan_sha256':sha256(execution_plan) if execution_plan else None,
        'hyd_model':cfg['hyd_model'], 'ann_ref':{'config':ann_spec,'note':ann_note},
        'features': FEATURES, 'physics_feature': cfg['physics_feature'],
        'derived_features': [PHYSICS_FEATURE_NAME] if cfg['physics_feature'] else [],
        'physics_feature_definition': 'log10(physics_heat_leak(X)/mdot_kg_h), fixed P0-2 formula, inside Pipeline' if cfg['physics_feature'] else None,
        'train_rows': len(train), 'test_rows': len(test), 'quick': quick,
        'cv_folds': len(folds), 'tuning_budget_other_models':cfg['budget'],
        'candidate_count_rule':'GPR exhaustive kernel3 x alpha2 =6; others use budget; fixed models0' if cfg['exhaustive_gpr'] else 'All tunable models use requested budget',
        'gpr_max_rows': cfg['gpr_max_rows'],
        'gpr_note': 'GPR uses a seed-2026 training-only subsample because of O(n^3) cost; other models use all training rows. Search folds use the same subsets as reporting CV.',
        'selection_metric': 'minimum mean CV log10 RMSE, separately per target',
        'cv_note': 'CV used for tuning and selection; these are not unbiased generalization estimates. Final test is separate.',
        'learning_curves': learning,
        'warnings_note': 'Unique parent-process warning records below; parallel search workers also emit warnings to stderr.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'models': {}, 'warnings': []}
    # 부분표본의 case_id를 따로 보존해 GPR 학습 자료와 순서를 재현할 수 있게 함.
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
    curve_validation, curve_sampling_plan = None, None
    # 학습 곡선은 개발 자료 안의 고정 검증 자료로 평가함. 최종 테스트는 사용하지 않음.
    # 크기별 재표집 ID도 저장하며, 검증 자료를 침범하거나 GPR 한도를 넘는 크기는 생략함.
    if learning:
        pool, curve_validation, curve_sampling_plan = curve_plan(len(X), labels, cfg['gpr_max_rows'])
        pd.DataFrame({'case_id': df.iloc[train].case_id.to_numpy(),
                      'split': np.where(np.isin(np.arange(len(X)), curve_validation), 'validation', 'train_pool')}).to_csv(out/'learning_curve_split.csv', index=False)
        curve_samples = [dict(n=n, repeat=repeat, case_id=case_id)
                         for n, repeat, ids, status in curve_sampling_plan if status == 'evaluated'
                         for case_id in df.iloc[train[ids]].case_id.tolist()]
        pd.DataFrame(curve_samples).to_csv(out/'learning_curve_samples.csv', index=False)
        manifest['learning_curve_note'] = ('Fixed 20% validation within development; 5 independent seed-2026 draws per n. '
            'Requested n above the remaining pool is marked skipped; full pool is an extra measured point. '
            'Hyperparameters fixed by development CV; descriptive, not independent model-selection evidence.')
    # 아래 비교는 출력별로 독립 수행함. 수력 비교에 재사용할 열적 압력의 OOF 예측도 보관함.
    # OOF는 해당 행을 학습에서 제외한 fold 모델의 예측이며, 최종 테스트 예측과는 구분됨.
    all_folds, summaries, lc_rows = [], [], []
    thermal_oof={}
    for target in TARGETS:
        y = df.iloc[train][target].to_numpy()
        for name, (est, space) in models_by_target[target].items():
            print(f"{'Evaluating fixed formula' if name == 'physics_baseline' else 'Tuning'} {target}: {name}", flush=True)
            # full10k에서 GPR은 커널 3종×alpha 2종의 6후보를 전수 탐색함.
            # Ridge·RF·MLP는 선택된 budget만큼 비교하고, 실제 후보 수를 그대로 기록함.
            sampled = candidate_parameters(name,space,cfg['budget'],cfg['exhaustive_gpr'])
            plan=resource_plan(cfg,max_tasks=len(sampled)*len(folds))
            manifest['resource_plans'].append({'target':target,'model':name,'stage':'search',**plan})
            grid = [{k: [v] for k, v in item.items()} for item in sampled]
            # 탐색 중 자동 refit을 끄고, worker 종료 후 상위 프로세스에서 한 번 학습함.
            # 병렬 수와 사전 작업 수를 함께 제한해 예약한 메모리·CPU 한도를 지키도록 함.
            search = GridSearchCV(est, grid, scoring=log_rmse_score, cv=folds, refit=False,
                                  n_jobs=plan['effective_jobs'], pre_dispatch=plan['effective_jobs'], error_score='raise')
            start = time.perf_counter()
            # 수렴 경고는 수집해 manifest에 남김. 병렬 worker의 경고도 stderr로 출력됨.
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', ConvergenceWarning)
                with parallel_config(backend='loky',inner_max_num_threads=cfg['blas_threads']):
                    search.fit(X, y)
            release_workers(plan['effective_jobs'])
            refit_plan=resource_plan(cfg,max_tasks=1)
            manifest['resource_plans'].append({'target':target,'model':name,'stage':'refit',**refit_plan})
            # CV에서 고른 설정으로 개발 자료를 다시 학습함. 이 단계에도 테스트 행은 넣지 않음.
            best=clone(est).set_params(**search.best_params_)
            refit_started=time.perf_counter()
            with warnings.catch_warnings(record=True) as refit_warnings:
                warnings.simplefilter('always')
                best.fit(X,y)
            refit_seconds=time.perf_counter()-refit_started
            captured.extend(refit_warnings)
            elapsed = time.perf_counter()-start
            manifest['warnings'].extend(sorted({f'{target}/{name}: {w.message}' for w in captured}))
            key = target+'__'+name
            joblib.dump(best, out/(key+'.joblib'))
            pd.DataFrame(search.cv_results_).to_csv(out/(key+'__search.csv'), index=False)
            manifest['models'][key] = {'parameters': search.best_params_, 'tuning_seconds': 0.0 if name in ['physics_baseline','ann_ref'] else elapsed,
                'tuning_candidates': 0 if not space else len(sampled),
                'evaluation_seconds': elapsed,'refit_seconds':refit_seconds,
                'fit_rows': len(best.sample_indices_) if hasattr(best,'sample_indices_') else len(X),
                'note': 'Fixed P0-2 formula; no tuning or learned parameters.' if name == 'physics_baseline' else ann_note if name=='ann_ref' else 'Development CV only.'}
            # 선정 설정을 공통 fold에서 다시 평가해 지표·분산·학습시간을 보고함.
            # 튜닝에 사용한 개발 CV를 재사용하므로 독립적인 일반화 성능 추정은 아님.
            row_start = len(all_folds)
            if target=='dp_thermal_Pa': thermal_oof[name]=[]
            report_plan=resource_plan(cfg,max_tasks=len(folds))
            manifest['resource_plans'].append({'target':target,'model':name,'stage':'reporting_cv',**report_plan})
            with parallel_config(backend='loky',inner_max_num_threads=cfg['blas_threads']):
                validations=Parallel(n_jobs=report_plan['effective_jobs'],pre_dispatch=report_plan['effective_jobs'])(
                    delayed(fit_validation)(best,X,y,a,b) for a,b in folds)
            release_workers(report_plan['effective_jobs'])
            for fold,((a,b),validation) in enumerate(zip(folds,validations)):
                pred=validation['prediction']
                manifest['warnings'].extend(f'{target}/{name}/fold{fold}: {message}' for message in validation['warnings'])
                if target=='dp_thermal_Pa': thermal_oof[name].append(pred)
                all_folds.append({'target':target,'model':name,'fold':fold,
                    'fit_rows':validation['fit_rows'],'validation_rows':len(b),
                    'fit_seconds':validation['fit_seconds'],**metrics(y[b],pred)})
            group = pd.DataFrame(all_folds[row_start:])
            # 예측 속도는 개발 입력을 반복해 1만 행으로 측정함. 새 정답을 만드는 과정은 아님.
            # 준비 호출 후 3회 측정의 중앙값을 남겨 첫 호출 비용의 영향을 줄임.
            timing_X = pd.concat([X]*int(np.ceil(10000/len(X))), ignore_index=True).iloc[:10000]
            best.predict(timing_X.iloc[:5])
            timing = []
            for _ in range(3):
                t = time.perf_counter()
                batched_predict(best, timing_X)
                timing.append(time.perf_counter()-t)
            summary = {'target': target, 'model': name, 'actual_candidates':len(sampled) if space else 0,
                'tuning_seconds': 0.0 if name in ['physics_baseline','ann_ref'] else elapsed,
                'predict_10k_seconds_median': float(np.median(timing)), 'final_refit_seconds':refit_seconds}
            for metric in ['r2', 'rmse', 'mape_pct', 'log10_rmse', 'log10_r2', 'fit_seconds']:
                summary[metric+'_mean'] = group[metric].mean()
                summary[metric+'_std'] = group[metric].std(ddof=1)
            summaries.append(summary)
            # MLP의 학습 손실과 내부 조기 종료 검증 이력을 남겨 수렴 상태를 확인함.
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
                                             curve_validation, curve_sampling_plan, metrics, manifest['warnings']))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'cv_summary.csv', index=False)
    pd.DataFrame(all_folds).to_csv(out/'cv_fold_metrics.csv', index=False)
    # 출력별 평균 CV log10 RMSE가 가장 작은 모델을 선정함.
    # 평균이 같으면 모델명 순으로 결정하며, 테스트 결과로 순위를 다시 정하지 않음.
    manifest['selected'] = {target: summary[summary.target == target].sort_values(['log10_rmse_mean', 'model']).iloc[0]['model'] for target in TARGETS}
    # dp_hyd/dp_rigid 비율은 원본 입력 6개로 학습함.
    # 같은 fold의 열적 압력 OOF 예측을 써 수력 보정 전후를 비교하고 테스트는 남겨 둠.
    if needs_hyd_gpr:
        hyd_plan=resource_plan(cfg,max_tasks=len(folds))
        manifest['resource_plans'].append({'target':'hyd_ratio','model':'gpr','stage':'hydraulic_cv',**hyd_plan})
        manifest['hydraulic'],hyd_table=hydraulic_cv(
            df.iloc[train].reset_index(drop=True),FEATURES,folds,None,rigid_pressure(df.iloc[train]),
            out,cfg['gpr_max_rows'],thermal_oof=thermal_oof[manifest['selected']['dp_thermal_Pa']],
            warning_records=manifest['warnings'],n_jobs=hyd_plan['effective_jobs'],blas_threads=cfg['blas_threads'])
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
    # 설정 기록과 완료 영수증을 남겨 무인 실행이 완료된 개발 단계를 다시 수행하지 않도록 함.
    atomic_json(out/'manifest.json', manifest)
    write_receipt(out,'development_complete.json',{'data_sha256':manifest['data_sha256']})
    print('Development complete. Selected models:', manifest['selected'], flush=True)
    return manifest



def development(data,out,quick=False,learning=False,physics_feature=None,*,
                config=None,budget=None,cv_repeats=None,n_jobs=None,gpr_max_rows=None,
                estimate_time=False,hyd_model=None,ann_ref_config=None,blas_threads=None,execution_plan=None):
    """상위 프로세스의 BLAS 스레드도 제한해 병렬 worker와의 CPU 중복 사용을 줄임.
    실제 개발 절차는 _development_impl에서 수행함.
    """
    _,cfg=settings(config,quick=quick,physics_feature=physics_feature,budget=budget,
                   cv_repeats=cv_repeats,n_jobs=n_jobs,gpr_max_rows=gpr_max_rows,
                   hyd_model=hyd_model,blas_threads=blas_threads)
    with threadpool_limits(limits=cfg['blas_threads']):
        return _development_impl(data,out,quick,learning,physics_feature,config=config,
            budget=budget,cv_repeats=cv_repeats,n_jobs=n_jobs,gpr_max_rows=gpr_max_rows,
            estimate_time=estimate_time,hyd_model=hyd_model,ann_ref_config=ann_ref_config,
            blas_threads=blas_threads,execution_plan=execution_plan)


# 4. 선정 완료 후 최종 테스트 1회 평가
def validate_finalize_config(manifest, config):
    """개발 단계와 다른 설정으로 테스트를 평가하지 않도록 원본 설정을 대조함.
    """
    if config is not None:
        declared, _ = settings(config)
        if declared != manifest.get('config', {}):
            raise ValueError('Config differs from development. Finalize must use the recorded configuration.')


def finalize(data, out, config=None):
    """선정된 모델을 재학습하지 않고 고정 테스트셋에서 한 번만 평가함.
    시작 기록이 있으면 실패·중단 여부와 관계없이 이 함수의 재실행을 막음.
    """
    # 완료 파일뿐 아니라 시작 기록도 검사함. 중단된 평가를 반복해 테스트를 다시 보는 것을 막음.
    if (out/'final_test_metrics.csv').exists() or (out/'final_evaluation_started.json').exists():
        raise FileExistsError('Final test already opened. Do not repeatedly evaluate or tune against it.')
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    validate_finalize_config(manifest, config)
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Source data changed since development.')
    df = load_data(data)
    splits = pd.read_csv(out/'split.csv')
    # seed로 새로 나누지 않고 개발 단계에 저장한 case_id와 순서로 테스트 행을 복원함.
    test = df.set_index('case_id').loc[splits.loc[splits.split == 'test', 'case_id']].reset_index()
    # 모델 예측 전에 시작 기록을 저장함. 이후 실패해도 이 함수는 평가를 재시도하지 않음.
    atomic_json(out/'final_evaluation_started.json', {'purpose': 'final holdout opened after model selection', 'selected': manifest['selected'],
        'config': manifest.get('config', {}), 'effective_config': manifest.get('effective_config', {})})
    rows, preds = [], pd.DataFrame({'case_id': test.case_id})
    selected_predictions = {}
    for target in TARGETS:
        # MLP는 사전에 정한 비교 대상임. 테스트에서 더 좋아도 선정 모델을 교체하지 않음.
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
    # 선정된 열적 압력 예측에 수력 성분을 더해 전체 압력과 48.3 kPa 제약을 평가함.
    mode,ratio=load_ratio_model(out,manifest)
    pressure = selected_predictions['dp_thermal_Pa']+predicted_ratio(ratio,test[FEATURES])*rigid_pressure(test)
    actual = test.dp_design_Pa.to_numpy()
    bad_actual, bad_pred = actual > 48300, pressure > 48300
    # 전체 평균 오차만으로 놓칠 수 있는 경계 판정을 실제 압력 기준 ±10 kPa에서 별도로 확인함.
    # false safe는 실제 초과를 안전으로, false unsafe는 실제 안전을 초과로 예측한 경우임.
    boundary = abs(actual-48300) <= 10000
    report = {'n': len(test), **metrics(actual, pressure), 'boundary_n': int(boundary.sum()),
        'boundary_metrics': metrics(actual[boundary], pressure[boundary]) if boundary.sum() >= 2 else None,
        'false_safe_count': int((bad_actual & ~bad_pred).sum()),
        'false_unsafe_count': int((~bad_actual & bad_pred).sum()),
        'actual_violations': int(bad_actual.sum()), 'constraint_agreement': float(np.mean(bad_actual == bad_pred))}
    # 개발 전에 정한 const/GPR 비교만 수행함. 여기서는 비율 모델도 다시 학습하지 않음.
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
    # 예측을 저장해 후속 보고서와 그림은 테스트셋 재평가 없이 만들 수 있게 함.
    preds.to_csv(out/'final_test_predictions.csv', index=False)
    save_json(out/'final_pressure_metrics.json', report)
    write_receipt(out,'finalization_complete.json',{'data_sha256':manifest['data_sha256']},
                  files=[*out.glob('final_*.csv'),out/'final_pressure_metrics.json',
                         out/'final_evaluation_started.json',*out.glob('*__parity.png')])
    print('Final test complete. Do not change models based on this holdout.', flush=True)


# 5. 명령줄 실행
def main():
    """개발 실행과 최종 평가를 명령으로 구분해 테스트 단계의 설정 변경을 막았음.
    """
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
    parser.add_argument('--blas-threads',type=int)
    parser.add_argument('--gpr-max-rows', type=int)
    parser.add_argument('--estimate-time', action='store_true')
    parser.add_argument('--execution-plan',type=Path)
    parser.add_argument('--hyd-model',choices=['const','gpr'])
    parser.add_argument('--ann-ref-config',type=Path)
    args = parser.parse_args()
    out = args.out or ROOT/'outputs'/('full10k' if args.config else 'pilot')
    if args.stage == 'develop':
        development(args.data, out, args.quick, args.learning_curves, args.physics_feature,
                    config=args.config, budget=args.budget, cv_repeats=args.cv_repeats,
                    n_jobs=args.n_jobs, gpr_max_rows=args.gpr_max_rows, estimate_time=args.estimate_time,
                    hyd_model=args.hyd_model,ann_ref_config=args.ann_ref_config,blas_threads=args.blas_threads,execution_plan=args.execution_plan)
    else:
        if args.estimate_time or args.quick or args.learning_curves or any(value is not None for value in
                [args.physics_feature, args.budget, args.cv_repeats, args.n_jobs, args.gpr_max_rows,args.hyd_model,args.ann_ref_config,args.blas_threads,args.execution_plan]):
            parser.error('finalize uses saved effective settings; only --config, --data and --out are accepted.')
        finalize(args.data, out, config=args.config)


if __name__ == '__main__':
    main()
