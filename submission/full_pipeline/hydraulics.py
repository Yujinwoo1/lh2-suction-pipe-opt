"""수력 압력과 강체 압력의 비율은 약 1.74–1.85로 달라 1.8 고정값과 GPR 보정을 비교한다.

파이프라인 단계: 수력 보정
입력: 제약 경계의 판정 차이를 확인하려고 개발 자료와 같은 fold의 열적 압력 예측을 받는다.
출력: 비율 모델과 부분표본 ID, 개발 CV 예측 및 압력 비교표를 저장한다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from joblib import Parallel,delayed,parallel_config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training_support import SubsampledTargetRegressor
from execution_plan import release_workers

RATIO_FILE='hyd_ratio__gpr.joblib'


def ratio_estimator(max_rows=3000):
    kernel=ConstantKernel(1,(1e-3,1e3))*RBF(np.ones(6))+WhiteKernel(1e-5,(1e-9,1e-1))
    return SubsampledTargetRegressor(regressor=Pipeline([
        ('scale',StandardScaler()),('model',GaussianProcessRegressor(kernel=kernel,
            alpha=1e-7,normalize_y=True,random_state=2026))]),max_rows=max_rows)


def hydraulic_mode(manifest, override=None):
    return override or manifest.get('effective_config',{}).get('hyd_model',manifest.get('hyd_model','const'))


def load_ratio_model(directory,manifest,override=None):
    mode=hydraulic_mode(manifest,override)
    if mode not in ['const','gpr']: raise ValueError('hyd_model must be const or gpr.')
    if mode=='const': return mode,None
    path=Path(directory)/manifest.get('hydraulic',{}).get('ratio_model_file',RATIO_FILE)
    if not path.exists(): raise ValueError('Hydraulic GPR was not trained in this model directory.')
    return mode,joblib.load(path)


def predicted_ratio(model,X):
    result=np.full(len(X),1.8) if model is None else model.predict(X)
    if not np.isfinite(result).all() or (result<=0).any():
        raise ValueError('Hydraulic ratio predictions must be finite and positive.')
    return result


def pressure_metrics(actual,prediction):
    actual,prediction=np.asarray(actual),np.asarray(prediction)
    boundary=(actual>=38300)&(actual<=58300)
    truth,pred=actual>48300,prediction>48300
    return dict(n=len(actual),mape_pct=float(np.mean(100*np.abs(prediction/actual-1))),
        boundary_n=int(boundary.sum()),
        boundary_mape_pct=float(np.mean(100*np.abs(prediction[boundary]/actual[boundary]-1))) if boundary.any() else None,
        boundary_mismatch_count=int(np.sum((truth!=pred)&boundary)),
        false_safe_count=int(np.sum(truth&~pred)),false_unsafe_count=int(np.sum(~truth&pred)),
        boundary_false_safe_count=int(np.sum(truth&~pred&boundary)),
        boundary_false_unsafe_count=int(np.sum(~truth&pred&boundary)),
        constraint_agreement=float(np.mean(truth==pred)))



def hydraulic_fold(model,X,target,a,b,thermal_values,thermal_estimator,thermal_prediction):
    ratio=clone(model)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        ratio.fit(X.iloc[a],target[a])
        if thermal_prediction is None:
            thermal=clone(thermal_estimator).fit(X.iloc[a],thermal_values[a])
            thermal_prediction=thermal.predict(X.iloc[b])
    return dict(selected=a[ratio.sample_indices_],ratios=predicted_ratio(ratio,X.iloc[b]),
                thermal_prediction=thermal_prediction,warnings=[str(w.message) for w in captured])


def hydraulic_cv(dev,features,folds,thermal_estimator,rigid,out,max_rows=3000,
                 thermal_oof=None,warning_records=None,n_jobs=1,blas_threads=None):
    """수력 보정의 차이만 비교하도록 같은 개발 fold의 열적 압력 예측을 사용한다.
    테스트셋은 보정 모델 학습이나 비교에 사용하지 않는다."""
    warning_records=[] if warning_records is None else warning_records
    X=dev[features]; target=dev.dp_hyd_Pa.to_numpy()/dev.dp_rigid_Pa.to_numpy()
    model=ratio_estimator(max_rows)
    rows=[]; samples=[]
    with parallel_config(backend='loky',inner_max_num_threads=blas_threads):
        fitted=Parallel(n_jobs=n_jobs,pre_dispatch=n_jobs)(
            delayed(hydraulic_fold)(model,X,target,a,b,dev.dp_thermal_Pa.to_numpy(),thermal_estimator,
                None if thermal_oof is None else thermal_oof[fold]) for fold,(a,b) in enumerate(folds))
    release_workers(n_jobs)
    for fold,((a,b),result) in enumerate(zip(folds,fitted)):
        warning_records.extend(f'hydraulic/fold{fold}: {message}' for message in result['warnings'])
        samples.extend(dict(stage='cv_train',fold=fold,case_id=case_id) for case_id in dev.iloc[result['selected']].case_id.tolist())
        ratios=result['ratios'];thermal_prediction=result['thermal_prediction']
        for local,index in enumerate(b):
            rows.append(dict(fold=fold,case_id=dev.iloc[index].case_id,
                ratio_actual=target[index],ratio_gpr=ratios[local],rigid_Pa=rigid[index],
                thermal_oof_Pa=thermal_prediction[local],thermal_actual_Pa=dev.iloc[index].dp_thermal_Pa,
                pressure_actual_Pa=dev.iloc[index].dp_design_Pa,
                pressure_const_Pa=thermal_prediction[local]+1.8*rigid[index],
                pressure_gpr_Pa=thermal_prediction[local]+ratios[local]*rigid[index]))
    predictions=pd.DataFrame(rows)
    summary=[]
    for method in ['const','gpr']:
        ratios=np.full(len(predictions),1.8) if method=='const' else predictions.ratio_gpr.to_numpy()
        row=dict(method=method,ratio_mape_pct=float(np.mean(100*np.abs(ratios/predictions.ratio_actual-1))),
            **pressure_metrics(predictions.pressure_actual_Pa,predictions[f'pressure_{method}_Pa']))
        row['unique_boundary_mismatch_cases']=int(predictions.loc[
            predictions.pressure_actual_Pa.between(38300,58300)&
            ((predictions[f'pressure_{method}_Pa']>48300)!=(predictions.pressure_actual_Pa>48300)), 'case_id'].nunique())
        # 열적 압력의 정답을 넣어 수력 근사에서 생긴 오차를 따로 확인함.
        control=pressure_metrics(predictions.pressure_actual_Pa,
                                 predictions.thermal_actual_Pa+ratios*predictions.rigid_Pa)
        row['true_thermal_boundary_mismatch_count']=control['boundary_mismatch_count']
        summary.append(row)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always'); model.fit(X,target)
    warning_records.extend(f'hydraulic/refit: {w.message}' for w in captured)
    samples.extend(dict(stage='refit',fold=-1,case_id=case_id)
                   for case_id in dev.iloc[model.sample_indices_].case_id.tolist())
    joblib.dump(model,out/RATIO_FILE)
    pd.DataFrame(samples).to_csv(out/'hydraulic_subsamples.csv',index=False)
    predictions.to_csv(out/'hydraulic_cv_predictions.csv',index=False)
    table=pd.DataFrame(summary); table.to_csv(out/'hydraulic_cv_comparison.csv',index=False)
    return dict(ratio_model_file=RATIO_FILE,fit_rows=len(model.sample_indices_),
                model='six-input fixed-structure GPR; no grid tuning; kernel optimized on training rows only',
                fitted_kernel=str(model.regressor_.named_steps['model'].kernel_),
                cv_note='Repeated development validation observations, not unique-case counts; thermal hyperparameters chosen by development CV.'),table


def run(models_path,data,out):
    import lh2_pipeline as p
    manifest=json.loads((models_path/'manifest.json').read_text())
    if hashlib.sha256(data.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Data hash differs from development.')
    df=p.load_data(data)
    cfg=manifest.get('effective_config',{})
    train,_=p.train_test_split(np.arange(len(df)),test_size=cfg.get('test_size',.2),
                               stratify=df.vaporized,random_state=2026)
    splits=pd.read_csv(models_path/'split.csv')
    if set(df.iloc[train].case_id)!=set(splits.loc[splits.split=='development','case_id']):
        raise ValueError('Development split mismatch.')
    saved=np.load(models_path/'cv_indices.npz')
    folds=[(saved[f'{i}_train'],saved[f'{i}_val']) for i in range(manifest['cv_folds'])]
    thermal=joblib.load(models_path/('dp_thermal_Pa__'+manifest['selected']['dp_thermal_Pa']+'.joblib'))
    out.mkdir(parents=True,exist_ok=False)
    warning_records=[]
    info,table=hydraulic_cv(df.iloc[train].reset_index(drop=True),p.FEATURES,folds,thermal,
        p.rigid_pressure(df.iloc[train]),out,cfg.get('gpr_max_rows',3000),warning_records=warning_records)
    print(table.to_string(index=False),flush=True)
    p.save_json(out/'hydraulic_manifest.json',dict(source_models=str(models_path),
        data_sha256=manifest['data_sha256'],seed=2026,development_rows=len(train),
        cv_folds=len(folds),hydraulic=info,warnings=sorted(set(warning_records)),finalize_executed=False))
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.models,args.data,args.out)
