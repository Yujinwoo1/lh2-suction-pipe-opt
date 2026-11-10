"""정답 없는 입력으로 합성 응답을 만들어 GPR의 시간과 메모리를 측정한다.

파이프라인 단계: 보조 도구
입력: 입력 CSV와 configs/full10k.json을 읽는다.
출력: 커널별 측정값과 개발 실행 시간 추정표를 저장한다. 모델과 합성 정답은 저장하지 않는다.
최종 결과에는 쓰이지 않는 보조 도구이다."""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

import lh2_pipeline as p
from physics_baseline import physics_heat_leak
from predict import validate_inputs
from training_support import sample_indices, sampled_parameters, settings


def benchmark_inputs(path):
    raw = pd.read_csv(path)
    # 솔버 CSV를 넣어도 정답이 입력에 섞이지 않도록 원본 입력 열만 선택함.
    inputs = validate_inputs(raw[list(p.BOUNDS)])[p.FEATURES]
    if len(inputs) < 3000:
        raise ValueError('Timing benchmark requires at least 3000 original input rows.')
    ids = sample_indices(len(inputs), 3000)
    rng = np.random.default_rng(2026)
    y = 10**(np.log10(physics_heat_leak(inputs)) + rng.normal(0, .01, len(inputs)))
    return inputs, y, ids


def worker(path, kernel_index, threads):
    X, y, ids = benchmark_inputs(path)
    estimator, space = p.candidates(physics_feature=True)['gpr']
    estimator.set_params(regressor__model__kernel=space['regressor__model__kernel'][kernel_index],
                         regressor__model__alpha=1e-10, max_rows=3000)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with threadpool_limits(limits=threads):
        started = time.perf_counter()
        estimator.fit(X.iloc[ids], y[ids])
        fit_seconds = time.perf_counter()-started
        fit_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        started = time.perf_counter()
        # 예측 비용만 비교하려고 기존 입력을 반복함.
        for start in range(0, 100000, 1000):
            positions = np.arange(start, min(start+1000, 100000)) % len(X)
            estimator.predict(X.iloc[positions])
        predict_seconds = time.perf_counter()-started
    return dict(kernel=['RBF','Matern_1.5','Matern_2.5'][kernel_index], fit_rows=3000,
                fit_calls=1, alpha=1e-10, threads=threads, fit_seconds=fit_seconds,
                prediction_rows=100000, predict_100k_seconds=predict_seconds,
                baseline_peak_mib=baseline, fit_peak_rss_mib=fit_peak,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def workload_table(measured, others, n_rows, cfg):
    """직렬 실행을 가정한 비용 추정이며 실제 합본의 학습 시간은 아니다."""
    rows=[]
    validation_rows = n_rows*(1-cfg['test_size'])/cfg['cv_splits']
    # 커널별 수렴 시간이 달라 평균과 범위를 함께 남김.
    for budget in [6,20,50]:
        per_kernel=[]
        for row in measured:
            # 비용이 빠지지 않도록 출력별 탐색, 보고용 CV, refit과 예측 측정을 포함
            fit_count=2*(budget*15+15+1)
            predict_seconds=2*((budget*15+15)*validation_rows/100000+.3)*row['predict_100k_seconds']
            per_kernel.append(fit_count*row['fit_seconds']+predict_seconds)
        other_seconds=0.0
        for row in others:
            count=budget if row['tunable'] else 1
            fold_count=count*15+15
            outputs=1 if row['model']=='physics_baseline' else 2
            other_seconds += outputs*(fold_count*(row['fit_seconds']+row['predict_fold_seconds'])
                                +row['fit_seconds']*1.25 + .3*row['predict_100k_seconds'])
        hydraulic=(16*np.mean([r['fit_seconds'] for r in measured])
                   +15*validation_rows/100000*np.mean([r['predict_100k_seconds'] for r in measured])) if 'gpr' in cfg.get('finalize_pressure_comparisons',[]) else 0.0
        rows.append(dict(budget=budget, cv_folds=15, hydraulic_proxy_hours=hydraulic/3600, gpr_hours_mean=np.mean(per_kernel)/3600,
                         gpr_hours_min=min(per_kernel)/3600, gpr_hours_max=max(per_kernel)/3600,
                         other_models_hours=other_seconds/3600,
                         total_develop_hours_mean=(np.mean(per_kernel)+other_seconds+hydraulic)/3600,
                         total_develop_hours_min=(min(per_kernel)+other_seconds+hydraulic)/3600,
                         total_develop_hours_max=(max(per_kernel)+other_seconds+hydraulic)/3600))
    return pd.DataFrame(rows)


def run(path, out, threads=1):
    # 잘못된 입력으로 빈 결과 폴더가 남지 않도록 먼저 검사함.
    X,y,ids=benchmark_inputs(path)
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame({'source_row':ids}).to_csv(out/'sample_rows.csv',index=False)
    measured=[]
    for index in range(3):
        print(f'Benchmark kernel {index+1}/3: 3000 rows, one fit, synthetic y only.',flush=True)
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(index),
                                  '--inputs',str(path.resolve()),'--threads',str(threads)],
                                 check=True, text=True, stdout=subprocess.PIPE)
        # 수렴 상태를 확인할 수 있도록 경고를 stderr에 남김.
        row=json.loads(completed.stdout)
        measured.append(row)
        pd.DataFrame(measured).to_csv(out/'gpr_timing.csv',index=False)
        print(json.dumps(row),flush=True)
    declared,cfg=settings(p.ROOT/'configs/full10k.json')
    cfg['finalize_pressure_comparisons']=declared.get('finalize_pressure_comparisons',[])
    # 다른 모델의 비용을 GPR 시간으로 대신하지 않고 모델별로 따로 측정함.
    train,_=p.train_test_split(np.arange(len(X)),test_size=.2,random_state=2026)
    fold,_=p.train_test_split(train,test_size=.2,random_state=2026)
    val=np.setdiff1d(train,fold)
    others=[]
    with threadpool_limits(limits=threads):
        for name,(est,space) in p.candidates(target='Q_leak_W',physics_feature=True).items():
            if name=='gpr': continue
            model=clone(est).set_params(**sampled_parameters(space,6)[0])
            started=time.perf_counter(); model.fit(X.iloc[fold],y[fold]); fit=time.perf_counter()-started
            started=time.perf_counter(); model.predict(X.iloc[val]); pred=time.perf_counter()-started
            probe=X.iloc[np.arange(1000)%len(X)]
            started=time.perf_counter(); model.predict(probe); pred100k=(time.perf_counter()-started)*100
            others.append(dict(model=name,fit_rows=len(fold),fit_seconds=fit,
                               predict_fold_seconds=pred,predict_100k_seconds=pred100k,tunable=bool(space)))
    pd.DataFrame(others).to_csv(out/'other_model_proxy_timing.csv',index=False)
    table=workload_table(measured,others,len(X),cfg)
    table.to_csv(out/'develop_estimates.csv',index=False)
    metadata=dict(purpose='TIMING ONLY, synthetic responses; no solver regression or saved model',
        input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_rows=len(X),seed=2026,noise_log10_std=.01,threads=threads,
        prediction_batch_size=1000,kernel_fit_calls=3,models_saved=False,
        estimate_note='Synthetic target proxy, two outputs, 15 folds, n_jobs=1, no learning curves; hydraulic comparison adds one GPR proxy with 15 CV fits and one refit. Range across measured kernels. GPR alpha fixed 1e-10; actual candidate alphas and real targets change cost.',
        memory_note='Fresh child process per kernel; absolute Linux ru_maxrss MiB includes interpreter/imports; fit peak and full-run peak reported.',
        environment={'python':sys.version,'numpy':np.__version__,'sklearn':p.sklearn.__version__})
    (out/'benchmark.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(table.to_string(index=False),flush=True)
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=p.ROOT/'cases_10k.csv')
    parser.add_argument('--out',type=Path,default=p.ROOT/'outputs/gpr_3000_benchmark')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--worker',type=int,choices=range(3))
    args=parser.parse_args()
    if args.threads<1: parser.error('--threads must be positive')
    if args.worker is not None:
        print(json.dumps(worker(args.inputs,args.worker,args.threads)))
    else:
        run(args.inputs,args.out,args.threads)


if __name__=='__main__': main()
