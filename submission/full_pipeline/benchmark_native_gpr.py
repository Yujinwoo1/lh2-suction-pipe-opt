"""BLAS 기본 스레드 수에서 GPR 비용만 측정한다.

파이프라인 단계: 보조 도구
입력: 입력 CSV에서 원본 입력 열만 읽는다.
출력: 학습 시간과 메모리 측정값을 저장한다. 합성 응답과 학습 모델은 저장하지 않는다.
최종 결과에는 쓰이지 않는 보조 도구이다."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import pandas as pd
from threadpoolctl import threadpool_info
from benchmark_gpr import benchmark_inputs
from training_support import sample_indices
import lh2_pipeline as p


def available_memory_mib():
    values={line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return values['MemAvailable']/1024


def measure(path,n_rows):
    X,y,_=benchmark_inputs(path)
    ids=sample_indices(len(X),n_rows)
    model,space=p.candidates(physics_feature=True)['gpr']
    model.set_params(max_rows=n_rows,regressor__model__kernel=space['regressor__model__kernel'][2],regressor__model__alpha=1e-10)
    pools=threadpool_info()
    available=available_memory_mib()
    started=time.perf_counter()
    # BLAS 기본 동작을 측정하려고 별도 스레드 제한을 두지 않음.
    model.fit(X.iloc[ids],y[ids])
    elapsed=time.perf_counter()-started
    return dict(kernel='Matern_2.5',fit_rows=n_rows,fit_calls=1,fit_seconds=elapsed,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        available_before_fit_mib=available,native_blas_threads=max(t['num_threads'] for t in pools if t['user_api']=='blas'),
        threadpools=pools,affinity_cpus=len(os.sched_getaffinity(0)),blas_limit_applied=False)


def run(path,out):
    benchmark_inputs(path)  # 잘못된 입력으로 결과 폴더가 남지 않도록 먼저 검사
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for rows in [3000,2000]:
        print(f'Native-BLAS Matern 2.5 benchmark: {rows} rows, one fit, no saved model.',flush=True)
        worker=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--inputs',str(path.resolve()),'--worker-rows',str(rows)],check=True,text=True,stdout=subprocess.PIPE)
        row=json.loads(worker.stdout);results.append(row)
        pd.DataFrame([{k:v for k,v in r.items() if k!='threadpools'} for r in results]).to_csv(out/'native_fit_timing.csv',index=False)
        (out/'native_benchmark.json').write_text(json.dumps(dict(seed=2026,purpose='synthetic timing only; no model or response files',
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results),ensure_ascii=False,indent=2))
        print(json.dumps(row),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=p.ROOT/'data/cases_10k.csv')
    parser.add_argument('--out',type=Path,default=p.ROOT/'outputs/gpr_native_benchmark')
    parser.add_argument('--worker-rows',type=int,choices=[2000,3000])
    args=parser.parse_args()
    if args.worker_rows:
        print(json.dumps(measure(args.inputs,args.worker_rows)))
    else:
        run(args.inputs,args.out)
