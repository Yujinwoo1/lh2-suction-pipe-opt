"""저장된 합성 응답 측정값으로 모델 성능과 별개인 실행 비용을 비교한다.

파이프라인 단계: 실행 관리
입력: 두 벤치마크의 시간 측정 CSV와 configs/full10k.json을 읽는다.
출력: 예상 시간표와 자원 계획, 비용 기준의 예산 판단을 저장한다.
최종 결과에는 쓰이지 않는 보조 도구이며, execution_plan.py를 호출해 자원과 시간 계획을 계산한다."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from execution_plan import resource_plan,search_wall_seconds,walltime_decision
from training_support import settings


def proxy_estimates(single,native,others,cfg,gpr_rows,budget,snapshot,mode='parallel_one_thread'):
    """학습을 다시 하지 않고 develop의 단계별 실행 시간을 추정한다.
    2,000행 단일 스레드 시간은 실측값이 아니라 native 측정 비율로 환산한 값이다.
    """
    config={**cfg,'gpr_max_rows':gpr_rows,'budget':budget}
    observed=native.set_index('fit_rows')
    if mode=='native_serial':
        config['blas_threads']=int(observed.loc[gpr_rows,'native_blas_threads'])
        fit=float(observed.loc[gpr_rows,'fit_seconds'])
    else:
        ratio=1 if gpr_rows==3000 else float(observed.loc[gpr_rows,'fit_seconds']/observed.loc[3000,'fit_seconds'])
        fit=float(single.fit_seconds.mean())*ratio
    pred100k=float(single.predict_100k_seconds.mean())*gpr_rows/3000
    rows=[]
    def add(target,name,fit_time,validation_time,predict100k,candidates,sensitive,refit_factor):
        plan=resource_plan(config,snapshot,max_tasks=max(1,candidates)*5)
        jobs=plan['effective_jobs']
        search=0 if target=='hyd_ratio' else search_wall_seconds(fit_time,validation_time,candidates,5,jobs)
        report=math.ceil(5/jobs)*(fit_time+validation_time)
        refit=fit_time*refit_factor
        bench=0 if target=='hyd_ratio' else .3*predict100k
        rows.append(dict(target=target,model=name,candidates=candidates,tunable=sensitive or (name=='gpr' and target!='hyd_ratio'),
            planned_tuning_candidates=candidates if sensitive or (name=='gpr' and target!='hyd_ratio') else 0,budget_sensitive=sensitive,
            cv_folds=5,search_workers=jobs,blas_threads=config['blas_threads'],
            one_fold_fit_seconds=fit_time,one_fold_predict_seconds=validation_time,
            estimated_search_seconds=search,estimated_reporting_cv_seconds=report,
            estimated_refit_seconds=refit,estimated_benchmark_seconds=bench,
            estimated_develop_seconds=search+report+refit+bench))
    for target in ['Q_leak_W','dp_thermal_Pa']:
        add(target,'gpr',fit,pred100k*.016,pred100k,6,False,1)
        for row in others.to_dict('records'):
            if row['model']=='physics_baseline' and target!='Q_leak_W':continue
            add(target,row['model'],row['fit_seconds'],row['predict_fold_seconds'],row['predict_100k_seconds'],
                budget if row['tunable'] else 1,bool(row['tunable']),1.25)
    if 'gpr' in cfg.get('finalize_pressure_comparisons',[]):
        add('hyd_ratio','gpr',fit,pred100k*.016,pred100k,1,False,1)
    return pd.DataFrame(rows)


def report(single_path,native_path,others_path,config_path,out):
    single,native,others=(pd.read_csv(path) for path in [single_path,native_path,others_path])
    declared,cfg=settings(config_path)
    cfg['finalize_pressure_comparisons']=declared.get('finalize_pressure_comparisons',[])
    snapshot=dict(available_mib=float(native.available_before_fit_mib.min()),
                  cpus=int(native.affinity_cpus.min()),native_blas_threads=int(native.native_blas_threads.max()))
    out.mkdir(parents=True,exist_ok=False)
    summary=[];details=[];plans={}
    for n in [3000,2000]:
        plans[n]=resource_plan({**cfg,'gpr_max_rows':n},snapshot)
        for budget in [6,20,50]:
            estimates=proxy_estimates(single,native,others,cfg,n,budget,snapshot)
            details.append(estimates.assign(gpr_rows=n,budget=budget))
            main=estimates[(estimates.model=='gpr') & (estimates.target!='hyd_ratio')].estimated_develop_seconds.sum()
            hyd=estimates[estimates.target=='hyd_ratio'].estimated_develop_seconds.sum()
            total=estimates.estimated_develop_seconds.sum()
            summary.append(dict(gpr_rows=n,budget=budget,cv_folds=5,gpr_candidates=6,
                other_tuning_candidates=budget,blas_threads=cfg['blas_threads'],fold_workers=plans[n]['effective_jobs'],
                main_gpr_wall_hours=main/3600,hydraulic_wall_hours=hyd/3600,
                other_models_wall_hours=(total-main-hyd)/3600,total_wall_hours=total/3600))
    table=pd.DataFrame(summary)
    table.to_csv(out/'walltime_estimates.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(out/'model_estimates.csv',index=False)
    base=proxy_estimates(single,native,others,cfg,3000,6,snapshot)
    reduced=proxy_estimates(single,native,others,cfg,2000,6,snapshot)
    decision=walltime_decision(base,declared['budget_rule'],reduced)
    comparison=[]
    for mode in ['parallel_one_thread','native_serial']:
        estimates=proxy_estimates(single,native,others,cfg,3000,50,snapshot,mode)
        comparison.append(dict(mode=mode,total_wall_hours=estimates.estimated_develop_seconds.sum()/3600))
    metadata=dict(config=declared,resource_snapshot=snapshot,plans=plans,
        proxy_policy_result_not_actual_data_decision=decision,thread_strategy_proxy=comparison,
        note='Synthetic response proxy; 2000-row one-thread time transferred by native timing ratio, no measured concurrent fits. Old other-model one-thread timings reused. Full data labels/validation scores never used. Excludes learning curves/I/O/plots and parallel contention. Native strategy reuses other-model one-thread timings, not native measurements. Budget rule must be reapplied using actual merged --estimate-time.')
    (out/'execution_plan.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(table.to_string(index=False),flush=True)
    print('Resource plans: '+json.dumps(plans),flush=True)
    print('Thread strategy proxy: '+json.dumps(comparison),flush=True)
    return table,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/full10k.json'))
    parser.add_argument('--single',type=Path,default=Path('outputs/gpr_3000_benchmark/gpr_timing.csv'))
    parser.add_argument('--native',type=Path,default=Path('outputs/gpr_native_benchmark/native_fit_timing.csv'))
    parser.add_argument('--others',type=Path,default=Path('outputs/gpr_3000_benchmark/other_model_proxy_timing.csv'))
    parser.add_argument('--out',type=Path,default=Path('outputs/full10k_execution_plan'))
    args=parser.parse_args();report(args.single,args.native,args.others,args.config,args.out)
