"""메모리/CPU 한도를 넘지 않도록 사전에 정한 비용 기준으로 병렬 수를 제한함."""
import math
import os
from pathlib import Path

import joblib
from threadpoolctl import threadpool_info


def resource_snapshot():
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    available=values['MemAvailable']/1024
    # 시스템 RAM보다 컨테이너 한도가 작을 수 있어 함께 확인함.
    cap,current=Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')
    if cap.exists() and current.exists() and cap.read_text().strip()!='max':
        available=min(available,(int(cap.read_text())-int(current.read_text()))/2**20)
    cpus=min(len(os.sched_getaffinity(0)),joblib.cpu_count())
    native=max((pool['num_threads'] for pool in threadpool_info() if pool['user_api']=='blas'),default=1)
    return dict(available_mib=max(0,available),cpus=cpus,native_blas_threads=native)


def resource_plan(cfg,snapshot=None,max_tasks=None):
    snap=resource_snapshot() if snapshot is None else dict(snapshot)
    reserve=cfg.get('memory_reserve_mib',2048)
    usable=min(cfg.get('memory_limit_mib',float('inf')),max(0,snap['available_mib']-reserve))
    measured=cfg.get('fit_memory_mib',{})
    rows=cfg['gpr_max_rows']
    if measured:
        key=min(measured,key=lambda k:abs(int(k)-rows))
        peak=max(cfg.get('other_model_memory_mib',1024),measured[key]*(rows/int(key))**2)
    else:
        peak=cfg.get('other_model_memory_mib',1024)
    per_fit=peak*cfg.get('memory_safety_factor',1.25)
    ram_workers=math.floor(usable/per_fit)
    if ram_workers<1:
        raise MemoryError(f'Insufficient memory: usable {usable:.1f} MiB, reserved per fit {per_fit:.1f} MiB.')
    threads=cfg.get('blas_threads') or snap['native_blas_threads']
    cpu_workers=max(1,snap['cpus']//threads)
    requested=cfg['n_jobs'] if cfg['n_jobs']>0 else max(1,snap['cpus']+1+cfg['n_jobs'])
    jobs=min(requested,ram_workers,cpu_workers,max_tasks if max_tasks is not None else requested)
    return dict(requested_jobs=cfg['n_jobs'],effective_jobs=max(1,jobs),blas_threads=threads,
                ram_max_workers=ram_workers,cpu_max_workers=cpu_workers,
                available_mib=snap['available_mib'],reserve_mib=reserve,usable_limit_mib=usable,
                measured_peak_mib=peak,reserved_per_fit_mib=per_fit,
                planned_worker_memory_mib=max(1,jobs)*per_fit)


def candidate_parameters(name,space,budget,exhaustive_gpr=False):
    from sklearn.model_selection import ParameterGrid
    from training_support import sampled_parameters
    if name=='gpr' and space and exhaustive_gpr:
        return list(ParameterGrid(space))
    return sampled_parameters(space,budget)


def search_wall_seconds(fit_seconds,predict_seconds,candidates,folds,workers):
    return math.ceil(candidates*folds/workers)*(fit_seconds+predict_seconds)


def projected_seconds(estimates,budget):
    total=0.0
    for row in estimates.to_dict('records'):
        sensitive=row.get('budget_sensitive',row.get('tunable',False))
        count=budget if sensitive else row['candidates']
        if 'one_fold_fit_seconds' in row and 'search_workers' in row:
            search=0 if row.get('target')=='hyd_ratio' else search_wall_seconds(
                row['one_fold_fit_seconds'],row['one_fold_predict_seconds'],count,row['cv_folds'],row['search_workers'])
        else:
            search=row['estimated_search_seconds']*(budget/row['candidates'] if sensitive else 1)
        total+=row['estimated_develop_seconds']-row['estimated_search_seconds']+search
    return total


def walltime_decision(estimates,rule,reduced_estimates=None):
    totals={str(b):projected_seconds(estimates,b) for b in [6,20,50]}
    limit=rule['max_wall_hours']*3600
    within=lambda seconds: seconds<=limit or math.isclose(seconds,limit,rel_tol=0,abs_tol=1e-6)
    reduced=not within(totals['6'])
    selected_totals=totals
    if reduced and reduced_estimates is not None:
        selected_totals={str(b):projected_seconds(reduced_estimates,b) for b in [6,20,50]}
    chosen=next((b for b in [50,20] if within(selected_totals[str(b)])),6)
    return dict(applied_budget=chosen,applied_gpr_max_rows=rule['fallback_gpr_rows'] if reduced else rule['initial_gpr_rows'],
        row_reduction_required=reduced,reduced_estimate_required=reduced and reduced_estimates is None,
        initial_wall_seconds=totals,selected_wall_seconds=selected_totals,
        target_met=within(selected_totals[str(chosen)]) and not (reduced and reduced_estimates is None),
        reason='Cost-only preregistered wall-clock rule: largest of 50/20/6 within 4h; reduce GPR rows when budget-6 at 3000 exceeds 4h. No solver performance or holdout results used.')


def release_workers(jobs):
    """대기 중인 worker도 메모리를 잡고 있어 refit 전에 종료함.
    남은 worker 메모리와 새 GPR 학습 메모리가 겹치는 것을 막기 위함.
    """
    if jobs>1:
        from joblib.externals.loky import get_reusable_executor
        executor=get_reusable_executor(reuse=True)
        if hasattr(executor,'terminate'):
            executor.terminate(kill_workers=True)
        else:
            executor.shutdown(wait=True,kill_workers=True)


def reschedule_estimates(estimates,cfg,snapshot=None):
    """부분표본 크기로 병렬 수가 달라지면 기존 실측 시간을 새 실행 순서에 맞춰 계산함."""
    if estimates.empty:return estimates.copy()
    rows=[]
    for row in estimates.to_dict('records'):
        plan=resource_plan(cfg,snapshot,max_tasks=row['candidates']*row['cv_folds'])
        jobs=plan['effective_jobs']
        search=0 if row.get('target')=='hyd_ratio' else search_wall_seconds(
            row['one_fold_fit_seconds'],row['one_fold_predict_seconds'],row['candidates'],row['cv_folds'],jobs)
        reporting=math.ceil(row['cv_folds']/jobs)*(row['one_fold_fit_seconds']+row['one_fold_predict_seconds'])
        row['estimated_develop_seconds']+=search+reporting-row['estimated_search_seconds']-row['estimated_reporting_cv_seconds']
        row.update(search_workers=jobs,estimated_search_seconds=search,
                   estimated_reporting_cv_seconds=reporting,blas_threads=plan['blas_threads'])
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)
