"""중단된 단계부터 이어 실행하되 finalize는 시작 기록이 있으면 재시도하지 않음"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from artifact_state import atomic_json,sha256,write_receipt,verify_receipt

ROOT=Path(__file__).resolve().parent
STAGES=['validate','estimate','develop','finalize','optimize_const','optimize_gpr','summary']
DESTINATIONS={'validate':'validation','estimate':'estimate','develop':'models',
              'finalize':'models','optimize_const':'optimization/const','optimize_gpr':'optimization/gpr','summary':'report'}
RECEIPTS={'validate':'stage_complete.json','estimate':'stage_complete.json','develop':'development_complete.json',
          'finalize':'finalization_complete.json','optimize_const':'optimization_complete.json',
          'optimize_gpr':'optimization_complete.json','summary':'summary_complete.json'}


def timestamp():return datetime.now().astimezone().isoformat(timespec='seconds')


def log(root,event):
    with (root/'run.log').open('a',encoding='utf-8') as output:
        output.write(f'[{timestamp()}] {event}\n');output.flush()


def complete(root,stage):
    return verify_receipt(root/DESTINATIONS[stage],RECEIPTS[stage])


def final_retry_forbidden(root,state):
    if complete(root,'finalize'):return False
    entry=state.get('stages',{}).get('finalize',{})
    return bool(entry.get('attempts')) or (root/'models/final_evaluation_started.json').exists()


def save_state(root,state):atomic_json(root/'run_state.json',state)


def initialize(root,data,quick):
    identity=dict(data_path=str(data),data_sha256=sha256(data),quick=quick,
        config_sha256=sha256(ROOT/'configs/full10k.json'),
        scenarios_sha256=sha256(ROOT/'configs/scenarios.json'),ann_sha256=sha256(ROOT/'configs/ann_ref.json'))
    path=root/'run_state.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state['identity']!=identity:raise ValueError('Resume requires the same data, config, scenarios, ANN spec and quick mode.')
        for filename,expected in [('run_config.json',identity['config_sha256']),('run_scenarios.json',identity['scenarios_sha256']),('run_ann_ref.json',identity['ann_sha256'])]:
            if sha256(root/filename)!=expected:raise ValueError('Frozen workflow configuration changed.')
        return state
    if any(p.name not in ['.run.lock','launcher.log','run.log'] for p in root.iterdir()):
        raise FileExistsError('Refusing to adopt a nonempty output directory without workflow state.')
    for src,dest in [('full10k.json','run_config.json'),('scenarios.json','run_scenarios.json'),('ann_ref.json','run_ann_ref.json')]:
        shutil.copy2(ROOT/'configs'/src,root/dest)
    state=dict(identity=identity,started_at=timestamp(),started_unix=time.time(),status='running',stages={})
    save_state(root,state);return state


def execute_stage(root,stage,work):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-stage',stage,'--run-root',str(root),'--work-dir',str(work)]
    attempt=json.loads((root/'run_state.json').read_text())['stages'][stage]['attempts']
    stage_log=root/'logs'/f'{stage}_{attempt:03d}.log';stage_log.parent.mkdir(exist_ok=True)
    with stage_log.open('w',encoding='utf-8') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                output.write(line);output.flush();log(root,f'{stage}: {line.rstrip()}')
            result=child.wait()
        except BaseException:
            child.terminate();child.wait();raise
    if result:raise RuntimeError(f'{stage} failed with exit {result}; see {stage_log}')


def run(data,root,quick=False):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Workflow already running; no duplicate launch allowed.')
        state=initialize(root,data,quick)
        if state.get('status')=='complete' and all(complete(root,stage) for stage in STAGES):
            log(root,'RUN_ALREADY_COMPLETE; all stages skipped, original completion time retained')
            return state
        log(root,'RUN_START '+json.dumps(state['identity'],ensure_ascii=False))
        try:
            from training_support import settings
            from execution_plan import resource_plan,resource_snapshot
            _,cfg=settings(root/'run_config.json',quick=quick)
            snap=resource_snapshot();plan=resource_plan(cfg,snap)
            state['start_resource_plan']=plan;save_state(root,state)
            log(root,'RAM_CHECK '+json.dumps(plan))
            requested=cfg['n_jobs'] if cfg['n_jobs']>0 else snap['cpus']
            if plan['effective_jobs']<requested:
                log(root,f'WARNING worker request {requested} reduced to {plan["effective_jobs"]} by RAM/CPU admission; each stage rechecks available memory.')
            for stage in STAGES:
                entry=state['stages'].get(stage,{})
                if complete(root,stage):
                    if entry.get('status')!='complete':
                        entry.update(status='complete',recovered=True,ended_at=timestamp());state['stages'][stage]=entry;save_state(root,state)
                    log(root,'STAGE_SKIP '+stage);continue
                if entry.get('status')=='complete':raise RuntimeError(f'Completed {stage} artifacts are missing/corrupt; refusing silent replacement.')
                if stage=='finalize' and final_retry_forbidden(root,state):
                    raise RuntimeError('Final holdout was already started; automatic finalize retry is forbidden, including a failed/interrupted attempt.')
                # 상태 기록 직전에 중단됐을 수 있어 완료된 산출물을 먼저 확인함
                prior=entry.get('work_dir')
                if stage!='finalize' and prior and verify_receipt(prior,RECEIPTS[stage]):
                    destination=root/DESTINATIONS[stage];destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists():raise FileExistsError(destination)
                    Path(prior).rename(destination)
                    entry.update(status='complete',recovered=True,ended_at=timestamp());state['stages'][stage]=entry;save_state(root,state)
                    log(root,'STAGE_RECOVER '+stage);continue
                attempt=entry.get('attempts',0)+1
                work=root/'work'/f'{stage}_{attempt:03d}'
                work.parent.mkdir(exist_ok=True)
                state['stages'][stage]=dict(status='running',attempts=attempt,started_at=timestamp(),work_dir=str(work))
                state['status']='running';save_state(root,state)  # durable before finalize child can open holdout
                started=time.monotonic();log(root,f'STAGE_START {stage} attempt={attempt}')
                execute_stage(root,stage,work)
                if stage!='finalize':
                    if not verify_receipt(work,RECEIPTS[stage]):raise RuntimeError(f'{stage} has no valid completion receipt.')
                    destination=root/DESTINATIONS[stage];destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists():raise FileExistsError(destination)
                    work.rename(destination)
                if not complete(root,stage):raise RuntimeError(f'{stage} completion verification failed.')
                state['stages'][stage].update(status='complete',ended_at=timestamp(),elapsed_seconds=time.monotonic()-started)
                save_state(root,state);log(root,f'STAGE_END {stage} elapsed_seconds={time.monotonic()-started:.3f}')
            state.update(status='complete',ended_at=timestamp(),total_elapsed_seconds=time.time()-state['started_unix'])
            save_state(root,state);log(root,f'RUN_END complete total_elapsed_seconds={state["total_elapsed_seconds"]:.3f}')
        except BaseException as error:
            for entry in state['stages'].values():
                if entry.get('status')=='running':entry.update(status='failed',ended_at=timestamp(),error=str(error))
            state.update(status='failed',last_error=str(error),last_failed_at=timestamp())
            save_state(root,state);log(root,'RUN_FAILED '+str(error));raise
    return state


def worker(root,stage,work):
    import lh2_pipeline as p
    from training_support import read_config
    state=json.loads((root/'run_state.json').read_text());data=Path(state['identity']['data_path']);quick=state['identity']['quick']
    config=root/'run_config.json';ann=root/'run_ann_ref.json';models=root/'models'
    if sha256(data)!=state['identity']['data_sha256']:raise ValueError('Data changed during workflow.')
    if stage=='validate':
        frame=p.load_data(data);work.mkdir(parents=True,exist_ok=False);p.audit(frame,work)
        atomic_json(work/'validation.json',dict(data_sha256=sha256(data),accepted_rows=len(frame),quick=quick))
        write_receipt(work,'stage_complete.json')
    elif stage=='estimate':
        estimate=p.development(data,root/'unused_estimate_output',config=config,quick=quick,learning=True,
                               estimate_time=True,ann_ref_config=ann)
        work.mkdir(parents=True,exist_ok=False);estimate.to_csv(work/'runtime_estimate.csv',index=False)
        plan=dict(data_sha256=sha256(data),config=read_config(config),quick=quick,learning_curves=True,
                  ann_ref_sha256=sha256(ann),effective_config=estimate.attrs['effective_config'],
                  budget_decision=estimate.attrs['budget_decision'])
        atomic_json(work/'execution_plan.json',plan);write_receipt(work,'stage_complete.json')
        if plan['budget_decision'].get('target_met') is False:print('WARNING: estimated runtime still exceeds target; lowest preregistered configuration retained.',flush=True)
    elif stage=='develop':
        p.development(data,work,config=config,quick=quick,learning=True,ann_ref_config=ann,
                      execution_plan=root/'estimate/execution_plan.json')
    elif stage=='finalize':
        p.finalize(data,models,config=config)
    elif stage.startswith('optimize_'):
        import optimize
        from threadpoolctl import threadpool_limits
        mode=stage.split('_',1)[1]
        with threadpool_limits(limits=read_config(config)['blas_threads']):
            optimize.run(models,root/'run_scenarios.json',work,hyd_model=mode)
    elif stage=='summary':
        from summarize_results import summarize
        summarize(models,work,{'const':root/'optimization/const','gpr':root/'optimization/gpr'},quick=quick)
    else:raise ValueError(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',nargs='?',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--quick',action='store_true')
    parser.add_argument('--worker-stage',choices=STAGES);parser.add_argument('--run-root',type=Path);parser.add_argument('--work-dir',type=Path)
    args=parser.parse_args()
    if args.worker_stage:worker(args.run_root,args.worker_stage,args.work_dir)
    else:
        if args.data is None:parser.error('merged data path required')
        root=args.out or ROOT/'outputs'/('dryrun_full10k' if args.quick else 'full10k')
        run(args.data.expanduser().resolve(),root.expanduser().resolve(),args.quick)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
