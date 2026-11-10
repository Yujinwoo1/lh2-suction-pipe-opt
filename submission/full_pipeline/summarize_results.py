"""저장된 결과만 읽어 보고서를 작성함. 재학습이나 테스트셋 재평가는 하지 않음."""
import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lh2_pipeline import TARGETS,plt
from artifact_state import write_receipt,sha256

UNITS={'Q_leak_W':'W','dp_thermal_Pa':'Pa'}


def save_figure(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=300,facecolor='white',transparent=False);plt.close(fig)


def model_table(cv,manifest):
    result=cv.copy();counts=[]
    for row in cv.itertuples():
        entry=manifest['models'].get(row.target+'__'+row.model,{})
        count=entry.get('tuning_candidates',getattr(row,'actual_candidates',None))
        counts.append(count)
    result['actual_candidates']=counts
    result['unit']=result.target.map(UNITS)
    result['evaluation_scope']='development CV, used for model selection'
    for metric in ['r2','rmse','mape_pct']:
        result[metric+'_mean_pm_std']=result.apply(lambda r:f'{r[metric+"_mean"]:.6g} ± {r[metric+"_std"]:.6g}',axis=1)
    return result


def hyperparameter_table(models,manifest):
    rows=[]
    for key,info in manifest['models'].items():
        target,name=key.split('__',1)
        model=joblib.load(models/(key+'.joblib'))
        learner=model.regressor_.named_steps['model'] if hasattr(model,'regressor_') else model
        params=learner.get_params(deep=False)
        if not params:params={'formula':'P0-2 fixed physics formula; no tuning'}
        if hasattr(learner,'kernel_'):params['fitted_kernel_']=str(learner.kernel_)
        params['actual_fit_rows']=info.get('fit_rows')
        params['training_row_cap']=getattr(model,'max_rows',None)
        params['physics_feature']=hasattr(model,'regressor_') and 'physics_feature' in model.regressor_.named_steps
        for parameter,value in params.items():
            rows.append(dict(target=target,model=name,parameter=parameter,
                             value=json.dumps(value,ensure_ascii=False,default=str),source='development-fitted model'))
    return pd.DataFrame(rows)


def parity_figure(predictions,manifest,path):
    fig,axes=plt.subplots(2,2,figsize=(10,8),squeeze=False)
    for i,target in enumerate(TARGETS):
        actual=predictions[target+'__actual'].to_numpy()
        for j,name in enumerate([manifest['selected'][target],'mlp']):
            ax=axes[i,j];predicted=predictions[target+'__'+name].to_numpy()
            ax.scatter(actual,predicted,s=18,alpha=.8)
            low=min(actual.min(),predicted.min());high=max(actual.max(),predicted.max())
            ax.plot([low,high],[low,high],'--',color='gray')
            if low>0:ax.set(xscale='log',yscale='log')
            ax.set(xlabel=f'Solver {target} [{UNITS[target]}]',ylabel=f'Prediction [{UNITS[target]}]',
                   title=f'{target}: {name} ({"selected" if j==0 else "MLP comparator"})')
    save_figure(fig,path)


def learning_figure(curve,path):
    curve=curve.copy()
    if 'status' in curve:curve=curve[curve.status=='evaluated']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    records=[]
    for ax,target in zip(axes,TARGETS):
        for name,frame in curve[curve.target==target].groupby('model'):
            agg=frame.groupby('n').rmse.agg(['mean','std','count'])
            ax.errorbar(agg.index,agg['mean'],yerr=agg['std'].fillna(0),label=name,marker='o',capsize=2)
            records.extend(dict(target=target,model=name,n=int(n),validation_rmse_mean=r['mean'],
                                validation_rmse_std=r['std'],repeats=int(r['count'])) for n,r in agg.iterrows())
        ax.set(xscale='log',yscale='log',xlabel='Training samples n [rows]',
               ylabel=f'Development validation RMSE [{UNITS[target]}]',title=target)
        ax.legend(fontsize=7)
    save_figure(fig,path)
    return pd.DataFrame(records)


def loss_figure(models,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));frames=[]
    for ax,target in zip(axes,TARGETS):
        frame=pd.read_csv(models/(target+'__mlp_history.csv'))
        ax.plot(frame.iteration,frame.training_loss_log_target)
        ax.set(xlabel='Iteration [count]',ylabel='Training loss [log10-space objective]',title=target)
        frames.append(frame.assign(target=target))
    save_figure(fig,path)
    return pd.concat(frames,ignore_index=True)


def cv_comparison(old,current,old_name='pilot',new_name='current'):
    metrics=[key for key in ['r2_mean','r2_std','rmse_mean','rmse_std','mape_pct_mean','mape_pct_std','log10_rmse_mean','log10_rmse_std'] if key in old and key in current]
    result=old[['target','model',*metrics]].merge(current[['target','model',*metrics]],on=['target','model'],
             suffixes=('_'+old_name,'_'+new_name),validate='one_to_one')
    for metric in ['rmse_mean','mape_pct_mean','log10_rmse_mean']:
        if metric in metrics:result[metric+'_change']=result[metric+'_'+new_name]-result[metric+'_'+old_name]
    result['comparison_scope']='saved development CV; separate experiments, not a final-holdout comparison'
    return result


def design_shifts(previous,current,hyd_model):
    rows=[]
    for scenario,b in current.groupby('scenario'):
        a=previous[previous.scenario==scenario]
        pairs=[]
        for role in ['A','B','C']:
            if not a[a.role==role].empty and not b[b.role==role].empty:
                pairs.append((a[a.role==role].iloc[0],b[b.role==role].iloc[0],'same_role'))
        ae,be=a[a.role=='extra'].reset_index(drop=True),b[b.role=='extra'].reset_index(drop=True)
        if len(ae) and len(be):
            costs=np.linalg.norm((ae[['D_mm','t_ins_mm']].to_numpy()[:,None]-be[['D_mm','t_ins_mm']].to_numpy()[None,:])/[38,23],axis=2)
            i,j=linear_sum_assignment(costs)
            pairs.extend((ae.iloc[x],be.iloc[y],'minimum_total_normalized_design_distance') for x,y in zip(i,j))
        for old,new,matching in pairs:
            row=dict(hyd_model=hyd_model,scenario=scenario,role=new.role,pilot_point_id=old.point_id,
                     current_point_id=new.point_id,matching=matching,pilot_hyd_model='const')
            for key in ['D_mm','t_ins_mm']:
                row.update({key+'_pilot':old[key],key+'_current':new[key],
                            'delta_'+key:new[key]-old[key],'abs_delta_'+key:abs(new[key]-old[key])})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(out=Path('outputs/pilot'),report=None,opt_dirs=None,pilot=Path('outputs/pilot'),
              pilot_opt=Path('outputs/opt_pilot'),physics_base=Path('outputs/pilot_v2'),
              physics_feature=Path('outputs/pilot_v3'),quick=False):
    out=Path(out);report=Path(report) if report else out/'report'
    report.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((out/'manifest.json').read_text())
    cv=pd.read_csv(out/'cv_summary.csv');test=pd.read_csv(out/'final_test_metrics.csv')
    pd.DataFrame({'warning':manifest.get('warnings',[])}).to_csv(report/'T_warnings.csv',index=False)
    predictions=pd.read_csv(out/'final_test_predictions.csv');curve=pd.read_csv(out/'learning_curve.csv')
    model_table(cv,manifest).to_csv(report/'T1_model_comparison.csv',index=False)
    hyperparameter_table(out,manifest).to_csv(report/'T2_hyperparameters.csv',index=False)
    test.to_csv(report/'T_final_test_metrics.csv',index=False)
    parity_figure(predictions,manifest,report/'F1_parity.png')
    learning_figure(curve,report/'F2_learning_curves.png').to_csv(report/'F2_learning_curve_statistics.csv',index=False)
    curve.to_csv(report/'F2_learning_curve_records.csv',index=False)
    loss_figure(out,report/'F3_mlp_loss.png').to_csv(report/'F3_mlp_history.csv',index=False)
    opt_dirs=opt_dirs or {};candidates=[];movements=[];notes=[]
    for mode,directory in opt_dirs.items():
        directory=Path(directory)
        shutil.copy2(directory/'F4.png',report/f'F4_{mode}.png')
        shutil.copy2(directory/'T3_candidates.csv',report/f'T3_{mode}.csv')
        for name in ['verification_points.csv','verification_inputs.csv','hv_summary.csv']:
            shutil.copy2(directory/name,report/f'{mode}_{name}')
        candidates.append(pd.read_csv(directory/'T3_candidates.csv').assign(hyd_model=mode))
        if (pilot_opt/'verification_points.csv').exists():
            movements.append(design_shifts(pd.read_csv(pilot_opt/'verification_points.csv'),
                                          pd.read_csv(directory/'verification_points.csv'),mode))
    if candidates:
        pd.concat(candidates,ignore_index=True).to_csv(report/'T3_candidates.csv',index=False)
        mode=manifest.get('effective_config',{}).get('hyd_model','const')
        shutil.copy2(report/f'F4_{mode}.png',report/'F4.png')
    if (pilot/'cv_summary.csv').exists():
        cv_comparison(pd.read_csv(pilot/'cv_summary.csv'),cv).to_csv(report/'T_pilot_full_model_errors.csv',index=False)
    else:
        pd.DataFrame([dict(status='unavailable',reason='Pilot CV results not supplied; no retrospective fitting performed')]).to_csv(report/'T_pilot_full_model_errors.csv',index=False)
    if movements and any(not f.empty for f in movements):
        shifts=pd.concat(movements,ignore_index=True)
        shifts.to_csv(report/'T_pilot_full_point_shifts.csv',index=False)
        shifts.groupby(['hyd_model','scenario']).agg(matched_points=('role','size'),
            mean_abs_delta_D_mm=('abs_delta_D_mm','mean'),max_abs_delta_D_mm=('abs_delta_D_mm','max'),
            mean_abs_delta_t_ins_mm=('abs_delta_t_ins_mm','mean'),max_abs_delta_t_ins_mm=('abs_delta_t_ins_mm','max')).to_csv(report/'T_pilot_full_point_shift_summary.csv')
    else:
        pd.DataFrame([dict(status='unavailable',reason='Pilot verification points not supplied')]).to_csv(report/'T_pilot_full_point_shifts.csv',index=False)
        pd.DataFrame([dict(status='unavailable')]).to_csv(report/'T_pilot_full_point_shift_summary.csv',index=False)
    if (physics_base/'cv_summary.csv').exists() and (physics_feature/'cv_summary.csv').exists():
        a=json.loads((physics_base/'manifest.json').read_text());b=json.loads((physics_feature/'manifest.json').read_text())
        if a['data_sha256']!=b['data_sha256'] or (physics_base/'split.csv').read_bytes()!=(physics_feature/'split.csv').read_bytes():
            raise ValueError('Physics-feature ablation sources have different data/splits.')
        x,y=np.load(physics_base/'cv_indices.npz'),np.load(physics_feature/'cv_indices.npz')
        if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Ablation CV folds differ.')
        table=cv_comparison(pd.read_csv(physics_base/'cv_summary.csv'),pd.read_csv(physics_feature/'cv_summary.csv'),'six_inputs','physics_feature')
        table['comparison_scope']='pilot development ablation on identical data/split/CV; not a full10k causal effect'
        table.to_csv(report/'T_physics_feature_effect.csv',index=False)
    else:
        pd.DataFrame([dict(status='unavailable',reason='Existing paired pilot ablation unavailable; no new models fit')]).to_csv(report/'T_physics_feature_effect.csv',index=False)
    hyd=[]
    for name,scope in [('hydraulic_cv_comparison.csv','development OOF'),('final_pressure_comparison.csv','single final holdout')]:
        if (out/name).exists():hyd.append(pd.read_csv(out/name).rename(columns={'hyd_model':'method'}).assign(evaluation_scope=scope))
    pd.concat(hyd,ignore_index=True).to_csv(report/'T_hydraulic_effect.csv',index=False) if hyd else pd.DataFrame([dict(status='unavailable')]).to_csv(report/'T_hydraulic_effect.csv',index=False)
    notes.extend(['F2 uses development fixed-validation RMSE, never the final holdout.',
                  'CV used for tuning/selection; learning curves use development-selected hyperparameters.',
                  'Surrogate Pareto and verification candidates are not solver validation.',
                  'Physics-feature effect is the existing paired pilot development ablation; full-data ablation was not run.',
                  'Pilot/current CV can differ in data size, folds and features; point shifts include all model changes.'])
    metadata=dict(quick=quick or manifest.get('quick',False),data_sha256=manifest['data_sha256'],
        model_directory=str(out),selected=manifest['selected'],notes=notes,
        source_sha256={str(p):sha256(p) for p in [out/'manifest.json',out/'cv_summary.csv',out/'final_test_predictions.csv']})
    (report/'report_sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    lines=['# LH2 결과 요약','',f'실행 구분: {"파일럿 quick 무인 실행 확인; 1만 개 성능 결과 아님" if metadata["quick"] else "합본 실행"}',
           f'개발 {manifest["train_rows"]}행 / 최종 테스트 {manifest["test_rows"]}행, CV {manifest["cv_folds"]}개 fold.',
           f'선정 모델: {manifest["selected"]}', f'저장된 고유 경고 기록: {len(manifest.get("warnings",[]))}개 (manifest 및 T_warnings.csv 참조).', '',
           'T1은 개발 CV 평균·표준편차, 실제 후보 수, 학습·예측 시간을 담습니다. 최종 테스트 성능은 별도 T_final_test_metrics.csv입니다.',
           'F2는 개발 검증 RMSE이며 최종 테스트셋을 학습 곡선에 사용하지 않았습니다.',
           '물리 특성 효과는 기존 파일럿의 동일 분할 비교이며, 합본에서 특성을 뺀 모델을 새로 학습한 결과가 아닙니다.',
           '대리모델 Pareto와 재검증 후보는 솔버 검증 결과가 아닙니다.','', '## 생성 파일','']
    lines.extend('- '+p.name for p in sorted(report.iterdir()) if p.is_file())
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_receipt(report,'summary_complete.json',metadata)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=Path('outputs/pilot'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--opt-const',type=Path);parser.add_argument('--opt-gpr',type=Path)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    summarize(args.models,args.out,{k:v for k,v in [('const',args.opt_const),('gpr',args.opt_gpr)] if v},quick=args.quick)
