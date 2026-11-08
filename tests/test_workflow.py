import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import lh2_pipeline as p
import full10k_runner as runner
from artifact_state import atomic_json,write_receipt,verify_receipt,sha256
from summarize_results import summarize
from training_support import settings,read_config


class WorkflowTests(unittest.TestCase):
    def test_atomic_receipt_detects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x').write_text('before')
            write_receipt(root,'done.json');self.assertTrue(verify_receipt(root,'done.json'))
            (root/'x').write_text('after');self.assertFalse(verify_receipt(root,'done.json'))

    def fake_stage(self,calls,failures,final_marker=True):
        def execute(root,stage,work):
            calls.append(stage)
            target=root/'models' if stage=='finalize' else work
            target.mkdir(parents=True,exist_ok=stage=='finalize')
            if stage=='finalize' and final_marker:atomic_json(target/'final_evaluation_started.json',{'fixture':True})
            if failures.get(stage,0):
                failures[stage]-=1;(target/'partial').write_text('failed fixture')
                raise RuntimeError('injected '+stage+' failure')
            (target/('result_'+stage)).write_text('synthetic fixture')
            write_receipt(target,runner.RECEIPTS[stage])
        return execute

    def test_resume_skips_completed_stages_and_never_repeats_successful_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';data=Path(tmp)/'fake.csv';data.write_text('synthetic workflow fixture')
            calls=[];failures={'develop':1,'optimize_const':1}
            with patch.object(runner,'execute_stage',side_effect=self.fake_stage(calls,failures)):
                with self.assertRaisesRegex(RuntimeError,'develop'):runner.run(data,root,True)
                with self.assertRaisesRegex(RuntimeError,'optimize_const'):runner.run(data,root,True)
                state=runner.run(data,root,True)
                self.assertEqual(state['status'],'complete')
                self.assertEqual(calls.count('develop'),2)
                self.assertEqual(calls.count('finalize'),1)
                before=list(calls);runner.run(data,root,True);self.assertEqual(calls,before)
            self.assertIn('STAGE_SKIP finalize',(root/'run.log').read_text())
            data.write_text('changed fixture')
            with self.assertRaisesRegex(ValueError,'same data'):runner.run(data,root,True)

    def test_failed_or_interrupted_finalize_is_never_retried_even_without_marker(self):
        for marker in [False,True]:
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';data=Path(tmp)/'fake.csv';data.write_text('synthetic fixture')
                calls=[]
                with patch.object(runner,'execute_stage',side_effect=self.fake_stage(calls,{'finalize':1},marker)):
                    with self.assertRaisesRegex(RuntimeError,'injected finalize'):runner.run(data,root,True)
                    with self.assertRaisesRegex(RuntimeError,'retry is forbidden'):runner.run(data,root,True)
                self.assertEqual(calls.count('finalize'),1)

    def test_saved_estimate_is_consumed_without_reestimation_or_mode_drift(self):
        config=read_config(p.ROOT/'configs/full10k.json');config['finalize_pressure_comparisons']=['const']
        _,cfg=settings(config,budget=20,gpr_max_rows=2000)
        original=p.candidates
        def linear_only(quick=False,target=None,physics_feature=False):
            return {'linear':original(quick,target,physics_feature)['linear']}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data=p.ROOT/'data/dataset_ai.csv'
            plan=dict(data_sha256=sha256(data),config=config,quick=False,learning_curves=False,
                      effective_config=cfg,budget_decision={'applied_budget':20,'reason':'frozen synthetic cost fixture'})
            atomic_json(root/'plan.json',plan)
            with patch.object(p,'candidates',side_effect=linear_only),patch.object(p,'estimate_runtime',side_effect=AssertionError('must not reestimate')):
                manifest=p.development(data,root/'models',config=config,execution_plan=root/'plan.json')
            self.assertEqual(manifest['effective_config']['budget'],20)
            self.assertEqual(manifest['effective_config']['gpr_max_rows'],2000)
            self.assertEqual(manifest['budget_decision'],plan['budget_decision'])
            self.assertTrue(verify_receipt(root/'models','development_complete.json'))
            plan['data_sha256']='wrong';atomic_json(root/'bad.json',plan)
            with self.assertRaisesRegex(ValueError,'does not match'):
                p.development(data,root/'bad_models',config=config,execution_plan=root/'bad.json')

    def test_summary_generates_tables_figures_from_saved_predictions_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';models.mkdir()
            rng=np.random.default_rng(2026)
            X=pd.DataFrame({name:rng.uniform(lo,hi,30) for name,(lo,hi) in p.BOUNDS.items()})
            X['log10_k_MLI']=np.log10(X.k_MLI_W_mK);X=X[p.FEATURES]
            manifest=dict(selected={t:'linear' for t in p.TARGETS},models={},data_sha256='synthetic fixture',
                          quick=True,train_rows=24,test_rows=6,cv_folds=3,effective_config={'hyd_model':'const'})
            cv=[];final=[];pred={};curves=[]
            for target in p.TARGETS:
                pred[target+'__actual']=np.arange(1,7,dtype=float)
                for name in ['linear','mlp']:
                    # Serialized regressors are fixtures for parameter inspection,
                    # never representations of a real solver experiment.
                    model=p.candidates()['linear'][0].fit(X,1+X.D_mm)
                    key=target+'__'+name;joblib.dump(model,models/(key+'.joblib'))
                    manifest['models'][key]=dict(tuning_candidates=0,fit_rows=24,parameters={})
                    cv.append(dict(target=target,model=name,r2_mean=.9,r2_std=.01,rmse_mean=1,rmse_std=.1,
                        mape_pct_mean=10,mape_pct_std=1,log10_rmse_mean=.1,log10_rmse_std=.01,
                        fit_seconds_mean=.01,fit_seconds_std=.001,tuning_seconds=.1,predict_10k_seconds_median=.01))
                    pred[key]=np.arange(1,7)*1.01
                    final.append(dict(target=target,model=name,n=6,r2=.99,rmse=.01,mape_pct=1))
                    curves.extend(dict(target=target,model=name,n=n,repeat=r,status='evaluated',rmse=1/n+.01*r)
                                  for n in [10,20] for r in range(5))
                pd.DataFrame(dict(iteration=[1,2,3],training_loss_log_target=[1,.5,.2])).to_csv(models/(target+'__mlp_history.csv'),index=False)
            atomic_json(models/'manifest.json',manifest)
            for name,frame in [('cv_summary.csv',pd.DataFrame(cv)),('final_test_metrics.csv',pd.DataFrame(final)),
                               ('final_test_predictions.csv',pd.DataFrame(pred)),('learning_curve.csv',pd.DataFrame(curves))]:frame.to_csv(models/name,index=False)
            pd.DataFrame([dict(method='const',ratio_mape_pct=1),dict(method='gpr',ratio_mape_pct=.2)]).to_csv(models/'hydraulic_cv_comparison.csv',index=False)
            opts={}
            for mode in ['const','gpr']:
                path=root/mode;path.mkdir();opts[mode]=path
                points=pd.DataFrame([dict(point_id=f'V{i}',scenario=s,role=role,D_mm=12+i,t_ins_mm=2+i*.1)
                    for s in ['S1','S2','S3'] for i,role in enumerate(['A','B','C']+['extra']*6)])
                for name in ['verification_points.csv','verification_inputs.csv','T3_candidates.csv']:points.to_csv(path/name,index=False)
                pd.DataFrame([dict(scenario='S1',hv_ratio=.9)]).to_csv(path/'hv_summary.csv',index=False)
                fig,ax=p.plt.subplots();ax.set(xlabel='Q [W]',ylabel='Pressure [kPa]');fig.savefig(path/'F4.png',dpi=300,facecolor='white');p.plt.close(fig)
            with patch.object(p.PhysicsBaseline,'fit',side_effect=AssertionError('summary must never fit')),patch.object(p.LinearRegression,'fit',side_effect=AssertionError('summary must never fit')):
                summarize(models,root/'report',opts,pilot=root/'missing',pilot_opt=opts['const'],physics_base=root/'missing',physics_feature=root/'missing',quick=True)
            report=root/'report'
            self.assertTrue(verify_receipt(report,'summary_complete.json'))
            self.assertEqual(len(pd.read_csv(report/'T3_candidates.csv')),54)
            self.assertEqual(sha256(report/'F4_gpr.png'),sha256(opts['gpr']/'F4.png'))
            for name in ['F1_parity.png','F2_learning_curves.png','F3_mlp_loss.png','F4.png']:
                with Image.open(report/name) as im:
                    self.assertTrue(all(abs(v-300)<.01 for v in im.info['dpi']))
                    self.assertEqual(im.convert('RGB').getpixel((0,0)),(255,255,255))
            self.assertIn('development fixed-validation',json.loads((report/'report_sources.json').read_text())['notes'][0])


if __name__=='__main__':unittest.main()
