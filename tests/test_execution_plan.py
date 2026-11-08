import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import lh2_pipeline as p
from execution_plan import resource_plan,candidate_parameters,search_wall_seconds,walltime_decision,reschedule_estimates
from training_support import settings,read_config
from plan_walltime import proxy_estimates


class ExecutionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.declared,cls.cfg=settings(p.ROOT/'configs/full10k.json')
        cls.snapshot=dict(available_mib=7978,cpus=18,native_blas_threads=18)

    def test_ram_admission_cpu_cap_and_insufficient_memory(self):
        three=resource_plan({**self.cfg,'gpr_max_rows':3000},self.snapshot)
        two=resource_plan({**self.cfg,'gpr_max_rows':2000},self.snapshot)
        self.assertEqual(three['effective_jobs'],2)
        self.assertEqual(two['effective_jobs'],4)
        self.assertLessEqual(three['planned_worker_memory_mib'],three['usable_limit_mib'])
        self.assertLessEqual(two['planned_worker_memory_mib'],two['usable_limit_mib'])
        native=resource_plan({**self.cfg,'blas_threads':18},self.snapshot)
        self.assertEqual(native['ram_max_workers'],2)
        self.assertEqual(native['effective_jobs'],1)
        with self.assertRaises(MemoryError):
            resource_plan(self.cfg,{**self.snapshot,'available_mib':2200})

    def test_gpr_is_always_exact_six_candidates(self):
        models=p.candidates(physics_feature=True)
        for budget in [6,20,50]:
            gpr=candidate_parameters('gpr',models['gpr'][1],budget,True)
            self.assertEqual(len(gpr),6)
            self.assertEqual({r['regressor__model__alpha'] for r in gpr},{1e-10,1e-7})
            self.assertEqual(len({str(r['regressor__model__kernel']) for r in gpr}),3)
            for name in ['polynomial_ridge','random_forest','mlp']:
                self.assertEqual(len(candidate_parameters(name,models[name][1],budget,True)),budget)

    def test_walltime_uses_task_waves_and_row_fallback(self):
        self.assertEqual(search_wall_seconds(2,.5,6,5,2),37.5)
        self.assertEqual(search_wall_seconds(2,.5,6,5,4),20)
        original=pd.DataFrame([dict(target='dp_thermal_Pa',model='random_forest',
            candidates=50,cv_folds=5,one_fold_fit_seconds=2,one_fold_predict_seconds=.5,
            estimated_search_seconds=312.5,estimated_reporting_cv_seconds=7.5,
            estimated_develop_seconds=323,search_workers=2)])
        revised=reschedule_estimates(original,{**self.cfg,'gpr_max_rows':2000},self.snapshot)
        self.assertEqual(revised.search_workers.iloc[0],4)
        self.assertEqual(revised.estimated_develop_seconds.iloc[0],165.5)
        rule=self.declared['budget_rule']
        def fixed(seconds):
            return pd.DataFrame([dict(target='dp_thermal_Pa',model='gpr',candidates=6,
                tunable=True,budget_sensitive=False,estimated_search_seconds=seconds,
                estimated_develop_seconds=seconds)])
        pending=walltime_decision(fixed(20000),rule)
        self.assertTrue(pending['reduced_estimate_required'])
        reduced=walltime_decision(fixed(20000),rule,fixed(10000))
        self.assertEqual((reduced['applied_budget'],reduced['applied_gpr_max_rows']),(50,2000))
        self.assertTrue(reduced['target_met'])
        missed=walltime_decision(fixed(20000),rule,fixed(18000))
        self.assertEqual((missed['applied_budget'],missed['applied_gpr_max_rows']),(6,2000))
        self.assertFalse(missed['target_met'])

    def test_fallback_remeasures_gpr_and_updates_saved_models_and_counts(self):
        config={**self.declared,'finalize_pressure_comparisons':['const']}
        original=p.candidates
        def just_gpr(quick=False,target=None,physics_feature=False):
            model,space=original(quick,target,physics_feature)['gpr']
            model.set_params(regressor__model__optimizer=None)
            return {'gpr':(model,space)}
        def estimate(seconds):
            return pd.DataFrame([dict(target=target,model='gpr',candidates=6,tunable=True,
                budget_sensitive=False,estimated_search_seconds=seconds/2,
                estimated_develop_seconds=seconds/2) for target in p.TARGETS])
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)
            with patch.object(p,'candidates',side_effect=just_gpr),patch.object(p,'estimate_runtime',side_effect=[estimate(20000),estimate(10000)]) as timing:
                manifest=p.development(p.ROOT/'data/dataset_ai.csv',out,config=config)
            self.assertEqual(timing.call_count,2)
            self.assertEqual(manifest['effective_config']['gpr_max_rows'],2000)
            self.assertEqual(manifest['effective_config']['budget'],50)
            self.assertTrue(manifest['budget_decision']['row_reduction_required'])
            self.assertEqual(manifest['execution_revision'],config['execution_revision'])
            self.assertEqual(manifest['cv_folds'],5)
            summary=pd.read_csv(out/'cv_summary.csv')
            self.assertTrue(summary.actual_candidates.eq(6).all())
            for target in p.TARGETS:
                model=joblib.load(out/(target+'__gpr.joblib'))
                self.assertEqual(model.max_rows,2000)
                self.assertEqual(manifest['models'][target+'__gpr']['tuning_candidates'],6)
            for plan in manifest['resource_plans']:
                self.assertLessEqual(plan['planned_worker_memory_mib'],plan['usable_limit_mib'])
            self.assertFalse((out/'final_evaluation_started.json').exists())

    def test_projection_has_all_six_combinations_without_extra_gpr_candidates(self):
        single=pd.DataFrame([dict(fit_seconds=200,predict_100k_seconds=40)]*3)
        native=pd.DataFrame([dict(fit_rows=3000,fit_seconds=130,native_blas_threads=18),
                             dict(fit_rows=2000,fit_seconds=80,native_blas_threads=18)])
        others=pd.DataFrame([dict(model='random_forest',fit_seconds=9,
                                 predict_fold_seconds=.05,predict_100k_seconds=4,tunable=True)])
        cfg={**self.cfg,'finalize_pressure_comparisons':['const','gpr']}
        for n,jobs in [(3000,2),(2000,4)]:
            times=[]
            for budget in [6,20,50]:
                rows=proxy_estimates(single,native,others,cfg,n,budget,self.snapshot)
                main=rows[(rows.model=='gpr') & (rows.target!='hyd_ratio')]
                self.assertTrue(main.candidates.eq(6).all())
                self.assertTrue(main.search_workers.eq(jobs).all())
                times.append(main.estimated_develop_seconds.sum())
            np.testing.assert_allclose(times,np.repeat(times[0],3))


if __name__=='__main__':unittest.main()
