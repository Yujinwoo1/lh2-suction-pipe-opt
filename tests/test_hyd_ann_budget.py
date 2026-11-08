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
from sklearn.base import clone

import lh2_pipeline as p
from ann_reference import load_ann_reference,UNREPORTED
from hydraulics import ratio_estimator,RATIO_FILE,pressure_metrics,hydraulic_cv
from optimize import CachedModels,objective_values,design_frame
from predict import predict
from training_support import budget_decision,read_config
from benchmark_gpr import benchmark_inputs,workload_table


class AdditionalTasksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=p.load_data(p.ROOT/'data/dataset_ai.csv')

    def test_budget_rule_boundaries_and_fixed_model_cost(self):
        rule=read_config(p.ROOT/'configs/full10k.json')['budget_rule']
        for seconds,expected in [(14400,50),(14400.01,20),(28800,20),(28800.01,6)]:
            search=seconds*6/50
            estimates=pd.DataFrame([dict(model='gpr',candidates=6,tunable=True,
                estimated_search_seconds=search,estimated_develop_seconds=search)])
            result=budget_decision(estimates,rule)
            self.assertEqual(result['applied_budget'],expected)
            self.assertAlmostEqual(result['reference_estimated_seconds'],seconds)
        fixed=pd.DataFrame([dict(model='ann_ref',candidates=1,tunable=False,
            estimated_search_seconds=100,estimated_develop_seconds=200)])
        self.assertEqual(budget_decision(fixed,rule)['reference_estimated_seconds'],200)

    def test_auto_budget_decision_is_applied_and_saved(self):
        config=read_config(p.ROOT/'configs/full10k.json')
        config['finalize_pressure_comparisons']=['const']
        models=p.candidates
        def linear_only(quick=False,target=None,physics_feature=False):
            return {'linear':models(quick,target,physics_feature)['linear']}
        estimate=pd.DataFrame([dict(model='linear',candidates=1,tunable=False,
            estimated_search_seconds=1,estimated_develop_seconds=2)])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(p,'candidates',side_effect=linear_only),patch.object(p,'estimate_runtime',return_value=estimate):
                manifest=p.development(p.ROOT/'data/dataset_ai.csv',Path(tmp),config=config)
            self.assertEqual(manifest['effective_config']['budget'],50)
            self.assertEqual(manifest['budget_decision']['applied_budget'],50)
            self.assertIn('Actual --data',manifest['budget_decision']['reason'])
            self.assertEqual(manifest['config']['budget'],6)

    def test_hydraulic_prediction_cache_objectives_and_constant_override(self):
        data=self.df.iloc[:30]; X=data[p.FEATURES]
        thermal=clone(p.candidates()['gpr'][0]).set_params(regressor__model__optimizer=None).fit(X,data.dp_thermal_Pa)
        ratio=ratio_estimator(20).set_params(regressor__model__optimizer=None).fit(X,np.full(len(X),1.9))
        heat=p.PhysicsBaseline()
        manifest={'selected':{'Q_leak_W':'physics_baseline','dp_thermal_Pa':'gpr'},
                  'effective_config':{'hyd_model':'gpr'},'hydraulic':{'ratio_model_file':RATIO_FILE}}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            joblib.dump(thermal,out/'dp_thermal_Pa__gpr.joblib')
            joblib.dump(heat,out/'Q_leak_W__physics_baseline.joblib')
            joblib.dump(ratio,out/RATIO_FILE)
            (out/'manifest.json').write_text(json.dumps(manifest))
            with patch('optimize.joblib.load',wraps=joblib.load) as loader:
                cached=CachedModels(out)
                actual=cached.evaluate(data)
                cached.evaluate(data.iloc[:2])
                self.assertEqual(loader.call_count,3)
            expected=predict(data,out)
            np.testing.assert_allclose(actual.dp_design_pred_Pa,expected.dp_design_Pa_pred)
            np.testing.assert_allclose(expected.hydraulic_ratio_pred,1.9)
            constant=predict(data,out,'const')
            np.testing.assert_allclose(constant.hydraulic_ratio_pred,1.8)
            np.testing.assert_allclose(expected.dp_design_Pa_pred-constant.dp_design_Pa_pred,.1*p.rigid_pressure(data))
            scenario=dict(f_Hz=6,L_m=2,mdot_kg_h=80,k_MLI_W_mK=8e-5)
            design=[[20,3],[40,15]]
            f,g=objective_values(cached,design,scenario)
            direct=predict(design_frame(design,scenario),out)
            np.testing.assert_allclose(f[:,1],direct.dp_design_Pa_pred)
            np.testing.assert_allclose(g[:,0],direct.dp_design_Pa_pred-48300)
            (out/RATIO_FILE).unlink()
            with self.assertRaisesRegex(ValueError,'not trained'):
                CachedModels(out,'gpr')

    def test_hydraulic_cv_target_is_the_supplied_solver_ratio(self):
        dev=self.df.iloc[:30].reset_index(drop=True)
        folds=list(p.RepeatedKFold(n_splits=3,n_repeats=1,random_state=2026).split(dev))
        thermal=[dev.dp_thermal_Pa.to_numpy()[b] for _,b in folds]
        original=ratio_estimator
        def fixed(cap):
            return original(cap).set_params(regressor__model__optimizer=None)
        with tempfile.TemporaryDirectory() as tmp,patch('hydraulics.ratio_estimator',side_effect=fixed):
            out=Path(tmp)
            hydraulic_cv(dev,p.FEATURES,folds,None,p.rigid_pressure(dev),out,20,thermal_oof=thermal)
            result=pd.read_csv(out/'hydraulic_cv_predictions.csv')
            truth=dev.set_index('case_id')
            expected=truth.loc[result.case_id,'dp_hyd_Pa'].to_numpy()/truth.loc[result.case_id,'dp_rigid_Pa'].to_numpy()
            np.testing.assert_allclose(result.ratio_actual,expected,rtol=1e-14)

    def test_pressure_comparison_metrics_count_false_safe_and_boundary(self):
        actual=[48000,50000,60000,30000]
        prediction=[49000,47000,47000,31000]
        report=pressure_metrics(actual,prediction)
        self.assertEqual(report['boundary_n'],2)
        self.assertEqual(report['boundary_mismatch_count'],2)
        self.assertEqual(report['false_safe_count'],2)
        self.assertEqual(report['false_unsafe_count'],1)
        self.assertEqual(report['boundary_false_safe_count'],1)
        self.assertEqual(report['boundary_false_unsafe_count'],1)

    def test_ann_blank_skips_and_ready_spec_is_fixed_and_train_capped(self):
        spec,model,note=load_ann_reference(p.ROOT/'configs/ann_ref.json')
        self.assertIsNone(model)
        self.assertTrue(all(value is None for value in spec.values()))
        self.assertIn('skipped',note)
        ready=dict(hidden_layer_sizes=[4],activation='tanh',solver='lbfgs',
                   learning_rate='constant',max_iter=10,n_train=20,source='unit-test fixture, not paper')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ann.json'; path.write_text(json.dumps(ready))
            spec,model,note=load_ann_reference(path)
            model.fit(self.df[p.FEATURES].iloc[:30],self.df.Q_leak_W.iloc[:30])
            self.assertEqual(len(model.sample_indices_),20)
            self.assertEqual(model.regressor_.named_steps['model'].hidden_layer_sizes,(4,))
            self.assertEqual(model.regressor_.named_steps['scale'].n_features_in_,6)
            ready['learning_rate']=UNREPORTED;path.write_text(json.dumps(ready))
            _,model,_=load_ann_reference(path)
            self.assertEqual(model.regressor.named_steps['model'].learning_rate,'constant')

    def test_ann_is_in_comparison_without_tuning_when_config_is_complete(self):
        ready=dict(hidden_layer_sizes=[2],activation='tanh',solver='lbfgs',
                   learning_rate='constant',max_iter=2,n_train=20,source='unit-test fixture, not paper')
        models=p.candidates
        def linear_only(quick=False,target=None,physics_feature=False):
            return {'linear':models(quick,target,physics_feature)['linear']}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            spec=Path(tmp)/'ann.json';spec.write_text(json.dumps(ready))
            out=Path(tmp)/'run'
            with patch.object(p,'candidates',side_effect=linear_only):
                manifest=p.development(p.ROOT/'data/dataset_ai.csv',out,quick=True,ann_ref_config=spec)
            summary=pd.read_csv(out/'cv_summary.csv')
            self.assertEqual(len(summary[summary.model=='ann_ref']),2)
            for target in p.TARGETS:
                entry=manifest['models'][target+'__ann_ref']
                self.assertEqual(entry['tuning_candidates'],0)
                self.assertEqual(entry['tuning_seconds'],0)
                self.assertEqual(entry['fit_rows'],20)

    def test_benchmark_uses_inputs_and_does_not_modify_source_or_save_model(self):
        data=pd.concat([self.df]*15,ignore_index=True)
        data['Q_leak_W']=-99999
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'cases.csv';data.to_csv(source,index=False)
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            X,y,ids=benchmark_inputs(source)
            self.assertEqual(list(X),p.FEATURES)
            self.assertEqual(len(ids),3000)
            self.assertTrue((y>0).all())
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(list(Path(tmp).iterdir()),[source])
            np.testing.assert_array_equal(ids,benchmark_inputs(source)[2])
        measured=[dict(fit_seconds=1,predict_100k_seconds=1)]*3
        table=workload_table(measured,[],10000,{'test_size':.2,'cv_splits':5})
        self.assertEqual(table.budget.tolist(),[6,20,50])
        self.assertTrue(table.gpr_hours_mean.is_monotonic_increasing)
        physics=[dict(model='physics_baseline',tunable=False,fit_seconds=1,
                      predict_fold_seconds=0,predict_100k_seconds=0)]
        table=workload_table(measured,physics,10000,{'test_size':.2,'cv_splits':5})
        np.testing.assert_allclose(table.other_models_hours*3600,31.25)



if __name__=='__main__': unittest.main()
