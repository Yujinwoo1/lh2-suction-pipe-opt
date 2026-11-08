import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import lh2_pipeline as p
from training_support import settings, sampled_parameters, sample_indices, read_config


class FullDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = p.load_data(p.ROOT/'data/dataset_ai.csv')
        cls.config = read_config(p.ROOT/'configs/full10k.json')

    def test_preregistered_config_overrides_and_finalization_guard(self):
        declared, cfg = settings(self.config, quick=True, gpr_max_rows=100, budget=4, n_jobs=2)
        self.assertEqual(declared, self.config)
        self.assertEqual((cfg['budget'], cfg['cv_repeats'], cfg['cv_splits'], cfg['gpr_max_rows']), (4, 1, 3, 100))
        self.assertTrue(cfg['physics_feature'])
        p.validate_finalize_config({'config': declared}, self.config)
        with self.assertRaisesRegex(ValueError, 'differs'):
            p.validate_finalize_config({'config': declared}, dict(self.config, budget=50))
        for changed in [dict(seed=2027), dict(budget=0), dict(cv_repeats=0), dict(n_jobs=0),
                        dict(gpr_max_rows=1), dict(unknown_option=1)]:
            with self.assertRaises(ValueError):
                settings({**self.config, **changed})

    def test_equal_large_budget_and_deterministic_candidates(self):
        for name, (_, space) in p.candidates().items():
            if not space:
                continue
            values = sampled_parameters(space, 50)
            self.assertEqual(len(values), 50, name)
            self.assertEqual(len({repr(sorted(v.items())) for v in values}), 50, name)
            self.assertEqual(repr(values), repr(sampled_parameters(space, 50)))

    def test_status_filter_precedes_output_validation_and_mixed_ids(self):
        data = self.df.copy()
        data.loc[data.index >= 100, 'case_id'] += 100000
        data['status'] = 'converged'
        data.loc[[0, 100, 101], 'status'] = 'failed'
        excluded = data.loc[[0, 100, 101], 'case_id'].tolist()
        data.loc[[0, 100, 101], p.TARGETS] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'status.csv'
            data.to_csv(path, index=False)
            loaded = p.load_data(path)
            self.assertEqual(len(loaded), 197)
            self.assertFalse(set(excluded) & set(loaded.case_id))
            self.assertTrue((loaded.case_id < 201).any() and (loaded.case_id > 100000).any())
            p.audit(loaded, Path(tmp))
            report = json.loads((Path(tmp)/'data_audit.json').read_text())
            self.assertEqual(report['source_rows'], 200)
            self.assertEqual(report['excluded_status_count'], 3)
            self.assertEqual(report['excluded_status_case_ids'], excluded)

    def test_large_changed_vaporization_rate_warns_without_rejecting(self):
        row = self.df[(self.df.vaporized == 0) & (self.df.t_ins_mm < 24)].iloc[:1]
        data = pd.concat([row]*8000, ignore_index=True)
        data['case_id'] = np.arange(100001, 108001)
        data['t_ins_mm'] += np.linspace(0, .0001, len(data))
        data['D_o_mm'] = data.D_mm+3+2*data.t_ins_mm
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'large_fake.csv'
            data.to_csv(path, index=False)
            with self.assertWarnsRegex(UserWarning, 'differs from pilot 6%'):
                loaded = p.load_data(path)
            self.assertEqual(len(loaded), 8000)

    @staticmethod
    def small_models(quick=False, target=None, physics_feature=False):
        models = p.candidates_original(quick, target, physics_feature)
        # Actual capped GPR and real preprocessing; kernel optimization is disabled
        # only in this fast unit test. The CLI smoke retains normal optimization.
        gpr, _ = models['gpr']
        from sklearn.gaussian_process.kernels import RBF
        gpr.set_params(regressor__model__kernel=RBF(np.ones(7 if physics_feature else 6)),
                       regressor__model__optimizer=None)
        return {'linear': models['linear'], 'gpr': (gpr, {'regressor__model__alpha': [1e-7, 1e-6]})}

    def test_gpr_cap_100_applies_to_refit_cv_and_curve_without_leakage(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'run'
            with patch.object(p, 'candidates_original', p.candidates, create=True), patch.object(p, 'candidates', side_effect=self.small_models):
                manifest = p.development(p.ROOT/'data/dataset_ai.csv', out, quick=True,
                    learning=True, config=self.config, gpr_max_rows=100, n_jobs=2)
            self.assertEqual(manifest['config'], self.config)
            self.assertEqual(manifest['effective_config']['n_jobs'], 2)
            records = pd.read_csv(out/'gpr_subsamples.csv')
            split = pd.read_csv(out/'split.csv')
            dev_ids = set(split.loc[split.split=='development', 'case_id'])
            test_ids = set(split.loc[split.split=='test', 'case_id'])
            self.assertTrue(set(records.case_id) <= dev_ids)
            self.assertFalse(set(records.case_id) & test_ids)
            train, _ = p.train_test_split(np.arange(len(self.df)), test_size=.2,
                                          random_state=2026, stratify=self.df.vaporized)
            folds = np.load(out/'cv_indices.npz')
            for target in p.TARGETS:
                refit_ids = records[(records.target==target) & (records.stage=='refit')].case_id.tolist()
                self.assertEqual(refit_ids, self.df.iloc[train[sample_indices(len(train), 100)]].case_id.tolist())
                model = joblib.load(out/(target+'__gpr.joblib'))
                self.assertEqual(len(model.regressor_.named_steps['model'].X_train_), 100)
                sampled_X = self.df.set_index('case_id').loc[refit_ids, p.FEATURES]
                derived = model.regressor_.named_steps['physics_feature'].transform(sampled_X)
                np.testing.assert_allclose(model.regressor_.named_steps['scale'].mean_, derived.mean())
                for fold in range(3):
                    ids = records[(records.target==target) & (records.fold==fold)].case_id.tolist()
                    a, b = folds[f'{fold}_train'], folds[f'{fold}_val']
                    self.assertEqual(len(ids), 100)
                    self.assertTrue(set(ids) <= set(self.df.iloc[train[a]].case_id))
                    self.assertFalse(set(ids) & set(self.df.iloc[train[b]].case_id))
            metrics = pd.read_csv(out/'cv_fold_metrics.csv')
            self.assertTrue(metrics.loc[metrics.model=='gpr', 'fit_rows'].eq(100).all())
            self.assertTrue(metrics.loc[metrics.model=='linear', 'fit_rows'].gt(100).all())
            curve_split = pd.read_csv(out/'learning_curve_split.csv')
            samples = pd.read_csv(out/'learning_curve_samples.csv')
            curve_train = set(curve_split.loc[curve_split.split=='train_pool', 'case_id'])
            curve_val = set(curve_split.loc[curve_split.split=='validation', 'case_id'])
            self.assertTrue(set(samples.case_id) <= curve_train)
            self.assertFalse((curve_train | curve_val) & test_ids)
            curve = pd.read_csv(out/'learning_curve.csv')
            evaluated = curve[curve.status=='evaluated']
            self.assertTrue(evaluated.groupby(['target', 'model', 'n']).size().eq(5).all())
            self.assertTrue((evaluated[evaluated.model=='gpr'].n <= 100).all())
            self.assertIn('skipped_gpr_max_rows', set(curve.status))
            self.assertIn('skipped_validation_holdout', set(curve.status))
            self.assertFalse((out/'final_evaluation_started.json').exists())

    def test_estimate_is_readonly_one_fold_and_one_candidate(self):
        models = p.candidates
        def linear_only(quick=False, target=None, physics_feature=False):
            return {'linear': models(quick, target, physics_feature)['linear']}
        original_fit = p.LinearRegression.fit
        observed_rows = []
        def track_fit(model, X, y, **kwargs):
            observed_rows.append(len(X))
            return original_fit(model, X, y, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            out = Path(tmp)/'must_not_exist'
            with patch.object(p, 'candidates', side_effect=linear_only), patch.object(p.LinearRegression, 'fit', track_fit):
                result = p.development(p.ROOT/'data/dataset_ai.csv', out, config=self.config,
                                       estimate_time=True)
            self.assertFalse(out.exists())
            self.assertEqual(len(observed_rows), 2)
            self.assertTrue(all(n == 128 for n in observed_rows))
            self.assertEqual(set(result.cv_folds), {15})
            self.assertIn('Estimated total develop', output.getvalue())


if __name__ == '__main__':
    unittest.main()
