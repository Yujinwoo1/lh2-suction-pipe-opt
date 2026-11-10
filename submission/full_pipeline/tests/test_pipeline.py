import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from unittest.mock import patch
import json
import lh2_pipeline as p
from predict import predict


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = p.load_data(p.ROOT/'data/dataset_ai.csv')

    def test_rigid_units_and_known_constraint_flip(self):
        np.testing.assert_allclose(p.rigid_pressure(self.df), self.df.dp_rigid_Pa, rtol=1e-4)
        row = self.df[self.df.case_id == 110]
        self.assertLess(float(row.dp_design_Pa.iloc[0]), 48300)
        self.assertGreater(float((row.dp_thermal_Pa.to_numpy()+1.8*p.rigid_pressure(row))[0]), 48300)

    def test_input_only_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'inputs.csv'
            self.df[['case_id', *p.BOUNDS]].to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, 'Missing solver-output'):
                p.load_data(file)

    def test_scaler_fits_training_rows_and_targets_roundtrip(self):
        X = self.df[p.FEATURES].iloc[:120]
        y = self.df.Q_leak_W.iloc[:120]
        model = clone(p.candidates()['linear'][0]).fit(X, y)
        np.testing.assert_allclose(model.regressor_.named_steps['scale'].mean_, X.mean())
        np.testing.assert_allclose(model.transformer_.inverse_transform(model.transformer_.transform(y.to_numpy().reshape(-1, 1))).ravel(), y)
        self.assertTrue(np.all(model.predict(self.df[p.FEATURES].iloc[120:]) > 0))

    def test_no_target_or_id_in_features(self):
        self.assertEqual(len(p.FEATURES), 6)
        self.assertFalse(set(p.FEATURES) & {'case_id', 'vaporized', 'dp_design_Pa', *p.TARGETS})

    def test_inference_rejects_extrapolation_before_loading_models(self):
        data = self.df.iloc[:1].copy()
        data['D_mm'] = 100
        with self.assertRaisesRegex(ValueError, 'outside training design domain'):
            predict(data, Path('nonexistent_model_directory'))


    def test_physics_formula_fixed_constants_and_features_only(self):
        # Independent scalar calculation uses raw k, not the implementation helper.
        row = self.df.iloc[0]
        from math import log, pi
        wall = row.D_mm/2000 + .0015
        outer = wall + row.t_ins_mm/1000
        resistance = log(outer/wall)/(2*pi*row.k_MLI_W_mK*row.L_m) + 1/(5*2*pi*outer*row.L_m)
        expected = (300-23.6)/resistance
        X = self.df[p.FEATURES].iloc[:1]
        model = p.PhysicsBaseline().fit(X, [1e99])
        np.testing.assert_allclose(model.predict(X), [expected], rtol=1e-14)
        np.testing.assert_allclose(clone(model).fit(X, [0]).predict(X), [expected])
        self.assertEqual(model.get_params(), {})
        self.assertNotIn('physics_baseline', p.candidates(target='dp_thermal_Pa'))
        self.assertIn('physics_baseline', p.candidates(target='Q_leak_W'))

    def test_physics_audit_matches_full_data_errors(self):
        error = 100*np.abs(p.physics_heat_leak(self.df)/self.df.Q_leak_W.to_numpy()-1)
        self.assertGreaterEqual(error.mean(), .05)
        self.assertLessEqual(error.mean(), .07)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p.audit(self.df, out)
            report = json.loads((out/'data_audit.json').read_text())
        self.assertAlmostEqual(report['physics_baseline_mean_relative_error_pct'], error.mean())
        self.assertAlmostEqual(report['physics_baseline_max_relative_error_pct'], error.max())

    def test_physics_development_uses_same_cv_validation_rows(self):
        original_candidates = p.candidates
        def small_candidates(quick=False, target=None, physics_feature=False):
            return {name: value for name, value in original_candidates(quick, target, physics_feature).items()
                    if name in ['linear', 'physics_baseline']}
        with tempfile.TemporaryDirectory() as tmp, patch.object(p, 'candidates', side_effect=small_candidates):
            out = Path(tmp)
            p.development(p.ROOT/'data/dataset_ai.csv', out, quick=True)
            split = pd.read_csv(out/'split.csv')
            # Reproduce the saved development order, excluding the final holdout.
            train, _ = p.train_test_split(np.arange(len(self.df)), test_size=.2,
                                          random_state=2026, stratify=self.df.vaporized)
            self.assertEqual(set(self.df.iloc[train].case_id), set(split.loc[split.split == 'development', 'case_id']))
            dev = self.df.iloc[train]
            folds = np.load(out/'cv_indices.npz')
            rows = pd.read_csv(out/'cv_fold_metrics.csv')
            baseline = rows[rows.model == 'physics_baseline']
            self.assertEqual(len(baseline), 3)
            self.assertEqual(set(baseline.target), {'Q_leak_W'})
            for row in baseline.itertuples():
                validation = dev.iloc[folds[f'{row.fold}_val']]
                expected = p.metrics(validation.Q_leak_W, p.physics_heat_leak(validation))
                for metric, value in expected.items():
                    self.assertAlmostEqual(getattr(row, metric), value)
            summary = pd.read_csv(out/'cv_summary.csv')
            self.assertEqual(len(summary[summary.model == 'physics_baseline']), 1)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['models']['Q_leak_W__physics_baseline']['tuning_candidates'], 0)
            self.assertEqual(manifest['models']['Q_leak_W__physics_baseline']['tuning_seconds'], 0)
            self.assertFalse((out/'final_evaluation_started.json').exists())
            self.assertEqual(p.PhysicsBaseline.__module__, 'physics_baseline')
            saved = p.joblib.load(out/'Q_leak_W__physics_baseline.joblib')
            np.testing.assert_allclose(saved.predict(dev[p.FEATURES]), p.physics_heat_leak(dev))
            self.assertEqual(manifest['selected']['Q_leak_W'], 'physics_baseline')
            predicted = predict(self.df, out)
            np.testing.assert_allclose(predicted.Q_leak_W_pred, p.physics_heat_leak(self.df))
            from optimize import CachedModels
            optimized = CachedModels(out).evaluate(self.df)
            np.testing.assert_allclose(optimized.Q_leak_pred_W, predicted.Q_leak_W_pred)



if __name__ == '__main__':
    unittest.main()
