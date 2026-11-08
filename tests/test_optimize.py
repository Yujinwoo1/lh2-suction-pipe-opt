import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
import lh2_pipeline as pipeline
import optimize
from predict import predict


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.scenario = {'id': 'S1', 'f_Hz': 6, 'L_m': 2, 'mdot_kg_h': 80, 'k_MLI_W_mK': 8e-5}
        rng = np.random.default_rng(2026)
        self.frame = pd.DataFrame({column: rng.uniform(low, high, 30) for column, (low, high) in pipeline.BOUNDS.items()})
        self.frame['log10_k_MLI'] = np.log10(self.frame.k_MLI_W_mK)
        selected = {}
        for target in pipeline.TARGETS:
            model = clone(pipeline.candidates()['linear'][0]).fit(self.frame[pipeline.FEATURES], 1+self.frame.D_mm)
            joblib.dump(model, self.directory/(target+'__linear.joblib'))
            selected[target] = 'linear'
        (self.directory/'manifest.json').write_text(json.dumps({'selected': selected}))

    def test_cached_vector_matches_predict_and_loads_once(self):
        with patch('optimize.joblib.load', wraps=joblib.load) as loader:
            models = optimize.CachedModels(self.directory)
            values = models.evaluate(self.frame)
            models.evaluate(self.frame.iloc[:2])
            self.assertEqual(loader.call_count, 2)
        expected = predict(self.frame, self.directory)
        np.testing.assert_allclose(values.Q_leak_pred_W, expected.Q_leak_W_pred)
        np.testing.assert_allclose(values.dp_design_pred_Pa, expected.dp_design_Pa_pred)
        self.assertTrue(values.sigma_log10_dp_thermal.isna().all())
        designs = self.frame[['D_mm', 't_ins_mm']].to_numpy()
        objectives, constraints = optimize.objective_values(models, designs, self.scenario)
        expected = predict(optimize.design_frame(designs, self.scenario), self.directory)
        np.testing.assert_allclose(objectives[:, 0], expected.Q_leak_W_pred)
        np.testing.assert_allclose(objectives[:, 1], expected.dp_design_Pa_pred)
        np.testing.assert_allclose(objectives[:, 2], expected.D_o_mm)
        np.testing.assert_allclose(constraints[:, 0], expected.dp_design_Pa_pred-48300)

    def test_gpr_sigma_is_log_space(self):
        model = clone(pipeline.candidates()['gpr'][0]).fit(self.frame[pipeline.FEATURES], 1+self.frame.D_mm)
        models = optimize.CachedModels(self.directory)
        models.models['dp_thermal_Pa'] = model
        expected = model.regressor_.predict(self.frame[pipeline.FEATURES], return_std=True)[1]
        np.testing.assert_allclose(models.evaluate(self.frame).sigma_log10_dp_thermal, expected)

    def test_scenario_outside_domain_rejected(self):
        for column in list(pipeline.BOUNDS)[2:]:
            scenario = dict(self.scenario, **{column: pipeline.BOUNDS[column][1]*2})
            with self.assertRaisesRegex(ValueError, 'outside training design domain'):
                optimize.design_frame([[12, 2]], scenario)

    @unittest.skipUnless(importlib.util.find_spec('pymoo'), 'pymoo dependency is not installed')
    def test_grid_pareto_is_nondominated(self):
        diameters, thicknesses = np.meshgrid(np.linspace(12, 50, 20), np.linspace(2, 25, 20))
        grid = optimize.CachedModels(self.directory).evaluate(optimize.design_frame(
            np.column_stack([diameters.ravel(), thicknesses.ravel()]), self.scenario))
        front = optimize.pareto(grid)
        self.assertGreater(len(front), 0)
        feasible = grid.loc[grid.dp_design_pred_Pa <= 48300, optimize.OBJECTIVES].to_numpy()
        for point in front[optimize.OBJECTIVES].to_numpy():
            self.assertFalse(np.any(np.all(feasible <= point, axis=1) & np.any(feasible < point, axis=1)))

    def test_nonfinite_scenario_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.assertRaisesRegex(ValueError, 'finite'):
                optimize.design_frame([[12, 2]], dict(self.scenario, f_Hz=value))

    def test_candidate_anchors_and_coverage(self):
        front = pd.DataFrame({'D_mm': np.arange(12, 32), 't_ins_mm': 2,
            'Q_leak_pred_W': np.arange(1, 21), 'dp_design_pred_Pa': np.linspace(48000, 30000, 20),
            'D_o_mm': np.arange(35, 15, -1), 'sigma_log10_dp_thermal': np.nan})
        selected = optimize.candidates(front)
        self.assertEqual(selected.role.tolist(), ['A', 'B', 'C']+['extra']*6)
        self.assertEqual(selected.iloc[0].Q_leak_pred_W, 1)
        self.assertEqual(selected.iloc[2].D_o_mm, 16)
        scaled = optimize.normalize(front[optimize.OBJECTIVES].to_numpy(),
                                    front[optimize.OBJECTIVES].min().to_numpy(),
                                    front[optimize.OBJECTIVES].max().to_numpy())
        self.assertEqual(selected.iloc[1].D_mm, front.iloc[np.argmin(np.linalg.norm(scaled, axis=1))].D_mm)
        self.assertEqual(len(selected.drop_duplicates(['D_mm', 't_ins_mm'])), 9)

    def test_compare_synthetic_solver_errors_and_constraint(self):
        points = self.directory/'points.csv'
        solver = self.directory/'solver.csv'
        pd.DataFrame({'point_id': ['V001', 'V002'], 'Q_leak_pred_W': [11, 9],
                      'dp_design_pred_Pa': [48000, 49000]}).to_csv(points, index=False)
        pd.DataFrame({'point_id': ['V002', 'V001'], 'Q_leak_W': [10, 10],
                      'dp_design_Pa': [48000, 50000]}).to_csv(solver, index=False)
        table = optimize.compare(points, solver, self.directory/'T4.csv')
        np.testing.assert_allclose(table.Q_leak_W_error_pct, [10, -10])
        self.assertFalse(table.constraint_agreement.any())
        with self.assertRaises(FileExistsError):
            optimize.compare(points, solver, self.directory/'T4.csv')

    def test_compare_rejects_unmatched_or_duplicate_ids(self):
        points = self.directory/'points.csv'
        solver = self.directory/'solver.csv'
        pd.DataFrame({'point_id': ['V001'], 'Q_leak_pred_W': [11],
                      'dp_design_pred_Pa': [48000]}).to_csv(points, index=False)
        for identifiers, message in [(['V002'], 'must match'), (['V001', 'V001'], 'unique')]:
            pd.DataFrame({'point_id': identifiers, 'Q_leak_W': 10,
                          'dp_design_Pa': 48000}).to_csv(solver, index=False)
            with self.assertRaisesRegex(ValueError, message):
                optimize.compare(points, solver, self.directory/'T4.csv')
            self.assertFalse((self.directory/'T4.csv').exists())


if __name__ == '__main__':
    unittest.main()
