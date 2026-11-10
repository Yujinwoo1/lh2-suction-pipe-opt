"""Check analysis decisions, evaluation separation and reported artifacts."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from analysis_conservative import conservative_table
from analysis_extrapolation import diameter_split
from lh2_pipeline import FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/full10k/analysis'


class AnalysisTests(unittest.TestCase):
    def test_margin_uses_thermal_only_and_inclusive_threshold(self):
        class FakePipeline:
            def predict(self, features, return_std=False):
                self.features = list(features)
                self.return_std = return_std
                # sigma is chosen to give exactly a 1 Pa first-order thermal std.
                return np.zeros(3), np.ones(3)/(np.log(10)*10)
        pipe = FakePipeline()
        points = pd.DataFrame({
            'hyd': ['gpr']*3, 'point_id': ['a', 'b', 'c'], 'scenario': ['S1']*3,
            'role': ['C']*3, 'D_mm': [20]*3, 't_ins_mm': [10]*3,
            'f_Hz': [6]*3, 'L_m': [2]*3, 'mdot_kg_h': [80]*3,
            'k_MLI_W_mK': [.001]*3, 'dp_thermal_pred_Pa': [10]*3,
            'dp_design_pred_Pa': [48297, 48298, 48299], 'dp_design_Pa': [48299]*3,
            'predicted_constraint_satisfied': [True]*3,
            'solver_constraint_satisfied': [True]*3})
        table = conservative_table(points, SimpleNamespace(regressor_=pipe))
        self.assertEqual(pipe.features, FEATURES)
        self.assertTrue(pipe.return_std)
        np.testing.assert_allclose(table.sigma_thermal_Pa, 1)
        self.assertEqual(table.conservative_safe.tolist(), [True, True, False])

    def test_diameter_split_is_disjoint_and_reproducible(self):
        dev = pd.DataFrame({'case_id': np.arange(100), 'D_mm': np.linspace(12, 50, 100)})
        a = diameter_split(dev)
        b = diameter_split(dev)
        self.assertTrue((a[0].D_mm < 40).all())
        self.assertTrue((a[1].D_mm < 40).all())
        self.assertTrue((a[2].D_mm >= 40).all())
        ids = [set(frame.case_id) for frame in a]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(dev.case_id))
        for left, right in zip(a, b):
            pd.testing.assert_frame_equal(left, right)

    @unittest.skipUnless((OUT/'T_extrapolation.csv').exists(), 'Local analysis artifacts not present')
    def test_artifacts_development_only_metrics_and_figures(self):
        split = pd.read_csv(ROOT/'outputs/full10k/models/split.csv')
        dev = set(split.loc[split.split == 'development', 'case_id'])
        assignments = pd.read_csv(OUT/'extrapolation_split.csv')
        self.assertEqual(set(assignments.case_id), dev)
        self.assertTrue(assignments.case_id.is_unique)
        self.assertEqual(assignments.gpr_training_sample.sum(), 3000)
        self.assertTrue(assignments.loc[assignments.gpr_training_sample, 'analysis_split'].eq('train').all())
        pred = pd.read_csv(OUT/'extrapolation_predictions.csv')
        self.assertFalse(set(pred.case_id) & set(assignments.loc[assignments.analysis_split == 'train', 'case_id']))
        table = pd.read_csv(OUT/'T_extrapolation.csv')
        for row in table.itertuples():
            group = pred.loc[(pred.model == row.model) & (pred.scope == row.scope)]
            self.assertEqual(len(group), row.n_evaluation)
            self.assertAlmostEqual(row.mape_pct, 100*np.abs(group.predicted_Pa/group.dp_thermal_Pa-1).mean())
        pdp = pd.read_csv(OUT/'T_pdp.csv')
        self.assertTrue((np.diff(pdp.loc[pdp.feature == 't_ins_mm', 'mean_Q_leak_W']) < 0).all())
        self.assertTrue((np.diff(pdp.loc[pdp.feature == 'log10_k_MLI', 'mean_Q_leak_W']) > 0).all())
        self.assertTrue(set(pd.read_csv(OUT/'pdp_background_ids.csv').case_id) <= dev)
        decisions = pd.read_csv(OUT/'T_conservative.csv')
        self.assertEqual(len(decisions), 54)
        c = decisions.loc[(decisions.hyd == 'gpr') & (decisions.role == 'C')]
        self.assertEqual(len(c), 3)
        self.assertFalse(c.conservative_safe.any())
        for filename in ['F_pdp.png', 'F_extrapolation.png']:
            with Image.open(OUT/filename) as im:
                np.testing.assert_allclose(im.info['dpi'], [300, 300], atol=.01)
                self.assertEqual(im.convert('RGB').getpixel((0, 0)), (255, 255, 255))

    def test_notebook_analysis_only_loads_results(self):
        book = json.loads((ROOT/'submission/LH2_core.ipynb').read_text())
        text = '\n'.join(''.join(c['source']) for c in book['cells'])
        self.assertIn('추가 분석', text)
        for filename in ['T_conservative.csv', 'T_length_scales.csv', 'F_pdp.png', 'T_extrapolation.csv', 'F_extrapolation.png']:
            self.assertIn(filename, text)
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                for forbidden in ['.fit(', '.predict(', 'finalize(', 'analysis_extrapolation.main']:
                    self.assertNotIn(forbidden, ''.join(cell['source']))


if __name__ == '__main__':
    unittest.main()
