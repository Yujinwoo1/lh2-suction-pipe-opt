"""제출 재검증 표와 읽기 전용 노트북의 수치·보존 계약."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image


class ReverificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(__file__).resolve().parents[1] / 'submission'
        cls.results = cls.base / 'results'
        cls.table = pd.read_csv(cls.results / 'T4_reverification.csv')

    def test_all_rows_ids_errors_and_constraint_decisions(self):
        t = self.table
        self.assertEqual(len(t), 54)
        self.assertTrue(t.status.eq('converged').all())
        self.assertFalse(t.duplicated(['hyd', 'point_id']).any())
        for method, start in [('const', 900001), ('gpr', 910001)]:
            g = t.loc[t.hyd.eq(method)]
            self.assertEqual(set(g.case_id), set(range(start, start + 27)))
            self.assertEqual(set(g.point_id), {f'V{i:03}' for i in range(1, 28)})
        for target, pred in [('Q_leak_W', 'Q_leak_pred_W'),
                             ('dp_thermal_Pa', 'dp_thermal_pred_Pa'),
                             ('dp_design_Pa', 'dp_design_pred_Pa')]:
            expected = 100 * (t[pred] / t[target] - 1)
            np.testing.assert_allclose(t[target + '_error_pct'], expected, atol=1e-12)
            np.testing.assert_allclose(t[target + '_abs_error_pct'], abs(expected), atol=1e-12)
            self.assertLess(abs(expected).max(), 5)
        np.testing.assert_array_equal(t.predicted_constraint_satisfied, t.dp_design_pred_Pa <= 48300)
        np.testing.assert_array_equal(t.solver_constraint_satisfied, t.dp_design_Pa <= 48300)
        np.testing.assert_array_equal(t.constraint_agreement,
                                      t.predicted_constraint_satisfied == t.solver_constraint_satisfied)
        summary = pd.read_csv(self.results / 'T4_summary.csv').set_index('hyd')
        self.assertEqual(summary.loc['const', 'agreement'], 27)
        self.assertEqual(summary.loc['gpr', 'agreement'], 24)
        for method, g in t.groupby('hyd'):
            for target in ['Q_leak_W', 'dp_thermal_Pa', 'dp_design_Pa']:
                self.assertAlmostEqual(summary.loc[method, target + '_mean_abs_error_pct'],
                                       g[target + '_error_pct'].abs().mean())
                self.assertAlmostEqual(summary.loc[method, target + '_max_abs_error_pct'],
                                       g[target + '_error_pct'].abs().max())
        const = t.loc[t.hyd.eq('const')]
        self.assertTrue((const.dp_hyd_pred_Pa > const.dp_hyd_Pa).all())
        self.assertGreater(const.dp_design_Pa_error_pct.mean(), 0)

    def test_mismatches_and_independent_crosscheck(self):
        m = self.table.loc[~self.table.constraint_agreement]
        self.assertEqual(m.point_id.tolist(), ['V003', 'V012', 'V021'])
        self.assertTrue(m.hyd.eq('gpr').all() and m.role.eq('C').all())
        np.testing.assert_allclose(m.solver_excess_Pa, [6.1, 18.3, 1.0])
        np.testing.assert_allclose(m.solver_excess_Pa, m.dp_design_Pa - 48300)
        self.assertTrue(m.predicted_constraint_satisfied.all())
        check = pd.read_csv(self.results / 'T4_crosscheck.csv')
        self.assertEqual(len(check), 17)
        self.assertTrue(check.matches.all())
        self.assertEqual(check.max_abs_difference.max(), 0)
        # Compare again to the independent input when it is available locally.
        independent = self.base.parent / 'outputs/full10k/reverification_check/T4_reverification.csv'
        if independent.exists():
            old = pd.read_csv(independent).set_index(['hyd', 'point_id']).sort_index()
            new = self.table.set_index(['hyd', 'point_id']).sort_index()
            for row in check.itertuples():
                np.testing.assert_array_equal(new[row.generated_column], old[row.column])

    def test_figure_and_read_only_notebook_with_embedded_output(self):
        with Image.open(self.results / 'F_T4_reverification.png') as im:
            np.testing.assert_allclose(im.info['dpi'], [300, 300], atol=.01)
            self.assertEqual(im.convert('RGB').getpixel((0, 0)), (255, 255, 255))
        book = json.loads((self.base / 'LH2_core.ipynb').read_text())
        code = [c for c in book['cells'] if c['cell_type'] == 'code']
        source = '\n'.join(''.join(c['source']) for c in code)
        for forbidden in ['.fit(', '.predict(', 'minimize(', 'p.finalize(', 'p.load_data(', '.to_csv(']:
            self.assertNotIn(forbidden, source)
        self.assertEqual([c['execution_count'] for c in code], list(range(1, len(code) + 1)))
        for cell in code:
            self.assertTrue(cell['outputs'])
            self.assertFalse(any(o['output_type'] == 'error' for o in cell['outputs']))
        self.assertIn('T4_reverification.csv', ''.join(code[-1]['source']))
        self.assertIn('Image(', ''.join(code[-1]['source']))
        self.assertTrue(any('image/png' in o.get('data', {}) for o in code[-1]['outputs']))


if __name__ == '__main__':
    unittest.main()
