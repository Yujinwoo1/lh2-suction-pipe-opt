"""제출본의 복사 무결성과 저장된 노트북 출력을 확인한다."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


class SubmissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(__file__).resolve().parents[2]
        cls.ref = cls.base / 'results' / 'reproduction'
        cls.book = json.loads((cls.base / 'LH2_core.ipynb').read_text())

    def test_copied_source_hashes(self):
        hashes = json.loads((self.ref / 'source_hashes.json').read_text())
        for name, expected in hashes.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((self.base / name).read_bytes()).hexdigest(), expected)

    def test_data_matches_original_manifest(self):
        manifest = json.loads((self.ref / 'manifest.json').read_text())
        data = self.base / 'data' / 'dataset_ai_10k.csv'
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), manifest['data_sha256'])
        self.assertEqual(manifest['seed'], 2026)
        self.assertEqual(manifest['train_rows'], 8000)
        self.assertEqual(manifest['test_rows'], 2000)

    def test_notebook_has_all_outputs_and_no_errors(self):
        cells = [c for c in self.book['cells'] if c['cell_type'] == 'code']
        self.assertEqual([c['execution_count'] for c in cells], list(range(1, len(cells)+1)))
        for cell in cells:
            self.assertTrue(cell['outputs'])
            self.assertFalse(any(o['output_type'] == 'error' for o in cell['outputs']))
        text = json.dumps(self.book, ensure_ascii=False)
        self.assertNotIn(('/' + 'home' + '/'), text)
        self.assertNotIn('joblib.load', text)
        self.assertNotIn('p.finalize(', text)

    def test_fixed_models_and_saved_subsamples(self):
        fixed = json.loads((self.ref / 'fixed_parameters.json').read_text())
        for key, params in fixed.items():
            self.assertEqual(params['random_state'], 2026)
            if key.endswith('__gpr'):
                self.assertIsNone(params['optimizer'])
                self.assertIn('class', params['kernel'])
        split = pd.read_csv(self.ref / 'split.csv')
        dev = set(split.loc[split.split == 'development', 'case_id'])
        test = set(split.loc[split.split == 'test', 'case_id'])
        self.assertFalse(dev & test)
        samples = pd.read_csv(self.ref / 'gpr_subsamples.csv')
        for _, group in samples.loc[samples.stage == 'refit'].groupby('target'):
            self.assertEqual(len(group), 3000)
            self.assertTrue(group.case_id.is_unique)
            self.assertTrue(set(group.case_id) <= dev)
        hyd = pd.read_csv(self.ref / 'hydraulic_subsamples.csv')
        ids = hyd.loc[hyd.stage == 'refit', 'case_id']
        self.assertEqual(len(ids), 3000)
        self.assertTrue(set(ids) <= dev)

    def test_comparisons_report_actual_differences(self):
        table = pd.read_csv(self.ref / 'notebook_test_comparison.csv')
        original = pd.read_csv(self.base / 'results' / 'T_final_test_metrics.csv')
        self.assertEqual(len(table), 4)
        for metric in ['r2', 'rmse', 'mape_pct']:
            np.testing.assert_allclose(table[metric+'_difference'],
                                       table[metric+'_notebook']-table[metric+'_original'],
                                       rtol=1e-5, atol=1e-12)
            joined = table.merge(original, on=['target', 'model'])
            np.testing.assert_array_equal(joined[metric+'_original'], joined[metric])
        summary = json.loads((self.ref / 'notebook_execution.json').read_text())
        self.assertGreater(summary['elapsed_seconds'], 0)
        self.assertFalse(summary['finalize_rerun'])

    def test_bundle_excludes_runtime_and_personal_paths(self):
        forbidden = {'.venv', '__pycache__', '.mplconfig', 'outputs', '.git', '.aws', '.codex', '.agents'}
        for path in self.base.rglob('*'):
            self.assertFalse(set(path.relative_to(self.base).parts) & forbidden)
            self.assertNotEqual(path.suffix, '.joblib')
            if path.is_file() and path.suffix in {'.py', '.md', '.json', '.sh', '.ipynb'}:
                self.assertNotIn(('/' + 'home' + '/'), path.read_text(), str(path.relative_to(self.base)))


if __name__ == '__main__':
    unittest.main()
