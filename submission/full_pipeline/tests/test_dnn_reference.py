"""DNN 비교에서 분할 누출과 기존 결과 변경을 막는 조건을 확인한다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import ann_reference as ann
from lh2_pipeline import FEATURES, TARGETS


class DNNComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = Path(__file__).resolve().parents[2]
        cls.reference = cls.submission/'results/reproduction'
        cls.fold_path = cls.submission/'results/dnn_audit/reference_cv_indices.npz'

    def test_development_loader_never_parses_holdout_targets(self):
        data_path = self.submission/'data/dataset_ai_10k.csv'
        original_read = pd.read_csv
        metadata = original_read(data_path, usecols=['case_id'])
        split = original_read(self.reference/'split.csv')
        holdout = set(split.loc[split.split == 'test', 'case_id'])
        holdout_rows = [index+1 for index, case_id in enumerate(metadata.case_id) if case_id in holdout]
        target_reads = []

        def guarded_read(path, *args, **kwargs):
            if Path(path) == data_path and set(kwargs.get('usecols', [])) & set(TARGETS):
                skip = kwargs.get('skiprows')
                self.assertTrue(callable(skip))
                self.assertTrue(all(skip(row) for row in holdout_rows))
                target_reads.append(True)
            return original_read(path, *args, **kwargs)

        with patch.object(pd, 'read_csv', side_effect=guarded_read):
            data, folds = ann.development_only_data(data_path, self.reference, self.fold_path)
        self.assertEqual(len(target_reads), 1)
        self.assertEqual(len(data), 8000)
        self.assertFalse(set(data.case_id) & holdout)
        validation = np.concatenate([b for _, b in folds])
        np.testing.assert_array_equal(np.sort(validation), np.arange(8000))
        with np.load(self.fold_path) as saved:
            for i, (a, b) in enumerate(folds):
                np.testing.assert_array_equal(a, saved[f'{i}_train'])
                np.testing.assert_array_equal(b, saved[f'{i}_val'])

    def test_deep_candidates_keep_original_mlp_preprocessing(self):
        candidates = ann.dnn_candidates()
        self.assertEqual(len(candidates), 50)
        self.assertEqual(len({repr(sorted(item.items())) for item in candidates}), 50)
        self.assertEqual(candidates, ann.dnn_candidates())
        for parameters in candidates:
            estimator = ann.dnn_estimator(parameters)
            self.assertEqual(list(estimator.regressor.named_steps), ['physics_feature', 'scale', 'model'])
            model = estimator.regressor.named_steps['model']
            self.assertIn(len(model.hidden_layer_sizes), [3, 4])
            self.assertTrue(all(64 <= width <= 256 for width in model.hidden_layer_sizes))
            self.assertEqual((model.activation, model.solver), ('relu', 'adam'))
            self.assertEqual((model.max_iter, model.random_state), (3000, 2026))
            self.assertTrue(model.early_stopping)

    def test_scaler_ignores_extreme_validation_rows(self):
        data, _ = ann.development_only_data(self.submission/'data/dataset_ai_10k.csv',
            self.reference, self.fold_path)
        train = data.iloc[:64][FEATURES]
        validation = data.iloc[64:68][FEATURES].copy()
        validation['D_mm'] = 1e8
        estimator = ann.dnn_estimator(ann.dnn_candidates()[0])
        estimator.set_params(regressor__model__hidden_layer_sizes=(64, 64, 64),
            regressor__model__max_iter=2)
        with threadpool_limits(limits=1):
            with self.assertWarns(Warning):
                estimator.fit(train, data.iloc[:64].Q_leak_W.to_numpy())
        pipeline = estimator.regressor_
        features = pipeline.named_steps['physics_feature'].transform(train)
        np.testing.assert_allclose(pipeline.named_steps['scale'].mean_, features.mean().to_numpy())
        before = pipeline.named_steps['scale'].mean_.copy()
        estimator.predict(validation)
        np.testing.assert_array_equal(before, pipeline.named_steps['scale'].mean_)

    def test_runtime_budget_reduces_both_targets_without_score_input(self):
        self.assertEqual(ann.dnn_budget(30), 50)
        self.assertEqual(ann.dnn_budget(120), 36)
        self.assertEqual(ann.dnn_budget(6000), 1)
        with self.assertRaises(ValueError):
            ann.dnn_budget(0)

    def test_summary_uses_log_rmse_and_sample_standard_deviation(self):
        rows = []
        for candidate in range(2):
            for fold in range(5):
                row = dict(candidate=candidate, selection_log10_rmse=2-candidate,
                    r2=.9, rmse=10+candidate+fold, mape_pct=1+candidate,
                    log10_rmse=2-candidate, log10_r2=.9, fit_seconds=1)
                rows.append(row)
        summary = ann.dnn_summary(rows, ann.dnn_candidates(), 'Q_leak_W', 2, 10)
        self.assertEqual(summary['selected_candidate'], 1)
        self.assertEqual(summary['rmse_mean'], 13)
        self.assertAlmostEqual(summary['rmse_std'], np.std(np.arange(11, 16), ddof=1))
        self.assertFalse(summary['final_holdout_used'])

    def test_append_preserves_original_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/'original.csv'
            output = Path(directory)/'combined.csv'
            source = b'target,model,rmse_mean\nQ_leak_W,mlp,0.626521900000001\n'
            original.write_bytes(source)
            ann.append_dnn_table(original, output,
                [dict(target='Q_leak_W', model='dnn', rmse_mean=.5)])
            self.assertEqual(original.read_bytes(), source)
            self.assertTrue(output.read_bytes().startswith(source))
            self.assertEqual(pd.read_csv(output).model.tolist(), ['mlp', 'dnn'])
            with self.assertRaises(FileExistsError):
                ann.append_dnn_table(original, output, [])


if __name__ == '__main__':
    unittest.main()
