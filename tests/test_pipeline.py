import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
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


if __name__ == '__main__':
    unittest.main()
