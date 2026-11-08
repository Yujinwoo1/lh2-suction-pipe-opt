import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import lh2_pipeline as p
from physics_baseline import PhysicsFeatureTransformer, PHYSICS_FEATURE_NAME
from predict import predict
from optimize import CachedModels, design_frame, objective_values


class PhysicsFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.data = pd.DataFrame({name: rng.uniform(lo, hi, 40)
                                  for name, (lo, hi) in p.BOUNDS.items()})
        self.data['log10_k_MLI'] = np.log10(self.data.k_MLI_W_mK)
        self.X = self.data[p.FEATURES]

    def test_transform_uses_only_sources_and_does_not_learn_targets(self):
        transformer = PhysicsFeatureTransformer(tuple(p.FEATURES))
        original = self.X.copy()
        transformed = transformer.fit_transform(self.X, np.full(len(self.X), 1e99))
        expected = np.log10(p.physics_heat_leak(self.X)/self.X.mdot_kg_h.to_numpy())
        np.testing.assert_allclose(transformed[PHYSICS_FEATURE_NAME], expected)
        self.assertEqual(list(transformed), [*p.FEATURES, PHYSICS_FEATURE_NAME])
        pd.testing.assert_frame_equal(original, self.X)
        contaminated = self.X.assign(Q_leak_W=-1, dp_design_Pa=np.nan,
                                      dp_thermal_Pa=1e100, case_id='ignored', vaporized=1)
        contaminated[PHYSICS_FEATURE_NAME] = -999
        pd.testing.assert_frame_equal(transformer.transform(contaminated), transformed)
        pd.testing.assert_frame_equal(clone(transformer).fit(self.X, np.zeros(len(self.X))).transform(self.X), transformed)
        self.assertEqual(list(transformer.get_feature_names_out()), list(transformed))

    def test_pipeline_and_kernel_dimensions_with_option(self):
        models = p.candidates(target='Q_leak_W', physics_feature=True)
        for name, (model, _) in models.items():
            if name == 'physics_baseline':
                self.assertIsInstance(model, p.PhysicsBaseline)
            else:
                self.assertIsInstance(model.regressor.steps[0][1], PhysicsFeatureTransformer)
        for kernel in models['gpr'][1]['regressor__model__kernel']:
            self.assertEqual(len(kernel.k1.k2.length_scale), 7)
        self.assertNotIn('physics_feature', dict(p.candidates()['linear'][0].regressor.steps))
        model = clone(models['linear'][0]).fit(self.X.iloc[:25], 1+self.X.D_mm.iloc[:25])
        expected = PhysicsFeatureTransformer(tuple(p.FEATURES)).fit_transform(self.X.iloc[:25])
        np.testing.assert_allclose(model.regressor_.named_steps['scale'].mean_, expected.mean())
        self.assertEqual(len(model.regressor_.named_steps['scale'].mean_), 7)

    def test_saved_pipeline_predict_and_optimizer_include_same_feature_and_sigma(self):
        model, space = p.candidates(target='dp_thermal_Pa', physics_feature=True)['gpr']
        model.set_params(regressor__model__kernel=space['regressor__model__kernel'][0],
                         regressor__model__optimizer=None)
        model.fit(self.X, 1+self.X.D_mm)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            joblib.dump(model, out/'dp_thermal_Pa__gpr.joblib')
            joblib.dump(p.PhysicsBaseline(), out/'Q_leak_W__physics_baseline.joblib')
            (out/'manifest.json').write_text(json.dumps({'selected': {
                'Q_leak_W': 'physics_baseline', 'dp_thermal_Pa': 'gpr'}, 'physics_feature': True}))
            expected = predict(self.data, out)
            cached = CachedModels(out)
            actual = cached.evaluate(self.data)
            np.testing.assert_allclose(actual.dp_thermal_pred_Pa, expected.dp_thermal_Pa_pred)
            np.testing.assert_allclose(actual.dp_design_pred_Pa, expected.dp_design_Pa_pred)
            sigma = model.regressor_.predict(self.X, return_std=True)[1]
            np.testing.assert_allclose(actual.sigma_log10_dp_thermal, sigma)
            scenario = dict(f_Hz=6, L_m=2, mdot_kg_h=80, k_MLI_W_mK=8e-5)
            designs = [[12, 2], [25, 10], [50, 25]]
            objectives, constraints = objective_values(cached, designs, scenario)
            predicted = predict(design_frame(designs, scenario), out)
            np.testing.assert_allclose(objectives[:, 1], predicted.dp_design_Pa_pred)
            np.testing.assert_allclose(constraints[:, 0], predicted.dp_design_Pa_pred-48300)


if __name__ == '__main__':
    unittest.main()
