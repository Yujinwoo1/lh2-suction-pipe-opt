"""Fixed heat-leak reference from docs/codex_tasks.md P0-2."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def physics_heat_leak(df):
    """Fixed P0-2 formula in docs/codex_tasks.md; no fitted constants.

    Uses only FEATURES, including log10 conductivity; ignores IDs/solver outputs.
    Wall resistance is neglected; wall thickness=0.0015 m, h=5 W/m²K,
    hot/cold temperatures=300/23.6 K.
    """
    r_i = df['D_mm'].to_numpy(dtype=float)/2000
    r_w = r_i + 0.0015
    r_o = r_w + df['t_ins_mm'].to_numpy(dtype=float)/1000
    conductivity = np.power(10., df['log10_k_MLI'].to_numpy(dtype=float))
    length = df['L_m'].to_numpy(dtype=float)
    resistance = (np.log(r_o/r_w)/(2*np.pi*conductivity*length)
                  + 1/(5*2*np.pi*r_o*length))
    return (300-23.6)/resistance


class PhysicsBaseline(RegressorMixin, BaseEstimator):
    """Parameter-free heat-leak reference; fit intentionally learns nothing."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return physics_heat_leak(X)
