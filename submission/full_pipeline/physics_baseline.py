"""Q_leak 물리식 기준선으로 열침입에 AI가 필요한지 비교한다.

파이프라인 단계: 모델 학습과 비교
입력: 원본 입력 6개가 담긴 표를 받는다.
출력: 물리식 열침입 예측과 선택적으로 추가할 log10(Q_phys/mdot_kg_h) 특성을 반환한다.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


def physics_heat_leak(df):
    """AI와 비교할 기준이므로 상수를 학습하지 않고 원통 전도와 외부 대류식을 사용한다.
    벽의 열저항은 무시하며, 원본 입력 6개만 사용한다.
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
    """물리식 자체와 비교하기 위해 fit에서 상수를 조정하지 않는다."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return physics_heat_leak(X)


PHYSICS_FEATURE_NAME = 'log10_Q_phys_per_mdot'


class PhysicsFeatureTransformer(TransformerMixin, BaseEstimator):
    """log10(Q_phys/mdot_kg_h)로 유량당 열 유입을 입력에 반영한다.
    원본 입력 6개로만 계산해 정답이나 ID가 모델에 들어가는 것을 막는다.
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.features)
        self.feature_names_in_ = np.asarray(self.features, dtype=object)
        return self

    def transform(self, X):
        data = X.loc[:, list(self.features)].copy()
        data[PHYSICS_FEATURE_NAME] = np.log10(
            physics_heat_leak(data)/data['mdot_kg_h'].to_numpy(dtype=float))
        return data

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.features, PHYSICS_FEATURE_NAME], dtype=object)
