"""Post-hoc thermal-only margin on saved reverification points; no optimization."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lh2_pipeline import FEATURES

OUT = Path(__file__).resolve().parent / 'outputs/full10k/analysis'


def conservative_table(points, thermal):
    table = points.copy()
    features = table.assign(log10_k_MLI=np.log10(table.k_MLI_W_mK))[FEATURES]
    _, sigma = thermal.regressor_.predict(features, return_std=True)
    table['sigma_log10_dp_thermal'] = sigma
    # First-order unit conversion: d(10**z)/dz = ln(10)*10**z.
    # Thus sigma_Pa ≈ ln(10)*thermal_prediction*sigma_log10 (thermal only).
    table['sigma_thermal_Pa'] = np.log(10) * table.dp_thermal_pred_Pa * sigma
    table['dp_conservative_Pa'] = table.dp_design_pred_Pa + 2*table.sigma_thermal_Pa
    table['conservative_safe'] = table.dp_conservative_Pa <= 48300
    table['conservative_agreement'] = table.conservative_safe == table.solver_constraint_satisfied
    columns = ['hyd', 'point_id', 'scenario', 'role', 'dp_design_pred_Pa',
               'sigma_log10_dp_thermal', 'sigma_thermal_Pa', 'dp_conservative_Pa',
               'dp_design_Pa', 'predicted_constraint_satisfied', 'conservative_safe',
               'solver_constraint_satisfied', 'conservative_agreement']
    return table[columns]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = OUT.parent
    points = pd.read_csv(root/'reverification/T4_reverification.csv')
    thermal = joblib.load(root/'models/dp_thermal_Pa__gpr.joblib')
    table = conservative_table(points, thermal)
    table.to_csv(OUT/'T_conservative.csv', index=False)
    print(table.groupby('hyd')[['conservative_safe', 'conservative_agreement']].sum())
    print(table.loc[(table.hyd == 'gpr') & (table.role == 'C')].to_string(index=False))


if __name__ == '__main__':
    main()
