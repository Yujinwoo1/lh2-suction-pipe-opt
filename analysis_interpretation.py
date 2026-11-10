"""Saved GPR ARD scales and two raw-feature PDPs on development inputs only."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lh2_pipeline import FEATURES, plt
from training_support import sample_indices

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/full10k/analysis'
GPR_COLOR = '#2878B5'
MLP_COLOR = '#E07B39'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = ROOT/'outputs/full10k/models'
    split = pd.read_csv(models/'split.csv')
    ids = split.loc[split.split == 'development', 'case_id']
    # Discard holdout rows immediately; neither their targets nor predictions are used.
    dev = pd.concat([part.loc[part.case_id.isin(ids)] for part in
                     pd.read_csv(ROOT/'data/dataset_ai_10k.csv', chunksize=1000)])
    dev['log10_k_MLI'] = np.log10(dev.k_MLI_W_mK)
    scales = pd.DataFrame({'input': [*FEATURES, 'log10_Q_phys_per_mdot']})
    for target in ['Q_leak_W', 'dp_thermal_Pa']:
        fitted = joblib.load(models/(target+'__gpr.joblib'))
        scales[target] = fitted.regressor_.named_steps['model'].kernel_.k1.k2.length_scale
    scales.to_csv(OUT/'T_length_scales.csv', index=False)
    heat = joblib.load(models/'Q_leak_W__gpr.joblib')
    # Fixed 200 development backgrounds keep this simple empirical PDP inexpensive.
    background = dev.iloc[sample_indices(len(dev), 200)][FEATURES].copy()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    with threadpool_limits(limits=1):
        for axis, feature, label in zip(axes, ['t_ins_mm', 'log10_k_MLI'],
                                       ['Insulation thickness [mm]', 'MLI conductivity [W/(m K)]']):
            grid = np.linspace(dev[feature].min(), dev[feature].max(), 30)
            means = []
            for value in grid:
                changed = background.copy()
                changed[feature] = value
                # Pipeline recomputes the physics feature after changing a raw input.
                means.append(float(heat.predict(changed).mean()))
                rows.append({'feature': feature, 'value': value, 'mean_Q_leak_W': means[-1]})
            x = 10**grid if feature == 'log10_k_MLI' else grid
            axis.plot(x, means, color=GPR_COLOR, label='GPR')
            if feature == 'log10_k_MLI':
                axis.set_xscale('log')
            axis.set(xlabel=label, ylabel='Mean Q_leak [W]')
            axis.legend()
    fig.tight_layout()
    fig.savefig(OUT/'F_pdp.png', dpi=300, facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'T_pdp.csv', index=False)
    dev.iloc[sample_indices(len(dev), 200)][['case_id']].to_csv(OUT/'pdp_background_ids.csv', index=False)
    print(scales.to_string(index=False))
    print(pd.DataFrame(rows).groupby('feature').mean_Q_leak_W.agg(['first', 'last']))


if __name__ == '__main__':
    main()
