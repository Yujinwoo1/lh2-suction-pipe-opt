"""Development-only diameter extrapolation, fixed selected GPR/MLP settings."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from lh2_pipeline import FEATURES, SEED, plt
from analysis_interpretation import GPR_COLOR, MLP_COLOR

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/full10k/analysis'


def diameter_split(dev):
    inside = dev.loc[dev.D_mm < 40]
    train, validation = train_test_split(inside, test_size=.2, random_state=SEED)
    outside = dev.loc[dev.D_mm >= 40]
    return train, validation, outside


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    directory = ROOT/'outputs/full10k/models'
    split = pd.read_csv(directory/'split.csv')
    ids = split.loc[split.split == 'development', 'case_id']
    dev = pd.concat([part.loc[part.case_id.isin(ids)] for part in
                     pd.read_csv(ROOT/'data/dataset_ai_10k.csv', chunksize=1000)])
    dev['log10_k_MLI'] = np.log10(dev.k_MLI_W_mK)
    train, inside, outside = diameter_split(dev)
    assignments = pd.concat([part[['case_id', 'D_mm']].assign(analysis_split=label)
                             for label, part in [('train', train), ('inside', inside), ('outside', outside)]])
    rows, predictions, parameters = [], [], {}
    with threadpool_limits(limits=1):
        for name, color in [('gpr', GPR_COLOR), ('mlp', MLP_COLOR)]:
            saved = joblib.load(directory/('dp_thermal_Pa__'+name+'.joblib'))
            model = clone(saved)
            if name == 'gpr':
                # Freeze the final learned kernel as well as alpha; no new kernel search.
                model.set_params(max_rows=3000, regressor__model__kernel=clone(
                    saved.regressor_.named_steps['model'].kernel_),
                    regressor__model__optimizer=None, regressor__model__n_restarts_optimizer=0)
            parameters[name] = repr(model.get_params(deep=True))
            model.fit(train[FEATURES], train.dp_thermal_Pa)
            if name == 'gpr':
                sampled = train.iloc[model.sample_indices_].case_id
                assignments['gpr_training_sample'] = assignments.case_id.isin(sampled)
            for scope, frame in [('inside', inside), ('outside', outside)]:
                pred = model.predict(frame[FEATURES])
                rows.append({'model': name.upper(), 'scope': scope, 'n_evaluation': len(frame),
                             'n_train': len(model.sample_indices_) if name == 'gpr' else len(train),
                             'mape_pct': 100*mean_absolute_percentage_error(frame.dp_thermal_Pa, pred)})
                predictions.append(frame[['case_id', 'D_mm', 'dp_thermal_Pa']].assign(
                    model=name.upper(), scope=scope, predicted_Pa=pred))
            # Analysis models are deliberately not saved.
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'T_extrapolation.csv', index=False)
    pd.concat(predictions).to_csv(OUT/'extrapolation_predictions.csv', index=False)
    assignments.to_csv(OUT/'extrapolation_split.csv', index=False)
    (OUT/'extrapolation_parameters.json').write_text(json.dumps(
        {'seed': SEED, 'parameters': parameters, 'tuning_candidates': 0,
         'note': 'Fixed final settings, including full-development learned GPR kernel. '
                 'Conditional diagnostic, not an independently selected extrapolation model.'}, indent=2))
    fig, axis = plt.subplots(figsize=(6, 4), facecolor='white')
    for i, (name, color) in enumerate([('GPR', GPR_COLOR), ('MLP', MLP_COLOR)]):
        values = table.loc[table.model == name, 'mape_pct'].to_numpy()
        bars = axis.bar(np.arange(2)+(i-.5)*.35, values, width=.35, color=color, label=name)
        axis.bar_label(bars, fmt='%.2f', padding=3)
    axis.set(xticks=[0, 1], xticklabels=['Inside: D < 40 mm', 'Outside: D ≥ 40 mm'],
             ylabel='dp_thermal MAPE [%]')
    axis.legend()
    axis.margins(y=.2)
    fig.tight_layout()
    fig.savefig(OUT/'F_extrapolation.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
