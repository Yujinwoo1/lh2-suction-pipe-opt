"""Predict new in-domain designs using the development-selected models."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lh2_pipeline import FEATURES, BOUNDS, TARGETS, rigid_pressure


def validate_inputs(frame):
    missing = set(BOUNDS)-set(frame)
    if missing:
        raise ValueError(f'Missing inputs: {sorted(missing)}')
    data = frame.copy()
    if not np.isfinite(data[list(BOUNDS)].to_numpy(dtype=float)).all():
        raise ValueError('Inputs must be finite numeric values.')
    for col, (lo, hi) in BOUNDS.items():
        if not data[col].between(lo, hi).all():
            raise ValueError(f'{col} outside training design domain; extrapolation is not supported.')
    data['log10_k_MLI'] = np.log10(data.k_MLI_W_mK)
    return data


def predict(frame, model_dir):
    data = validate_inputs(frame)
    manifest = json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    for target in TARGETS:
        name = manifest['selected'][target]
        model = joblib.load(model_dir/(target+'__'+name+'.joblib'))
        data[target+'_pred'] = model.predict(data[FEATURES])
    data['dp_design_Pa_pred'] = data.dp_thermal_Pa_pred+1.8*rigid_pressure(data)
    data['D_o_mm'] = data.D_mm+3+2*data.t_ins_mm
    data['predicted_constraint_satisfied'] = data.dp_design_Pa_pred <= 48300
    data['n_star'] = 1040/(4*data.L_m*data.f_Hz)
    # Report the exact low-order rule in the source, not a broader validated criterion.
    data['low_order_resonance_flag'] = (data.n_star < 20) & (abs(data.n_star-np.round(data.n_star)) < .1)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--models', type=Path, default=Path('outputs/pilot'))
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output file.')
    result = predict(pd.read_csv(args.input), args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print('Pilot predictions only. Constraint and resonance flags require solver review.')
