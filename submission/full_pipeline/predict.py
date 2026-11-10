"""개발 자료로 선정한 모델을 학습 범위 안의 설계에만 적용한다."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lh2_pipeline import FEATURES, BOUNDS, TARGETS, rigid_pressure
from hydraulics import load_ratio_model, predicted_ratio


def validate_inputs(frame):
    """개발 자료의 외삽 시험에서 오차가 약 4배 늘어 범위 밖 입력을 거부함.
    이 시험은 고정 설정의 진단이며 독립적인 외삽 성능 검증은 아님.
    """
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


def predict(frame, model_dir, hyd_model=None):
    data = validate_inputs(frame)
    manifest = json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    for target in TARGETS:
        name = manifest['selected'][target]
        model = joblib.load(model_dir/(target+'__'+name+'.joblib'))
        data[target+'_pred'] = model.predict(data[FEATURES])
    mode,ratio_model=load_ratio_model(model_dir,manifest,hyd_model)
    data['hydraulic_ratio_pred']=predicted_ratio(ratio_model,data[FEATURES])
    data['hyd_model']=mode
    data['dp_design_Pa_pred'] = data.dp_thermal_Pa_pred+data.hydraulic_ratio_pred*rigid_pressure(data)
    data['D_o_mm'] = data.D_mm+3+2*data.t_ins_mm
    data['predicted_constraint_satisfied'] = data.dp_design_Pa_pred <= 48300
    data['n_star'] = 1040/(4*data.L_m*data.f_Hz)
    # 공진 표시는 원본의 저차 판별식이며 솔버 확인을 대신하지 않는다.
    data['low_order_resonance_flag'] = (data.n_star < 20) & (abs(data.n_star-np.round(data.n_star)) < .1)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--models', type=Path, default=Path('outputs/pilot'))
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    parser.add_argument('--hyd-model',choices=['const','gpr'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output file.')
    result = predict(pd.read_csv(args.input), args.models,args.hyd_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print('Pilot predictions only. Constraint and resonance flags require solver review.')
