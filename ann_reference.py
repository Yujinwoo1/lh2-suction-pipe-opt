"""Explicit reference ANN configuration; unknown paper values stay unknown."""
import json
from pathlib import Path
import numpy as np
from scipy.special import exp10
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from training_support import SubsampledTargetRegressor

FIELDS=['hidden_layer_sizes','activation','solver','learning_rate','max_iter','n_train','source']
UNREPORTED='논문 미기재, 기본값 사용'


def load_ann_reference(path):
    spec=json.loads(Path(path).read_text(encoding='utf-8'))
    missing=[key for key in FIELDS if spec.get(key) is None or spec.get(key)=='' or spec.get(key)==[]]
    if missing:
        return spec,None,'ann_ref skipped: unfilled fields '+', '.join(missing)
    kwargs={key:value for key,value in spec.items() if key in FIELDS[:5] and value!=UNREPORTED}
    if 'hidden_layer_sizes' in kwargs:
        sizes=kwargs['hidden_layer_sizes']
        if not isinstance(sizes,list) or not sizes or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in sizes):
            raise ValueError('ann_ref hidden_layer_sizes must be a nonempty list of positive integers.')
        kwargs['hidden_layer_sizes']=tuple(sizes)
    n=spec['n_train']
    if n!=UNREPORTED and (isinstance(n,bool) or not isinstance(n,int) or n<2):
        raise ValueError('ann_ref n_train must be integer >=2 or the explicit unreported marker.')
    # Exactly the paper-provided settings; no guessed early stopping or target
    # labels as inputs. Six original features, no optional physics feature.
    model=SubsampledTargetRegressor(regressor=Pipeline([
        ('scale',StandardScaler()),('model',MLPRegressor(random_state=2026,**kwargs))]),
        func=np.log10,inverse_func=exp10,max_rows=n if n!=UNREPORTED else 2**31-1)
    return spec,model,'ann_ref enabled: fixed structure, zero tuning; original six inputs. n_train is capped to available fold training rows and actual counts recorded.'
