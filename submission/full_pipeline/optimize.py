"""열침입, 전체 압력, 외경을 줄이면서 dp_design ≤ 48.3 kPa인 후보를 찾는다.

파이프라인 단계: 최적화
입력: 선정 모델과 시나리오 설정을 읽는다. compare는 후보표와 솔버 결과도 읽는다.
출력: 시나리오별 Pareto와 격자 비교, 재검증 후보, F4 그림 또는 솔버 오차표를 저장한다.
"""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from lh2_pipeline import ROOT, SEED, FEATURES, BOUNDS, TARGETS, rigid_pressure, plt, save_json
from artifact_state import write_receipt
from predict import validate_inputs
from hydraulics import load_ratio_model, predicted_ratio
from sklearn.gaussian_process import GaussianProcessRegressor

OBJECTIVES = ['Q_leak_pred_W', 'dp_design_pred_Pa', 'D_o_mm']
NOTE = ('Surrogate Pareto only, not solver validation. B is minimum Euclidean '
        'distance to ideal after Pareto min-max normalization. '
        'Extras use farthest-point sampling; within 1% distance ties prefer '
        'higher thermal log10 sigma, then proximity to 38.3–48.3 kPa.')


class CachedModels:
    def __init__(self, directory, hyd_model=None):
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        self.models = {target: joblib.load(directory/(target+'__'+self.manifest['selected'][target]+'.joblib'))
                       for target in TARGETS}
        self.hyd_mode,self.ratio_model=load_ratio_model(directory,self.manifest,hyd_model)

    def evaluate(self, frame):
        data = validate_inputs(frame)
        predictions = {target: [] for target in TARGETS}
        deviations = []
        ratios = []
        for start in range(0, len(data), 1000):
            features = data[FEATURES].iloc[start:start+1000]
            ratios.extend(predicted_ratio(self.ratio_model,features))
            for target, model in self.models.items():
                predictions[target].extend(model.predict(features))
            thermal = self.models['dp_thermal_Pa'].regressor_
            if isinstance(thermal.named_steps['model'], GaussianProcessRegressor):
                deviations.extend(thermal.predict(features, return_std=True)[1])
            else:
                deviations.extend(np.full(len(features), np.nan))
        data['Q_leak_pred_W'] = predictions['Q_leak_W']
        data['dp_thermal_pred_Pa'] = predictions['dp_thermal_Pa']
        data['hydraulic_ratio_pred']=ratios
        data['hyd_model']=self.hyd_mode
        data['dp_design_pred_Pa'] = data.dp_thermal_pred_Pa+data.hydraulic_ratio_pred*rigid_pressure(data)
        data['D_o_mm'] = data.D_mm+3+2*data.t_ins_mm
        data['sigma_log10_dp_thermal'] = deviations
        data['n_star'] = 1040/(4*data.L_m*data.f_Hz)
        return data


def design_frame(designs, scenario):
    data = pd.DataFrame(designs, columns=['D_mm', 't_ins_mm'])
    for column in list(BOUNDS)[2:]:
        data[column] = scenario[column]
    return validate_inputs(data)


def objective_values(models, designs, scenario):
    values = models.evaluate(design_frame(designs, scenario))
    return values[OBJECTIVES].to_numpy(), (values.dp_design_pred_Pa.to_numpy()-48300).reshape(-1, 1)


def design_problem(models, scenario):
    from pymoo.core.problem import Problem

    class DesignProblem(Problem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=3, n_ieq_constr=1, xl=[12, 2], xu=[50, 25])

        def _evaluate(self, designs, out, *args, **kwargs):
            out['F'], out['G'] = objective_values(models, designs, scenario)

    return DesignProblem()


def pareto(frame):
    """대리모델에서 제약을 만족하는 비지배 후보만 남긴다.
    솔버 검증과 구분하기 위해 예측 기준의 Pareto로 다룬다."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    feasible = frame.loc[frame.dp_design_pred_Pa <= 48300].reset_index(drop=True)
    if feasible.empty:
        return feasible
    indices = NonDominatedSorting(method='efficient_non_dominated_sort').do(
        feasible[OBJECTIVES].to_numpy(), only_non_dominated_front=True)
    return feasible.iloc[indices].sort_values(OBJECTIVES).reset_index(drop=True)


def normalize(values, ideal, nadir):
    return (values-ideal)/np.where(nadir > ideal, nadir-ideal, 1)


def candidates(front):
    """B는 Pareto 목적값을 0–1로 정규화한 뒤 이상점까지의 거리가 가장 가까운 점으로 정의한다.
    단위가 다른 세 목적을 같은 척도에서 비교하기 위한 기준이다."""
    front = front.drop_duplicates(['D_mm', 't_ins_mm']).reset_index(drop=True)
    values = front[OBJECTIVES].to_numpy()
    scaled = normalize(values, values.min(axis=0), values.max(axis=0))
    anchors = [np.argmin(values[:, 0]), np.argmin(np.linalg.norm(scaled, axis=1)), np.argmin(values[:, 2])]
    chosen, roles = [], []
    for index, role in zip(anchors, ['A', 'B', 'C']):
        chosen.append(int(index))
        roles.append(role)
    for _ in range(min(6, len(front)-len(set(chosen)))):
        distance = np.linalg.norm(scaled[:, None]-scaled[chosen], axis=2).min(axis=1)
        distance[chosen] = -np.inf
        eligible = np.flatnonzero(distance >= distance.max()-.01)
        sigma = np.nan_to_num(front.sigma_log10_dp_thermal.to_numpy(), nan=-np.inf)
        pressure = front.dp_design_pred_Pa.to_numpy()
        boundary_distance = np.maximum(38300-pressure, 0)
        index = sorted(eligible, key=lambda index: (-sigma[index], boundary_distance[index], -pressure[index], index))[0]
        chosen.append(int(index))
        roles.append('extra')
    result = front.iloc[chosen].copy()
    result['role'] = roles
    return result


def run(models_dir, scenarios_path, out, hyd_model=None):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.indicators.hv import HV
    scenarios = json.loads(scenarios_path.read_text(encoding='utf-8'))
    identifiers = [scenario['id'] for scenario in scenarios]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('Scenario IDs must be unique and nonempty.')
    for scenario in scenarios:
        if not scenario['id'].isalnum():
            raise ValueError('Scenario IDs must be alphanumeric.')
        design_frame([[12, 2], [50, 25]], scenario)
    models = CachedModels(models_dir,hyd_model)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'optimization.json', {'seed': SEED, 'population': 100, 'generations': 200,
              'grid_shape': [200, 200], 'hyd_model':models.hyd_mode, 'models': str(models_dir), 'selected': models.manifest['selected'],
              'note': NOTE, 'scenarios': scenarios, 'hv_reference': [1.1]*3})
    points, summaries = [], []
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6*len(scenarios), 5), squeeze=False)
    for scenario, axis in zip(scenarios, axes[0]):
        identifier = scenario['id']
        n_star = 1040/(4*scenario['L_m']*scenario['f_Hz'])
        if n_star < 20 and abs(n_star-round(n_star)) < .1:
            warnings.warn(f'{identifier}: low-order resonance rule triggered, n*={n_star:.6f}')
        diameters, thicknesses = np.meshgrid(np.linspace(12, 50, 200), np.linspace(2, 25, 200))
        grid = pareto(models.evaluate(design_frame(np.column_stack([diameters.ravel(), thicknesses.ravel()]), scenario)))
        if grid.empty:
            raise ValueError(f'{identifier}: grid has no feasible designs.')
        result = minimize(design_problem(models, scenario), NSGA2(pop_size=100), ('n_gen', 200), seed=SEED)
        if result.X is None:
            raise ValueError(f'{identifier}: NSGA-II found no feasible designs.')
        front = pareto(models.evaluate(design_frame(result.X, scenario)))
        for table, prefix in [(front, 'pareto'), (grid, 'grid_pareto')]:
            table['scenario'] = identifier
            table.to_csv(out/f'{prefix}_{identifier}.csv', index=False)
        ideal, nadir = grid[OBJECTIVES].min().to_numpy(), grid[OBJECTIVES].max().to_numpy()
        indicator = HV(ref_point=np.array([1.1]*3))
        grid_hv = indicator(normalize(grid[OBJECTIVES].to_numpy(), ideal, nadir))
        nsga_hv = indicator(normalize(front[OBJECTIVES].to_numpy(), ideal, nadir))
        summaries.append({'scenario': identifier, 'n_star': n_star, 'nsga_points': len(front), 'grid_points': len(grid),
                          'nsga_hv': nsga_hv, 'grid_hv': grid_hv, 'hv_ratio': nsga_hv/grid_hv,
                          **{f'ideal_{name}': ideal[index] for index, name in enumerate(OBJECTIVES)},
                          **{f'nadir_{name}': nadir[index] for index, name in enumerate(OBJECTIVES)}})
        selected = candidates(pareto(pd.concat([front, grid], ignore_index=True)))
        selected['scenario'] = identifier
        points.append(selected)
        axis.scatter(grid.Q_leak_pred_W, grid.dp_design_pred_Pa/1000, c='lightgray', s=8, label='Grid surrogate Pareto')
        scatter = axis.scatter(front.Q_leak_pred_W, front.dp_design_pred_Pa/1000, c=front.D_o_mm, s=15)
        for row in selected[selected.role != 'extra'].itertuples():
            axis.scatter(row.Q_leak_pred_W, row.dp_design_pred_Pa/1000, marker='x', c='black')
            axis.annotate(row.role, (row.Q_leak_pred_W, row.dp_design_pred_Pa/1000))
        axis.axhline(48.3, color='red', linestyle='--', label='48.3 kPa constraint')
        axis.set(xlabel='Q_leak [W]', ylabel='dp_design [kPa]', title=f'{identifier}: surrogate predictions')
        axis.legend(fontsize=8)
        fig.colorbar(scatter, ax=axis, label='D_o [mm]')
        print(f'{identifier}: grid Pareto={len(grid)}, NSGA-II={len(front)}, HV ratio={nsga_hv/grid_hv:.6f}', flush=True)
    verification = pd.concat(points, ignore_index=True)
    verification.insert(0, 'point_id', [f'V{index+1:03d}' for index in range(len(verification))])
    columns = ['point_id', 'scenario', 'role', *BOUNDS, *OBJECTIVES[:1], 'dp_thermal_pred_Pa',
               'dp_design_pred_Pa', 'D_o_mm', 'sigma_log10_dp_thermal', 'n_star']
    verification[columns].to_csv(out/'verification_points.csv', index=False)
    verification[list(BOUNDS)].to_csv(out/'verification_inputs.csv', index=False)
    verification[columns].assign(selection_definition=NOTE).to_csv(out/'T3_candidates.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'hv_summary.csv', index=False)
    fig.tight_layout()
    fig.savefig(out/'F4.png', dpi=300, facecolor='white')
    plt.close(fig)
    write_receipt(out,'optimization_complete.json',{'hyd_model':models.hyd_mode})
    print(f'{NOTE}\nVerification rows: {len(verification)}; output: {out}')


def compare(points_path, solver_path, output):
    points, solver = pd.read_csv(points_path), pd.read_csv(solver_path)
    if points.point_id.duplicated().any() or solver.point_id.duplicated().any():
        raise ValueError('point_id must be unique.')
    if set(points.point_id) != set(solver.point_id):
        raise ValueError('Solver point_id set must match verification points exactly.')
    actual = solver[['point_id', 'Q_leak_W', 'dp_design_Pa']]
    if not np.isfinite(actual.iloc[:, 1:].to_numpy(dtype=float)).all() or (actual.iloc[:, 1:] <= 0).any().any():
        raise ValueError('Solver outputs must be finite and positive.')
    table = points.merge(actual, on='point_id', validate='one_to_one')
    for target, prediction in [('Q_leak_W', 'Q_leak_pred_W'), ('dp_design_Pa', 'dp_design_pred_Pa')]:
        table[target+'_error_pct'] = 100*(table[prediction]/table[target]-1)
        table[target+'_abs_error_pct'] = abs(table[target+'_error_pct'])
    table['predicted_constraint_satisfied'] = table.dp_design_pred_Pa <= 48300
    table['solver_constraint_satisfied'] = table.dp_design_Pa <= 48300
    table['constraint_agreement'] = table.predicted_constraint_satisfied == table.solver_constraint_satisfied
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    search = commands.add_parser('run')
    search.add_argument('--models', type=Path, default=ROOT/'outputs/pilot')
    search.add_argument('--scenarios', type=Path, default=ROOT/'configs/scenarios.json')
    search.add_argument('--out', type=Path)
    search.add_argument('--hyd-model',choices=['const','gpr'])
    comparison = commands.add_parser('compare')
    comparison.add_argument('--points', type=Path, required=True)
    comparison.add_argument('--solver', type=Path, required=True)
    comparison.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.models, args.scenarios, args.out or ROOT/'outputs'/('opt_'+args.models.resolve().name),args.hyd_model)
    else:
        table = compare(args.points, args.solver, args.output or args.points.parent/'T4_comparison.csv')
        print(f'T4: {len(table)} rows; constraint agreement={table.constraint_agreement.mean():.3f}')


if __name__ == '__main__':
    main()
