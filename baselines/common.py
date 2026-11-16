"""Four-metric baseline reports with equal scene weighting and seed repetitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_utils import METRIC_FIELDS
from scripts.aggregate_seeds import (_read_rows, _statistics, _write_csv,
                                     aggregate as aggregate_seed_reports)


METHOD_IDS = {'fspl': 'E1_fspl', 'knn': 'B1_knn', 'mlp': 'B2_mlp', 'nerf2': 'B3_nerf2'}
_METHOD_NAMES = {value: key for key, value in METHOD_IDS.items()}


def normalize_methods(names=None):
    """Resolve CLI names or experiment IDs in the requested order."""
    if names is None:
        return list(METHOD_IDS.values())
    aliases = {**METHOD_IDS, **{key.lower(): key for key in _METHOD_NAMES}}
    names = [names] if isinstance(names, str) else names
    try:
        methods = [aliases[name.lower()] for name in names]
    except (KeyError, AttributeError) as error:
        raise ValueError(f'Choose baseline methods from {list(METHOD_IDS)}') from error
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Choose at least one baseline, without duplicates')
    return methods


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def _read_baseline_run(run_dir, summary=None):
    run_dir = Path(run_dir)
    summary = summary if summary is not None else json.loads((run_dir / 'summary.json').read_text())
    if not summary['complete'] or summary['methods']:
        raise ValueError(f'{run_dir}: a completed baseline-only run is required')
    rows = _read_rows(run_dir / 'baseline_per_scene.csv', summary['selected_scenes'],
                      list(summary['baselines']))
    selected = [dict(seed=summary['seed'], scene_id=row['scene_id'], building_id=row['building_id'],
                     method=_METHOD_NAMES[row['approach']], approach=row['approach'],
                     **{field: row[field] for field in METRIC_FIELDS}) for row in rows]
    return summary, selected


def save_baseline_summary(run_dir, report):
    """Save the four scene-mean metrics plus individual scene results."""
    run_dir = Path(run_dir)
    path = run_dir / 'baseline_summary.json'
    if path.exists():
        raise ValueError(f'Baseline summary already exists: {path}')
    summary, rows = _read_baseline_run(run_dir, report)
    result = dict(complete=True, seed=summary['seed'], selected_scenes=summary['selected_scenes'],
                  label_normalization=summary.get('label_normalization'),
                  evaluation_limited=summary['evaluation_limited'],
                  aggregation='equal scene weighting within this seed', baselines={})
    for method in summary['baselines']:
        selected = [row for row in rows if row['approach'] == method]
        result['baselines'][_METHOD_NAMES[method]] = dict(
            approach=method,
            **{field: float(np.mean([row[field] for row in selected])) for field in METRIC_FIELDS},
            per_scene={row['scene_id']: {field: row[field] for field in METRIC_FIELDS}
                       for row in selected})
    _write_json(path, result)
    return result


def aggregate_baseline_sweep(root, seeds=range(10), output_dir=None):
    """Report each scene over seeds, then scene macro means over seeds."""
    root, seeds = Path(root), list(seeds)
    out = Path(output_dir) if output_dir is not None else root / 'sweep_summary'
    all_rows = [row for seed in seeds for row in _read_baseline_run(root / f'seed_{seed}')[1]]
    # The shared seed aggregator checks completeness and mode selection once.
    report = aggregate_seed_reports(root, seeds, out)
    scenes = sorted({row['scene_id'] for row in all_rows})
    methods = sorted({row['approach'] for row in all_rows})
    per_scene, scene_rows = {}, []
    for scene in scenes:
        per_scene[scene] = {}
        for method in methods:
            selected = [row for row in all_rows if row['scene_id'] == scene and row['approach'] == method]
            statistics = {field: _statistics([row[field] for row in selected]) for field in METRIC_FIELDS}
            per_scene[scene][_METHOD_NAMES[method]] = dict(approach=method, **statistics)
            row = dict(scene_id=scene, method=_METHOD_NAMES[method], approach=method, n_seeds=len(seeds))
            for field, values in statistics.items():
                row[field + '_mean'], row[field + '_std'] = values['mean'], values['std']
            scene_rows.append(row)
    report.update(complete=True, comparison='baselines_only', per_scene=per_scene,
                  n_scene_seed_method_runs=len(all_rows))
    _write_csv(out / 'per_scene_seed.csv', all_rows)
    _write_csv(out / 'per_scene_summary.csv', scene_rows)
    _write_json(out / 'summary.json', report)
    print(f'Baseline comparison: {len(scenes)} scenes × {len(seeds)} seeds × {len(methods)} methods: {out}')
    return report
