"""Aggregate completed runs: average scenes within each seed, then compare seeds.

Each random seed is one repetition. This deliberately does not treat the
scene-by-seed rows as independent experimental repetitions.
All artifacts must share the current coordinate/frequency input contract.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS = ('rmse_db', 'mae_db', 'bias_db')



def _read_rows(path: Path, expected_scenes, expected_methods):
    if not expected_methods:
        return []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(scene, method) for scene in expected_scenes for method in expected_methods}
    actual = [(row['scene_id'], row['approach']) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f'{path}: missing, duplicated or unexpected scene/method rows')
    for row in rows:
        if not row.get('building_id'):
            raise ValueError(f'{path}: missing building_id')
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not np.isfinite(row[metric]):
                raise ValueError(f'{path}: nonfinite {metric}')
        if row['rmse_db'] < 0 or row['mae_db'] < 0 or int(row['n_points']) < 1:
            raise ValueError(f'{path}: invalid error or evaluation count')
    return rows


def _scope(config, summary):
    normalized = copy.deepcopy(config)
    normalized.pop('output_dir', None)
    normalized.get('train', {}).pop('seed', None)
    return json.dumps({
        'config': normalized,
        'scenes': sorted(summary['selected_scenes']),
        'methods': sorted(summary['methods']),
        'baselines': sorted(summary.get('baselines', {})),
        'evaluation_limited': summary['evaluation_limited'],
        'input_contract': summary['input_contract'],
        'method_prediction_contracts': summary['method_prediction_contracts'],
        'source_sha256': summary.get('source_sha256'),
        'dataset_sha256': summary.get('dataset_sha256'),
    }, sort_keys=True)


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def _write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root, seeds=range(10), output_dir=None):
    root = Path(root)
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Choose distinct nonnegative seed IDs')
    out = Path(output_dir) if output_dir is not None else root / 'seed_summary'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is nonempty: {out}')
    reference = None
    per_seed, gain_rows = [], []
    scene_count = None
    limited = False
    for seed in seeds:
        run_dir = root / f'seed_{seed}'
        with (run_dir / 'summary.json').open() as stream:
            summary = json.load(stream)
        with (run_dir / 'config.yml').open() as stream:
            config = yaml.safe_load(stream)
        if not summary.get('complete'):
            raise ValueError(f'Seed {seed} is incomplete')
        if int(config['train']['seed']) != seed:
            raise ValueError(f'Seed {seed} does not match its saved configuration')
        scenes, methods = summary['selected_scenes'], summary['methods']
        baselines = list(summary.get('baselines', {}))
        scope = _scope(config, summary)
        if reference is None:
            reference = scope
            scene_count = len(scenes)
            limited = bool(summary['evaluation_limited'])
        elif scope != reference:
            raise ValueError(f'Seed {seed}: scenes, methods, input contracts, budgets, configuration or code/data fingerprints differ')
        main_rows = _read_rows(run_dir / 'main_per_scene.csv', scenes, methods)
        baseline_rows = _read_rows(run_dir / 'baseline_per_scene.csv', scenes, baselines)
        for group_name, group_rows, group_methods in (
                ('main_experiments', main_rows, methods), ('baselines', baseline_rows, baselines)):
            for method in group_methods:
                selected = [row for row in group_rows if row['approach'] == method]
                result = dict(seed=seed, group=group_name, approach=method, n_scenes=len(selected))
                for metric in METRICS:
                    result['scene_mean_' + metric] = float(np.mean([row[metric] for row in selected]))
                buildings = sorted({row['building_id'] for row in selected})
                result['building_mean_rmse_db'] = float(np.mean([
                    np.mean([row['rmse_db'] for row in selected if row['building_id'] == building])
                    for building in buildings]))
                # Verify persisted summary agrees with the current CSV rather than
                # accepting unrelated/stale summary files with matching names.
                recorded = summary[group_name][method]
                for metric in METRICS:
                    key = 'scene_mean_' + metric
                    if not np.isclose(recorded[key], result[key], rtol=1e-8, atol=1e-8):
                        raise ValueError(f'Seed {seed}/{method}: summary and CSV disagree for {key}')
                per_seed.append(result)
        by_scene = {(row['scene_id'], row['approach']): row['rmse_db'] for row in main_rows}
        if {'E2_scratch', 'E3_transfer'}.issubset(methods):
            for scene in scenes:
                gain_rows.append(dict(seed=seed, scene_id=scene,
                                      prediction_contract=PREDICTION_CONTRACT,
                                      input_contract=INPUT_CONTRACT,
                                      model_prediction_contract=method_prediction_contract('E2_scratch'),
                                      scratch_minus_transfer_rmse_db=(by_scene[scene, 'E2_scratch']
                                                                       - by_scene[scene, 'E3_transfer'])))
    report = dict(seeds=seeds, n_seeds=len(seeds), n_scenes_per_seed=scene_count,
                  **prediction_metadata(), metric_unit='dB', loss_unit='dB^2',
                  source_sha256=summary.get('source_sha256'), dataset_sha256=summary.get('dataset_sha256'),
                  method_prediction_contracts=summary['method_prediction_contracts'],
                  evaluation_limited=limited,
                  aggregation='scene macro mean within each seed; mean and sample standard deviation across seeds',
                  std_ddof=1 if len(seeds) > 1 else 0,
                  main_experiments={}, baselines={})
    for group, method in sorted({(row['group'], row['approach']) for row in per_seed}):
        selected = [row for row in per_seed if row['group'] == group and row['approach'] == method]
        result = dict(n_seeds=len(selected), n_scenes_per_seed=scene_count,
                      model_prediction_contract=method_prediction_contract(method))
        for metric in (*('scene_mean_' + name for name in METRICS), 'building_mean_rmse_db'):
            result[metric] = _statistics([row[metric] for row in selected])
        report[group][method] = result
    if gain_rows:
        gains = [np.mean([row['scratch_minus_transfer_rmse_db'] for row in gain_rows if row['seed'] == seed])
                 for seed in seeds]
        report['scratch_minus_transfer_rmse_db'] = _statistics(gains)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / 'per_seed.csv', per_seed)
    _write_csv(out / 'paired_scene_gains.csv', gain_rows)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Aggregated {len(seeds)} seeds, {scene_count} scenes each: {out}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Parent of seed_0, seed_1, ...')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=9, help='Inclusive; default 0..9 means ten repetitions')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        parser.error('--seed-end must be at least --seed-start')
    aggregate(args.root, range(args.seed_start, args.seed_end + 1), args.output_dir)


if __name__ == '__main__':
    main()
