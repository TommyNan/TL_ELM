"""Average scenes within each seed, then report mean and sample std over seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import METRIC_FIELDS


def _read_rows(path, scenes, methods):
    """Read four metrics and require the complete scene/method matrix."""
    if not methods:
        return []
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    actual = [(row['scene_id'], row['approach']) for row in rows]
    expected = {(scene, method) for scene in scenes for method in methods}
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'{path}: incomplete or repeated scene/method results')
    for row in rows:
        row.update({field: float(row[field]) for field in METRIC_FIELDS})
    return rows


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def _write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def aggregate(root, seeds=range(10), output_dir=None, scenes=None):
    """Summarize each scene across seeds, then equal scene means within seeds."""
    root, seeds = Path(root), list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('Choose distinct nonnegative seeds')
    out = Path(output_dir) if output_dir is not None else root / 'seed_summary'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is nonempty: {out}')
    per_seed, per_scene_seed, reference = [], [], None
    for seed in seeds:
        run_dir = root / f'seed_{seed}'
        summary = json.loads((run_dir / 'summary.json').read_text())
        config = yaml.safe_load((run_dir / 'config.yml').read_text())
        if not summary['complete']:
            raise ValueError(f'Seed {seed} is incomplete')
        run_scenes, methods = summary['selected_scenes'], summary['methods']
        selected_scenes = list(run_scenes if scenes is None else scenes)
        if (not selected_scenes or len(set(selected_scenes)) != len(selected_scenes)
                or not set(selected_scenes).issubset(run_scenes)):
            raise ValueError(f'Choose distinct scenes from {run_scenes}')
        baselines = list(summary.get('baselines', {}))
        mode = config.get('model', {}).get('prediction_mode', 'fspl_residual') if methods else None
        normalization = summary.get('label_normalization')
        # Older min-max runs normalized NeRF2 as well as MLP and WESN.
        normalized_methods = summary.get('normalized_methods',
                                        methods + [b for b in baselines if b in ('B2_mlp', 'B3_nerf2')]
                                        if normalization else [])
        scope = (tuple(sorted(run_scenes)), tuple(sorted(methods)), tuple(sorted(baselines)),
                 mode, normalization, tuple(sorted(normalized_methods)))
        if reference is not None and scope != reference:
            raise ValueError('Seeds must use the same scenes, methods, WESN mode and label normalization')
        reference = scope
        for group, selected_methods, filename in (
                ('main_experiments', methods, 'main_per_scene.csv'),
                ('baselines', baselines, 'baseline_per_scene.csv')):
            rows = _read_rows(run_dir / filename, run_scenes, selected_methods)
            rows = [row for row in rows if row['scene_id'] in selected_scenes]
            per_scene_seed.extend(dict(seed=seed, group=group, scene_id=row['scene_id'],
                                       building_id=row['building_id'], approach=row['approach'],
                                       **{field: row[field] for field in METRIC_FIELDS})
                                  for row in rows)
            for method in selected_methods:
                selected = [row for row in rows if row['approach'] == method]
                per_seed.append(dict(seed=seed, group=group, approach=method,
                                     n_scenes=len(selected),
                                     **{field: float(np.mean([row[field] for row in selected]))
                                        for field in METRIC_FIELDS}))
    report = dict(seeds=seeds, n_seeds=len(seeds), n_scenes_per_seed=len(selected_scenes),
                  selected_scenes=selected_scenes,
                  wesn_prediction_mode=mode, evaluation_limited=summary['evaluation_limited'],
                  label_normalization=normalization,
                  normalized_methods=normalized_methods,
                  aggregation='equal scene means within seeds; mean and sample std across seeds',
                  std_ddof=1 if len(seeds) > 1 else 0, main_experiments={}, baselines={},
                  per_scene={})
    for group, method in sorted({(row['group'], row['approach']) for row in per_seed}):
        selected = [row for row in per_seed if row['group'] == group and row['approach'] == method]
        report[group][method] = {
            field: _statistics([row[field] for row in selected]) for field in METRIC_FIELDS}
    scene_rows = []
    for scene, group, method in sorted({(row['scene_id'], row['group'], row['approach'])
                                       for row in per_scene_seed}):
        selected = [row for row in per_scene_seed
                    if (row['scene_id'], row['group'], row['approach']) == (scene, group, method)]
        statistics = {field: _statistics([row[field] for row in selected]) for field in METRIC_FIELDS}
        report['per_scene'].setdefault(scene, {'main_experiments': {}, 'baselines': {}})[group][method] = statistics
        row = dict(scene_id=scene, building_id=selected[0]['building_id'], group=group,
                   approach=method, n_seeds=len(selected))
        for field, values in statistics.items():
            row[field + '_mean'], row[field + '_std'] = values['mean'], values['std']
        scene_rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / 'per_seed.csv', per_seed)
    _write_csv(out / 'per_scene_seed.csv', per_scene_seed)
    _write_csv(out / 'per_scene_summary.csv', scene_rows)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(f'Aggregated {len(seeds)} seeds, {len(selected_scenes)} scenes each: {out}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=9, help='Inclusive: 0..9 is ten seeds')
    parser.add_argument('--output-dir')
    parser.add_argument('--scenes', nargs='+', help='Summarize only these scenes; default: all completed scenes')
    args = parser.parse_args()
    aggregate(args.root, range(args.seed_start, args.seed_end + 1), args.output_dir, args.scenes)


if __name__ == '__main__':
    main()
