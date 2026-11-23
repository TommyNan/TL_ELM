"""Summarize target-label budget variants over scenes and seeds."""
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
from scripts.aggregate_seeds import _statistics, _write_csv


PROTOCOL_FIELDS = (
    'train_pool_fraction', 'wesn_prediction_mode', 'source_steps', 'target_steps',
    'finetune_lr', 'source_frequency_ghz', 'target_frequency_ghz',
    'dataset_sha256', 'source_sha256', 'experiment_sha256',
)


def _read_run(run_dir, seed):
    summary = json.loads((run_dir / 'summary.json').read_text())
    config = yaml.safe_load((run_dir / 'config.yml').read_text())
    if not summary['complete'] or summary['experiment'] != 'finetune_budget':
        raise ValueError(f'{run_dir}: a completed fine-tuning budget run is required')
    with (run_dir / 'finetune_per_scene.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['seed'] = int(row['seed'])
        row['finetune_fraction'] = float(row['finetune_fraction'])
        for field in METRIC_FIELDS:
            row[field] = float(row[field])
    actual = [(row['scene_id'], row['finetune_fraction']) for row in rows]
    expected = {(scene, fraction) for scene in summary['selected_scenes']
                for fraction in summary['finetune_fractions']}
    if (len(expected) != len(summary['selected_scenes']) * len(summary['finetune_fractions'])
            or len(actual) != len(expected) or set(actual) != expected
            or summary['seed'] != seed or any(row['seed'] != seed for row in rows)):
        raise ValueError(f'{run_dir}: incomplete, repeated or incorrectly identified scene/budget results')
    if not all(np.isfinite(row[field]) for row in rows for field in METRIC_FIELDS):
        raise ValueError(f'{run_dir}: metrics must be finite')
    # Optimizer, model and evaluation settings must match across repetitions.
    training = {key: value for key, value in config['train'].items() if key != 'seed'}
    summary['evaluation_limited'] = bool(training.get('max_eval_points', 0))
    scope = (tuple(sorted(summary['selected_scenes'])),
             tuple(sorted(summary['finetune_fractions'])),
             tuple(summary[field] for field in PROTOCOL_FIELDS),
             config['model'], training, config.get('source_training'),
             bool(summary.get('reuse_endpoints', False)))
    return summary, rows, scope


def _summary_rows(rows, identity):
    result = dict(identity)
    for field in METRIC_FIELDS:
        values = _statistics([row[field] for row in rows])
        result[field + '_mean'] = values['mean']
        result[field + '_std'] = values['std']
    return result


def _plot(rows, summary, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row['finetune_fraction'])
    x = np.asarray([100 * row['finetune_fraction'] for row in ordered])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, metric, color in zip(axes, ('rmse_db', 'mae_db'), ('#238A8D', '#D38A38')):
        ax.errorbar(x, [row[metric + '_mean'] for row in ordered],
                    yerr=[row[metric + '_std'] for row in ordered],
                    marker='o', linewidth=1.8, capsize=4, color=color)
        ax.set_xticks(x, [f'{value:g}' for value in x])
        ax.set_xlabel('Target labels (% of valid receiver locations)')
        ax.set_ylabel(metric.split('_')[0].upper() + ' (dB)')
        ax.grid(alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    mode = summary['wesn_prediction_mode'].replace('_', ' ')
    test_percent = 100 * (1 - summary['train_pool_fraction'])
    fig.suptitle(f'NeRF-WESN: {mode} | fixed {test_percent:g}% test split')
    evaluation_note = ' Test subsampling enabled.' if summary['evaluation_limited'] else ''
    fig.text(.5, .015,
             f"{summary['n_scenes_per_seed']} scenes, {summary['n_seeds']} seeds; "
             '0% = zero-shot. Error bars: across-seed SD of scene means.' + evaluation_note,
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'rmse_mae_vs_finetune_fraction.{extension}', dpi=220)
    plt.close(fig)


def aggregate(root, seeds=range(10), output_dir=None, scenes=None):
    """Average scenes within seeds, then summarize the seed means and each scene."""
    root, seeds = Path(root), list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('Choose distinct nonnegative seeds')
    out = Path(output_dir) if output_dir is not None else root / 'seed_summary'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is nonempty: {out}')
    all_rows, reference = [], None
    for seed in seeds:
        summary, rows, scope = _read_run(root / f'seed_{seed}', seed)
        if reference is not None and scope != reference:
            raise ValueError('Seeds must use matching scenes, budgets, data, model and training settings')
        reference = scope
        selected_scenes = list(summary['selected_scenes'] if scenes is None else scenes)
        if (not selected_scenes or len(set(selected_scenes)) != len(selected_scenes)
                or not set(selected_scenes).issubset(summary['selected_scenes'])):
            raise ValueError(f"Choose distinct scenes from {summary['selected_scenes']}")
        all_rows.extend(row for row in rows if row['scene_id'] in selected_scenes)

    fractions = sorted(summary['finetune_fractions'])
    approaches = {}
    for fraction in fractions:
        names = {row['approach'] for row in all_rows if row['finetune_fraction'] == fraction}
        if len(names) != 1:
            raise ValueError('Each fine-tuning fraction must have a consistent variant name')
        approaches[fraction] = names.pop()

    per_seed = []
    for seed in seeds:
        for fraction in fractions:
            selected = [row for row in all_rows
                        if row['seed'] == seed and row['finetune_fraction'] == fraction]
            per_seed.append(dict(seed=seed, approach=approaches[fraction],
                                 finetune_fraction=fraction, n_scenes=len(selected),
                                 **{field: float(np.mean([row[field] for row in selected]))
                                    for field in METRIC_FIELDS}))

    budget_rows = [_summary_rows(
        [row for row in per_seed if row['finetune_fraction'] == fraction],
        dict(approach=approaches[fraction], finetune_fraction=fraction,
             n_scenes=len(selected_scenes), n_seeds=len(seeds))) for fraction in fractions]
    scene_rows = []
    for scene in sorted(selected_scenes):
        for fraction in fractions:
            selected = [row for row in all_rows
                        if row['scene_id'] == scene and row['finetune_fraction'] == fraction]
            if len({row['building_id'] for row in selected}) != 1:
                raise ValueError(f'Inconsistent building identity for {scene}')
            scene_rows.append(_summary_rows(selected, dict(
                scene_id=scene, building_id=selected[0]['building_id'],
                approach=approaches[fraction], finetune_fraction=fraction, n_seeds=len(seeds))))

    def metric_statistics(row):
        return {field: dict(mean=row[field + '_mean'], std=row[field + '_std'])
                for field in METRIC_FIELDS}

    report = dict(experiment='finetune_budget', complete=True, seeds=seeds, n_seeds=len(seeds),
                  selected_scenes=selected_scenes, n_scenes_per_seed=len(selected_scenes),
                  finetune_fractions=fractions,
                  evaluation_limited=summary['evaluation_limited'],
                  source_training=summary.get('source_training'),
                  reuse_endpoints=bool(summary.get('reuse_endpoints', False)),
                  **{field: summary[field] for field in PROTOCOL_FIELDS},
                  std_ddof=1 if len(seeds) > 1 else 0,
                  aggregation='equal scene means within seeds; mean and sample std across seeds',
                  variants={row['approach']: dict(finetune_fraction=row['finetune_fraction'],
                                                 **metric_statistics(row)) for row in budget_rows},
                  per_scene={scene: {} for scene in selected_scenes})
    for row in scene_rows:
        report['per_scene'][row['scene_id']][row['approach']] = dict(
            finetune_fraction=row['finetune_fraction'], **metric_statistics(row))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('per_seed', per_seed), ('per_scene_seed', all_rows),
                       ('per_scene_summary', scene_rows), ('budget_summary', budget_rows)):
        _write_csv(out / f'{name}.csv', rows)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    _plot(budget_rows, report, out)
    print(f'Aggregated {len(seeds)} seeds, {len(selected_scenes)} scenes, {len(fractions)} budgets: {out}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=9, help='Inclusive')
    parser.add_argument('--output-dir')
    parser.add_argument('--scenes', nargs='+', help='Summarize only these completed scenes')
    args = parser.parse_args()
    aggregate(args.root, range(args.seed_start, args.seed_end + 1), args.output_dir, args.scenes)


if __name__ == '__main__':
    main()
