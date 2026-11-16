"""Compare ICASSP FSPL, KNN, MLP and NeRF2 on matched held-out receivers.

Use sweep_baselines.sh for seeds 0..9, or --aggregate-only to summarize completed repetitions.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from baselines.common import (aggregate_baseline_sweep, normalize_methods,
                              save_baseline_summary)
from main import run


def run_comparison(config, scenes=None, methods=None, *, seed=None, output_dir=None):
    """Run only selected baselines, preserving the main runner's data protocol."""
    selected = normalize_methods(methods)
    configuration = copy.deepcopy(config)
    if seed is not None:
        configuration['train']['seed'] = seed
    if output_dir is not None:
        configuration['output_dir'] = str(output_dir)
    report = run(configuration, scenes, methods=[], baselines=selected)
    save_baseline_summary(configuration['output_dir'], report)
    return report


def update_config(section, **overrides):
    """Apply supplied CLI options, keeping all other YAML settings."""
    section.update({key: value for key, value in overrides.items() if value is not None})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', help='YAML configuration; default: configs/icassp.yml')
    parser.add_argument('--scenes', '--scene', nargs='+', help='Default: all 24 paired scenes')
    parser.add_argument('--methods', '--baselines', nargs='+',
                        help='Default: fspl knn mlp nerf2; canonical experiment IDs are also accepted')
    parser.add_argument('--seed', type=int, help='Single-run seed; default: 0')
    parser.add_argument('--output-dir', '--output_dir',
                        help='New run directory, or new summary directory in aggregate-only mode')
    parser.add_argument('--train-fraction', '--train_fraction', '--train-ratio', '--train_ratio',
                        type=float, help='Default from config: 0.1; no validation split')
    parser.add_argument('--dataset-root', '--dataset_root', help='Paired ICASSP directory containing the CSV files')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--target-steps', '--target_steps', '--steps', type=int,
                        help='Fixed optimizer steps per neural baseline, not epochs')
    parser.add_argument('--batch-size', '--batch_size', type=int)
    parser.add_argument('--eval-batch-size', '--eval_batch_size', type=int)
    parser.add_argument('--lr', type=float, help='Target-band Adam learning rate')
    parser.add_argument('--weight-decay', '--weight_decay', type=float)
    parser.add_argument('--num-threads', '--num_threads', type=int)
    parser.add_argument('--max-eval-points', '--max_eval_points', type=int,
                        help='0: complete test set; positive values are only for short checks')
    parser.add_argument('--k', type=int, help='KNN neighbor count; default from config: 5')
    parser.add_argument('--aggregate-only', '--aggregate_only', action='store_true',
                        help='Summarize completed seed_N runs without training')
    parser.add_argument('--root', help='Directory containing seed_0, seed_1, ...')
    parser.add_argument('--seed-start', '--seed_start', type=int, help='First seed to aggregate; default: 0')
    parser.add_argument('--seed-end', '--seed_end', type=int, help='Last seed, inclusive; default: 9')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.aggregate_only:
        if not args.root:
            parser.error('--aggregate-only requires --root')
        training_options = ('config', 'scenes', 'methods', 'seed', 'train_fraction', 'dataset_root',
                            'device', 'target_steps', 'batch_size', 'eval_batch_size', 'lr',
                            'weight_decay', 'num_threads', 'max_eval_points', 'k')
        if any(getattr(args, name) is not None for name in training_options):
            parser.error('Training options cannot be used with --aggregate-only')
        first = 0 if args.seed_start is None else args.seed_start
        last = 9 if args.seed_end is None else args.seed_end
        if first < 0 or last < first:
            parser.error('Choose a nonnegative seed range with seed-end >= seed-start')
        try:
            return aggregate_baseline_sweep(args.root, range(first, last + 1), args.output_dir)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    if args.root is not None or args.seed_start is not None or args.seed_end is not None:
        parser.error('--root and seed ranges belong to --aggregate-only; use --seed for one run')
    try:
        selected = normalize_methods(args.methods)
        config_path = Path(args.config) if args.config else PROJECT_ROOT / 'configs/icassp.yml'
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        update_config(config['train'],
                      seed=0 if args.seed is None else args.seed,
                      device=args.device, target_steps=args.target_steps,
                      batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                      target_lr=args.lr, weight_decay=args.weight_decay,
                      num_threads=args.num_threads, max_eval_points=args.max_eval_points)
        update_config(config['data'], train_fraction=args.train_fraction, root=args.dataset_root)
        update_config(config['baseline']['knn'], k=args.k)
        config['output_dir'] = (args.output_dir if args.output_dir is not None else
                                str(PROJECT_ROOT / 'runs/baseline-comparison-minmax' /
                                    f'seed_{config["train"]["seed"]}'))
        print(f'Baselines only: {", ".join(selected)}; seed={config["train"]["seed"]}', flush=True)
        report = run_comparison(config, args.scenes, selected)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({'seed': report['seed'], 'n_scenes': len(report['selected_scenes']),
                      'baselines': report['baselines'], 'output_dir': config['output_dir']}, indent=2))
    return report


if __name__ == '__main__':
    main()
