"""Vary target-band labels within a fixed training pool and a fixed test set."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.data_loader import discover_scenes
from experiment_utils import (METRIC_FIELDS, aggregate_results, choose_device,
                              file_fingerprint, metrics, reproducibility_metadata,
                              scene_seed, sync_if_cuda)
from main import (copy_state, fit_field, make_train_test_split, new_model,
                  predict_field, record_prediction, save_checkpoint,
                  scene_from_config, state_digest, validate_config)
from scripts.aggregate_seeds import _write_csv


DEFAULT_FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10)
SOURCE_TRAIN_KEYS = ('pretrain_steps', 'pretrain_lr', 'weight_decay', 'gradient_clip_norm', 'batch_size')


def variant_name(fraction):
    return f'FT_{100 * fraction:.12g}pct'


def _fractions(values, train_fraction):
    values = tuple(sorted(map(float, values)))
    if (not values or len(set(values)) != len(values)
            or any(not math.isfinite(f) or not 0 <= f <= train_fraction for f in values)):
        raise ValueError(f'Choose distinct fractions between 0 and {train_fraction}, inclusive')
    return values


def make_finetune_splits(valid_mask, train_fraction, fractions, seed):
    """Nested random prefixes; fractions refer to all valid scene locations.

    The original train/test split is unchanged. Unused target training-pool
    locations stay outside the test set, even for the zero-label variant.
    """
    fractions = _fractions(fractions, train_fraction)
    split = make_train_test_split(valid_mask, train_fraction, seed)
    valid = np.flatnonzero(np.asarray(valid_mask).reshape(-1))
    order = np.random.default_rng(seed).permutation(valid)
    budgets = {f: np.sort(order[:math.floor(len(valid) * f)]) for f in fractions}
    if any(f > 0 and len(indices) == 0 for f, indices in budgets.items()):
        raise ValueError('A positive fine-tuning fraction must select at least one location')
    return split, budgets


def _source_run(config, source_root, dataset_root, device):
    """Locate and check a completed source run, without modifying that run."""
    if source_root is None:
        return None
    root = Path(source_root) / f'seed_{config["train"]["seed"]}'
    saved = yaml.safe_load((root / 'config.yml').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    data = config['data']
    keys = ('source_band', 'target_band', 'source_frequency_ghz',
            'target_frequency_ghz', 'train_fraction')
    if (not summary['complete'] or summary['seed'] != config['train']['seed']
            or any(saved['data'][key] != data[key] for key in keys)):
        raise ValueError('Source run must be complete and use the same seed, bands and training pool')
    current = reproducibility_metadata(PROJECT_ROOT, dataset_root, summary['selected_scenes'],
                                       data['source_band'], data['target_band'], device)
    if current['dataset_sha256'] != summary['dataset_sha256']:
        raise ValueError('Source run and current dataset differ; use matching data or pretrain a new source')
    # When reusing weights, their actual pretraining budget comes from the run.
    config['train']['pretrain_steps'] = saved['train']['pretrain_steps']
    config['source_training'] = {key: saved['train'][key] for key in SOURCE_TRAIN_KEYS}
    return root


def _prepare_source(scene, split, params, config, folder, device, seed, provenance, source_run):
    training, data = config['train'], config['data']
    model = new_model(params, seed, device)
    initial_digest = state_digest(model.state_dict())
    destination = folder / 'source_pretrain'
    destination.mkdir(parents=True)
    if source_run is not None:
        origin = source_run / scene.scene_id
        checkpoint_path = origin / 'source_pretrain/final.pt'
        if not checkpoint_path.exists():
            checkpoint_path = origin / 'E4_zero_shot/final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if (checkpoint['scene_id'] != scene.scene_id
                or checkpoint['method'] not in ('source_pretrain', 'E4_zero_shot')
                or checkpoint['model_type'] != 'radio_map' or checkpoint['model_params'] != params):
            raise ValueError('Source checkpoint architecture or WESN mode differs from the requested model')
        if checkpoint['method'] == 'E4_zero_shot' and (
                checkpoint['metrics']['target_steps'] != 0
                or checkpoint['metrics']['f3_label_count'] != 0
                or not (checkpoint['state_sha256'] == checkpoint['metrics']['source_state_sha256']
                        == state_digest(checkpoint['state_dict']))):
            raise ValueError('E4 must contain the unchanged source model with zero target training steps')
        with np.load(origin / 'split.npz', allow_pickle=False) as saved:
            if not all(np.array_equal(saved[key], split[key]) for key in ('train', 'test')):
                raise ValueError('Source checkpoint and fine-tuning experiment use different receiver splits')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        seconds = float(checkpoint['metrics']['train_time_sec'])
        shutil.copy2(checkpoint_path, destination / 'final.pt')
        history = checkpoint_path.with_name('history.json')
        if history.exists():
            shutil.copy2(history, destination / 'history.json')
        print(f'{scene.scene_id}: reusing source {checkpoint_path}', flush=True)
    else:
        indices = split['train']
        history, seconds = fit_field(
            model, scene.rx_m[indices], scene.tx_m, data['source_frequency_ghz'],
            scene.source_pl_db[indices], training, training['pretrain_steps'],
            training['pretrain_lr'], seed + 17, stage=f'{scene.scene_id}/source_pretrain')
        record = dict(train_time_sec=seconds, f2_label_count=len(indices), f3_label_count=0,
                      source_steps=training['pretrain_steps'])
        save_checkpoint(destination / 'final.pt', model, params, config, scene.scene_id,
                        'source_pretrain', initial_digest, record, provenance=provenance)
        (destination / 'history.json').write_text(json.dumps(history, indent=2))
    state = copy_state(model)
    return state, state_digest(state), seconds


def _load_endpoint(source_run, scene, fraction, test, params, training, source_hash):
    """Read a compatible completed endpoint without training or inference."""
    method = 'E4_zero_shot' if fraction == 0 else 'E3_transfer'
    origin = source_run / scene.scene_id / method
    checkpoint = torch.load(origin / 'final.pt', map_location='cpu', weights_only=True)
    record = checkpoint['metrics']
    if (checkpoint['scene_id'] != scene.scene_id or checkpoint['method'] != method
            or checkpoint['model_type'] != 'radio_map' or checkpoint['model_params'] != params
            or record['source_state_sha256'] != source_hash
            or state_digest(checkpoint['state_dict']) != checkpoint['state_sha256']):
        raise ValueError(f'{method}: endpoint and fine-tuning variants must share the same source model')
    if fraction == 0:
        if (checkpoint['state_sha256'] != source_hash or record['target_steps'] != 0
                or record['f3_label_count'] != 0):
            raise ValueError('The reused 0% endpoint must be the unchanged zero-shot source model')
    else:
        saved_training = checkpoint['config']['train']
        keys = ('finetune_lr', 'weight_decay', 'gradient_clip_norm', 'batch_size')
        if (checkpoint['initial_state_sha256'] != source_hash
                or record['target_steps'] != training['target_steps']
                or any(saved_training[key] != training[key] for key in keys)):
            raise ValueError('Existing full-budget endpoint uses different fine-tuning settings; '
                             'match --target-steps and optimizer settings before reusing endpoints')
    with np.load(origin / 'test_predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['indices'], test):
            raise ValueError('Existing endpoint test indices differ; use the same evaluation settings')
        if not np.array_equal(saved['target_db'], scene.target_pl_db[test]):
            raise ValueError('Existing endpoint target labels differ from the current dataset')
        prediction = saved['prediction_db'].copy()
    history = origin / 'history.json'
    return dict(checkpoint=checkpoint, prediction=prediction, origin=origin,
                scores={field: float(record[field]) for field in METRIC_FIELDS},
                history=json.loads(history.read_text()) if history.exists() else [])


def _run_scene(scene, split, budgets, config, out, device, seed, provenance, source_run,
               reuse_endpoints):
    data, training = config['data'], config['train']
    folder = out / scene.scene_id
    folder.mkdir()
    test = split['test']
    limit = training['max_eval_points']
    if limit and len(test) > limit:
        test = np.sort(np.random.default_rng(seed + 1).choice(test, limit, replace=False))
    np.savez_compressed(folder / 'split.npz', **split, source_train=split['train'],
                        train_pool=split['train'], test_evaluation=test)
    mode = config['model'].get('prediction_mode', 'fspl_residual')
    params = dict(config['model'], width_m=scene.width_m, height_m=scene.height_m,
                  prediction_mode=mode)
    params.pop('label_normalization', None)
    source_state, source_hash, source_seconds = _prepare_source(
        scene, split, params, config, folder, device, seed, provenance, source_run)
    # Check both requested endpoints before spending time on any middle budget.
    endpoints = {fraction: _load_endpoint(source_run, scene, fraction, test, params, training, source_hash)
                 for fraction in budgets if reuse_endpoints and fraction in (0., data['train_fraction'])}

    rows = []
    for fraction, indices in budgets.items():
        approach = variant_name(fraction)
        model = new_model(params, seed, device)
        # Every budget starts from the same source, including its fixed reservoir.
        model.load_state_dict(source_state, strict=True)
        initial_hash = state_digest(model.state_dict())
        history, fine_seconds = [], 0.0
        target_steps = training['target_steps'] if len(indices) else 0
        reused = endpoints.get(fraction)
        if reused is not None:
            model.load_state_dict(reused['checkpoint']['state_dict'], strict=True)
            prediction, scores, history = reused['prediction'], reused['scores'], reused['history']
            print(f'{scene.scene_id}/{approach}: reusing {reused["origin"]}', flush=True)
        else:
            if target_steps:
                history, fine_seconds = fit_field(
                    model, scene.rx_m[indices], scene.tx_m, data['target_frequency_ghz'],
                    scene.target_pl_db[indices], training, target_steps,
                    training['finetune_lr'], seed + 17, stage=f'{scene.scene_id}/{approach}')
            sync_if_cuda(device)
            start = time.perf_counter()
            prediction = predict_field(model, scene.rx_m[test], scene.tx_m,
                                       data['target_frequency_ghz'], training['eval_batch_size'])
            sync_if_cuda(device)
            scores = dict(train_time_sec=source_seconds + fine_seconds,
                          test_time_sec=time.perf_counter() - start,
                          **metrics(prediction, scene.target_pl_db[test]))
        record = dict(
            scene_id=scene.scene_id, building_id=scene.building_id, approach=approach,
            seed=training['seed'], scene_seed=seed, finetune_fraction=fraction,
            n_train=len(indices), n_source_train=len(split['train']),
            n_unused_target_pool=len(split['train']) - len(indices),
            n_test_total=len(split['test']), n_test=len(test),
            f2_label_count=len(split['train']), f3_label_count=len(indices),
            source_steps=training['pretrain_steps'], target_steps=target_steps,
            source_state_sha256=source_hash, initial_state_sha256=initial_hash,
            state_sha256=state_digest(model.state_dict()), wesn_prediction_mode=mode,
            reused_from=str(reused['origin']) if reused is not None else '',
            n_points=len(test), **{field: scores[field] for field in METRIC_FIELDS})
        destination = folder / approach
        checkpoint_config = dict(config, finetune_fraction=fraction)
        save_checkpoint(destination / 'final.pt', model, params, checkpoint_config,
                        scene.scene_id, approach, initial_hash, record, provenance=provenance)
        np.savez_compressed(destination / 'split.npz', target_train=indices,
                            test=split['test'], test_evaluation=test)
        (destination / 'history.json').write_text(json.dumps(history, indent=2))
        record_prediction(destination, record, test, prediction, scene.target_pl_db[test])
        rows.append(record)
        print(f'{scene.scene_id} {approach}: labels={len(indices)}, '
              f'RMSE={record["rmse_db"]:.3f}, MAE={record["mae_db"]:.3f} dB', flush=True)
    return rows


def run(config, fractions=DEFAULT_FRACTIONS, scenes=None, source_root=None, reuse_endpoints=False):
    config = copy.deepcopy(config)
    validate_config(config)
    if reuse_endpoints and source_root is None:
        raise ValueError('--reuse-endpoints requires --source-root')
    fractions = _fractions(fractions, config['data']['train_fraction'])
    config['finetune_fractions'] = list(fractions)
    config['reuse_endpoints'] = bool(reuse_endpoints)
    config['label_normalization'] = None
    dataset_root = Path(config['data']['root'])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    available = discover_scenes(dataset_root)
    selected = list(available if scenes is None else scenes)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError('Choose distinct scene IDs from the dataset')
    training, data = config['train'], config['data']
    torch.set_num_threads(training['num_threads'])
    device = choose_device(training['device'])
    config['source_training'] = {key: training[key] for key in SOURCE_TRAIN_KEYS}
    source_run = _source_run(config, source_root, dataset_root, device)
    out = Path(config['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is nonempty; choose a new output_dir: {out}')
    out.mkdir(parents=True, exist_ok=True)
    provenance = reproducibility_metadata(PROJECT_ROOT, dataset_root, selected,
                                          data['source_band'], data['target_band'], device)
    provenance['experiment_sha256'] = file_fingerprint(PROJECT_ROOT, [Path(__file__)])
    (out / 'config.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    (out / 'runtime.json').write_text(json.dumps(provenance, indent=2))
    summary = dict(
        experiment='finetune_budget', complete=False, selected_scenes=selected,
        seed=training['seed'], train_pool_fraction=data['train_fraction'],
        finetune_fractions=list(fractions),
        wesn_prediction_mode=config['model'].get('prediction_mode', 'fspl_residual'),
        source_steps=training['pretrain_steps'], target_steps=training['target_steps'],
        finetune_lr=training['finetune_lr'],
        source_frequency_ghz=data['source_frequency_ghz'],
        target_frequency_ghz=data['target_frequency_ghz'],
        expected_runs=len(selected) * len(fractions), completed_runs=0,
        evaluation_limited=bool(training['max_eval_points']),
        source_root=str(source_root) if source_root is not None else None,
        reuse_endpoints=bool(reuse_endpoints),
        source_training=config['source_training'],
        inventory=[], variants={}, **{key: provenance[key] for key in
                                     ('source_sha256', 'dataset_sha256', 'experiment_sha256')})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    rows = []
    for scene_id in selected:
        scene = scene_from_config(config, scene_id)
        seed = scene_seed(training['seed'], scene_id)
        split, budgets = make_finetune_splits(scene.valid_mask, data['train_fraction'], fractions, seed)
        rows.extend(_run_scene(scene, split, budgets, config, out, device, seed, provenance, source_run,
                               reuse_endpoints))
        summary['inventory'].append(dict(scene_id=scene_id, building_id=scene.building_id,
                                         n_valid=int(scene.valid_mask.sum()),
                                         n_source_train=len(split['train']), n_test=len(split['test'])))
        summary['variants'] = aggregate_results(rows)
        summary['completed_runs'] = len(rows)
        summary['complete'] = len(rows) == summary['expected_runs']
        _write_csv(out / 'finetune_per_scene.csv', rows)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs/icassp.yml'))
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=('fspl_residual', 'direct_pl'))
    parser.add_argument('--fractions', nargs='+', type=float, default=DEFAULT_FRACTIONS,
                        help='Fractions of all valid scene locations; default: 0 .01 .02 .05 .10')
    parser.add_argument('--train-fraction', type=float,
                        help='Fixed source/training-pool fraction; default from YAML: 0.1')
    parser.add_argument('--pretrain-steps', type=int,
                        help='Steps for a new source; reused sources retain their actual saved budget')
    parser.add_argument('--target-steps', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--max-eval-points', type=int)
    parser.add_argument('--source-root', help='Completed main runs containing seed_N/source checkpoints')
    parser.add_argument('--reuse-endpoints', action='store_true',
                        help='Copy existing E4 zero-shot and E3 full-budget results without training or inference')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    config['train'].update({name: getattr(args, name) for name in
                            ('pretrain_steps', 'target_steps', 'device', 'max_eval_points')
                            if getattr(args, name) is not None})
    config['train']['seed'] = args.seed
    if args.mode is not None:
        config['model']['prediction_mode'] = args.mode
    if args.train_fraction is not None:
        config['data']['train_fraction'] = args.train_fraction
    config['output_dir'] = args.output_dir or f'runs/finetune-budget/seed_{args.seed}'
    run(config, args.fractions, args.scenes, args.source_root, args.reuse_endpoints)


if __name__ == '__main__':
    main()
