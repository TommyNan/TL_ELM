#!/usr/bin/env python3
"""Plot ground truth, sparse f3 labels, Scratch and Transfer in a 2x2 figure.

Examples, from the project root:
  python scripts/visualize_wesn_comparison.py \
    --root runs/wesn-residual-10scenarios-10000steps --seed 5 --scene Scene01
  python scripts/visualize_wesn_comparison.py \
    --root runs/wesn-residual-10scenarios-10000steps --prediction-source test

Checkpoint mode predicts every valid pixel without training. Test mode reads
the existing test_predictions.npz files and leaves other pixels blank. Figure
scores always describe the original saved held-out predictions, not training
pixels or the whole dense image. No measured labels are substituted into a
model prediction map.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tl-elm-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / 'Models').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.model import RadioMapField
from datasets.data_loader import load_scene
from datasets.normalization import BandMinMax
from experiment_utils import choose_device, metrics
from propagation import numpy_fspl_db


METHODS = ('E2_scratch', 'E3_transfer')
DEFAULT_ROOT = 'runs/wesn-residual-10scenarios-10000steps'
RECOMMENDED_SEED, RECOMMENDED_SCENE = 5, 'Scene01'


def select_case(csv_path, seed=None, scene=None, criterion='transfer_rmse'):
    """Prefer improvement in both metrics, then lowest Transfer RMSE or gain.

    This selects an illustrative case using held-out performance, and is not
    a substitute for the complete across-scene/across-seed experiment report.
    """
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get('group', 'main_experiments') == 'main_experiments'
                and row['approach'] in METHODS]
    pairs = {}
    for row in rows:
        key = (int(row['seed']), row['scene_id'])
        if (seed is not None and key[0] != seed) or (scene is not None and key[1] != scene):
            continue
        group = pairs.setdefault(key, {})
        if row['approach'] in group:
            raise ValueError(f'Duplicate seed/scene/method in {csv_path}: {key}')
        group[row['approach']] = row
    candidates = []
    for (current_seed, scene_id), pair in pairs.items():
        if set(pair) != set(METHODS):
            raise ValueError(f'Missing Scratch or Transfer for seed {current_seed}, {scene_id}')
        scratch, transfer = pair['E2_scratch'], pair['E3_transfer']
        sr, sm = float(scratch['rmse_db']), float(scratch['mae_db'])
        tr, tm = float(transfer['rmse_db']), float(transfer['mae_db'])
        candidates.append(dict(
            seed=current_seed, scene_id=scene_id, building_id=transfer['building_id'],
            scratch_rmse_db=sr, scratch_mae_db=sm, transfer_rmse_db=tr, transfer_mae_db=tm,
            rmse_reduction_percent=100 * (sr - tr) / sr if sr else None,
            mae_reduction_percent=100 * (sm - tm) / sm if sm else None,
            both_metrics_better=tr < sr and tm < sm))
    if not candidates:
        raise ValueError('No matched seed/scene results satisfy the requested selection')
    if criterion not in ('transfer_rmse', 'rmse_gain'):
        raise ValueError('Select by transfer_rmse or rmse_gain')

    def rank(row):
        score = row['transfer_rmse_db']
        if criterion == 'rmse_gain':
            gain = row['rmse_reduction_percent']
            score = -gain if gain is not None else float('inf')
        return (not row['both_metrics_better'], score, row['transfer_mae_db'],
                row['seed'], row['scene_id'])

    chosen = dict(min(candidates, key=rank), criterion=criterion,
                  selection_source=str(Path(csv_path).resolve()), n_candidates=len(candidates))
    chosen['selection_note'] = 'Post-hoc illustrative case; report full scene/seed statistics separately.'
    return chosen


@torch.no_grad()
def checkpoint_prediction(checkpoint_path, scene, config, method, device, eval_batch_size=None):
    """Load this method only and predict the entire map without optimizer steps."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    params = checkpoint['model_params']
    mode = params.get('prediction_mode', 'fspl_residual')
    expected_mode = config['model'].get('prediction_mode', 'fspl_residual')
    if (checkpoint['scene_id'] != scene.scene_id or checkpoint['method'] != method
            or checkpoint['model_type'] != 'radio_map'
            or checkpoint['config']['train']['seed'] != config['train']['seed']
            or mode != expected_mode
            or not np.allclose([params['width_m'], params['height_m']],
                               [scene.width_m, scene.height_m])
            or checkpoint['config']['data']['target_frequency_ghz'] != scene.target_frequency_ghz):
        raise ValueError(f'{checkpoint_path}: checkpoint does not match the selected scene, seed or mode')
    selected_device = choose_device(device)
    torch.set_num_threads(config['train'].get('num_threads', 2))
    model = RadioMapField(params).to(selected_device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    # Match the existing checkpoint exporter if an older normalized model is used.
    scale = BandMinMax(params.get('label_normalization'))
    residual_scaling = scale.enabled and mode == 'fspl_residual'
    predictor = model.predict_residual_db if residual_scaling else model.predict_path_loss_db
    indices = np.flatnonzero(scene.valid_mask)
    result = np.full(len(scene.rx_m), np.nan, dtype=np.float32)
    batch_size = eval_batch_size or config['train'].get('eval_batch_size', 256)
    tx = torch.as_tensor(scene.tx_m, dtype=torch.float32, device=selected_device)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        rx = torch.as_tensor(scene.rx_m[batch], dtype=torch.float32, device=selected_device)
        values = predictor(rx, tx.expand(len(batch), -1), scene.target_frequency_ghz).cpu().numpy()
        if residual_scaling:
            prior = numpy_fspl_db(scene.rx_m[batch], scene.tx_m, scene.target_frequency_ghz)
            values = prior + scale.denormalize_residual(values, scene.target_frequency_ghz)
        elif scale.enabled:
            values = scale.denormalize(values, scene.target_frequency_ghz)
        result[batch] = values
    return result


def collect_panels(run_root, seed, scene_id, prediction_source='checkpoint', device='auto',
                   dataset_root=None, eval_batch_size=None):
    """Return exactly four maps and their saved held-out test scores."""
    if prediction_source not in ('checkpoint', 'test'):
        raise ValueError('prediction_source must be checkpoint or test')
    run = Path(run_root).resolve() / f'seed_{seed}'
    config = yaml.safe_load((run / 'config.yml').read_text())
    if config['train']['seed'] != seed:
        raise ValueError('The saved config seed differs from the selected seed folder')
    data = config['data']
    root = Path(dataset_root) if dataset_root is not None else Path(data['root'])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    scene = load_scene(root, scene_id, data['source_band'], data['target_band'],
                       band_frequencies_ghz={data['source_band']: data['source_frequency_ghz'],
                                            data['target_band']: data['target_frequency_ghz']})
    folder = run / scene_id
    with np.load(folder / 'split.npz', allow_pickle=False) as split:
        train, test = split['train'].copy(), split['test'].copy()
    valid = np.asarray(scene.valid_mask, dtype=bool).reshape(-1)
    truth = np.asarray(scene.target_pl_db, dtype=np.float64).reshape(-1).copy()
    truth[~valid] = np.nan
    sparse = np.full(truth.shape, np.nan)
    sparse[train] = truth[train]
    fraction = len(train) / int(valid.sum())
    panels = [dict(method='target', title='(a) Ground truth', values=truth),
              dict(method='sparse', title=f'(b) Sparse measurements ({100 * fraction:.1f}%)', values=sparse)]
    saved, test_metrics, previous = {}, {}, None
    # Validate pairing before running either model's dense inference.
    for method in METHODS:
        path = folder / method / 'test_predictions.npz'
        with np.load(path, allow_pickle=False) as arrays:
            indices = arrays['indices'].copy()
            values, targets = arrays['prediction_db'].copy(), arrays['target_db'].copy()
        if (not np.isin(indices, test).all() or np.intersect1d(indices, train).size
                or not np.array_equal(targets, scene.target_pl_db[indices])
                or (previous is not None and not np.array_equal(indices, previous))):
            raise ValueError(f'{path}: test indices or labels do not match the selected scene and split')
        previous = indices
        saved[method] = (indices, values)
        test_metrics[method] = metrics(values, targets)
    for method, letter, name in zip(METHODS, ('c', 'd'), ('Scratch', 'Transfer')):
        indices, saved_values = saved[method]
        if prediction_source == 'checkpoint':
            values = checkpoint_prediction(folder / method / 'final.pt', scene, config,
                                           method, device, eval_batch_size)
            suffix = ''
        else:
            values = np.full(truth.shape, np.nan)
            values[indices] = saved_values
            suffix = ' — test points only'
        values = np.asarray(values, dtype=np.float64).reshape(-1).copy()
        values[~valid] = np.nan
        score = test_metrics[method]
        panels.append(dict(method=method, values=values,
                           title=f'({letter}) NeRF-WESN {name}{suffix}\n'
                                 f'Test RMSE {score["rmse_db"]:.3f} dB | MAE {score["mae_db"]:.3f} dB',
                           rmse_db=score['rmse_db'], mae_db=score['mae_db']))
    metadata = dict(seed=seed, scene_id=scene_id, building_id=scene.building_id,
                    run_dir=str(run), dataset_root=str(root.resolve()),
                    prediction_source=prediction_source,
                    wesn_prediction_mode=config['model'].get('prediction_mode', 'fspl_residual'),
                    target_frequency_ghz=scene.target_frequency_ghz,
                    target_steps=config['train'].get('target_steps'),
                    width=scene.width, height=scene.height, tx_m=scene.tx_m.tolist(),
                    n_valid=int(valid.sum()), n_training_measurements=len(train),
                    measurement_fraction=fraction, n_test_total=len(test),
                    evaluation_limited=len(previous) < len(test), test_metrics=test_metrics,
                    metric_source='Saved test_predictions.npz; held-out locations only',
                    prediction_files={method: str(folder / method /
                                      ('final.pt' if prediction_source == 'checkpoint' else 'test_predictions.npz'))
                                      for method in METHODS})
    return scene, panels, metadata


def render_figure(scene, panels, output_path, metadata=None, *, vmin=None, vmax=None,
                  dpi=300, cmap='viridis'):
    """Render one shared-scale 2x2 figure; save a JSON sidecar with provenance."""
    metadata = dict(metadata or {})
    finite = np.concatenate([np.asarray(panel['values']).reshape(-1)[
        np.isfinite(np.asarray(panel['values']).reshape(-1))] for panel in panels])
    low, high = float(finite.min() if vmin is None else vmin), float(finite.max() if vmax is None else vmax)
    if low == high:
        high = low + 1
    if low > high or len(panels) != 4:
        raise ValueError('Provide four panels and an increasing common color range')
    norm = Normalize(vmin=low, vmax=high)
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad((1., 1., 1., 0.))
    extent = (-scene.pixel_size_m / 2, scene.width_m + scene.pixel_size_m / 2,
              scene.height_m + scene.pixel_size_m / 2, -scene.pixel_size_m / 2)
    background = np.ones((scene.height, scene.width, 4))
    background[~np.asarray(scene.valid_mask).reshape(scene.height, scene.width)] = (.75, .75, .75, 1.)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.2), constrained_layout=True)
    for ax, panel in zip(axes.flat, panels):
        ax.imshow(background, origin='upper', extent=extent, interpolation='nearest')
        values = np.asarray(panel['values']).reshape(scene.height, scene.width)
        picture = ax.imshow(np.ma.masked_invalid(values), origin='upper', extent=extent,
                            interpolation='nearest', cmap=colors, norm=norm)
        ax.scatter(scene.tx_m[0], scene.tx_m[1], marker='*', s=90, c='#E53935',
                   edgecolors='black', linewidths=.5, label='Tx', zorder=4)
        ax.set(title=panel['title'], xlabel='x (m)', ylabel='y (m)', aspect='equal')
    axes[0, 0].legend(loc='upper right', fontsize=8, framealpha=.85)
    fig.colorbar(picture, ax=list(axes.flat), label='Path loss (dB)', shrink=.86)
    seed = metadata.get('seed', '?')
    frequency = metadata.get('target_frequency_ghz', scene.target_frequency_ghz)
    fig.suptitle(f'{scene.scene_id} | seed {seed} | Target frequency {frequency:g} GHz', fontsize=15)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
    finally:
        plt.close(fig)
    report = dict(metadata, layout_rows=2, layout_columns=2, vmin_db=low, vmax_db=high,
                  cmap=cmap, dpi=dpi, coordinate_frame='local raster; y increases with image row',
                  excluded_receiver_color='gray', unmeasured_or_unpredicted_color='white',
                  panels=[{key: value for key, value in panel.items() if key != 'values'} for panel in panels])
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', default=DEFAULT_ROOT, help='Directory containing seed_N folders')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--scene', '--scenario', dest='scene')
    parser.add_argument('--results-csv', help='Per-scene/per-seed CSV for automatic case selection')
    parser.add_argument('--select-by', choices=('transfer_rmse', 'rmse_gain'), default='transfer_rmse')
    parser.add_argument('--prediction-source', choices=('checkpoint', 'test'), default='checkpoint')
    parser.add_argument('--dataset-root', help='Override the dataset path when moving saved runs')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--output', help='PNG, PDF or SVG filename; default: ROOT/figures/seed_N_SceneXX_SOURCE.png')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('--cmap', default='viridis')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)
    root = Path(args.root)
    csv_path = Path(args.results_csv) if args.results_csv else root / 'seed_summary/per_scene_seed.csv'
    if args.seed is not None and args.scene is not None:
        seed, scene_id = args.seed, args.scene
        selection = dict(criterion='explicit seed and scene', seed=seed, scene_id=scene_id)
    elif csv_path.exists():
        selection = select_case(csv_path, args.seed, args.scene, args.select_by)
        seed, scene_id = selection['seed'], selection['scene_id']
    else:
        if args.results_csv:
            raise FileNotFoundError(csv_path)
        seed = RECOMMENDED_SEED if args.seed is None else args.seed
        scene_id = args.scene or RECOMMENDED_SCENE
        selection = dict(criterion='recommended case or explicit override', seed=seed, scene_id=scene_id,
                         selection_note='Default seed 5 / Scene01 was selected from the supplied 100-case CSV.')
    output = args.output or root / f'figures/seed_{seed}_{scene_id}_{args.prediction_source}.png'
    print(f'Plotting seed {seed}, {scene_id}; selection: {selection["criterion"]}', flush=True)
    scene, panels, metadata = collect_panels(root, seed, scene_id, args.prediction_source,
                                            args.device, args.dataset_root, args.eval_batch_size)
    metadata['case_selection'] = selection
    render_figure(scene, panels, output, metadata, vmin=args.vmin, vmax=args.vmax,
                  dpi=args.dpi, cmap=args.cmap)
    print(f'Saved {output} and {output}.json')


if __name__ == '__main__':
    main()
