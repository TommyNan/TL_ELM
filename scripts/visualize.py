#!/usr/bin/env python3
"""Plot target PL, sparse measurements and saved-model predictions.

All panels share a PL (dB) color scale. Dense prediction maps include training
locations; the title reports the saved held-out test RMSE. No model is trained.
"""
from __future__ import annotations

import argparse
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.fspl import FSPL
from baselines.knn import KNN
from main import predict_checkpoint, scene_from_config


METHOD_LABELS = {
    'E1_fspl': 'FSPL',
    'B1_knn': 'KNN',
    'B2_mlp': 'MLP (direct PL)',
    'B3_nerf2': 'NeRF2-MLP (direct PL)',
    'E2_scratch': 'FSPL + WESN scratch',
    'E3_transfer': 'FSPL + WESN transfer',
    'E4_zero_shot': 'FSPL + WESN zero-shot',
}
METHOD_ORDER = tuple(METHOD_LABELS)
WESN_METHODS = ('E2_scratch', 'E3_transfer', 'E4_zero_shot')


def _method_label(method, prediction_mode):
    if method in WESN_METHODS and prediction_mode == 'direct_pl':
        suffix = {'E2_scratch': 'scratch', 'E3_transfer': 'transfer', 'E4_zero_shot': 'zero-shot'}[method]
        return f'WESN {suffix} (direct PL)'
    return METHOD_LABELS[method]


def sparse_measurements(target_pl, train_indices, valid_mask):
    """Show training measurements only; leave other receiver positions blank."""
    target = np.asarray(target_pl).reshape(-1)
    measured = np.full(target.shape, np.nan, dtype=np.float64)
    measured[train_indices] = target[train_indices]
    measured[~np.asarray(valid_mask, dtype=bool).reshape(-1)] = np.nan
    return measured


def _neural_prediction(checkpoint, output, device, method, prediction_mode):
    predict_checkpoint(checkpoint, output, device)
    with np.load(output, allow_pickle=False) as exported:
        if method in WESN_METHODS and exported['wesn_prediction_mode'].item() != prediction_mode:
            raise ValueError('Checkpoint and run use different WESN prediction modes')
        return exported['prediction_db'].copy()


def collect_panels(run_dir, scene_id, device='auto'):
    """Load predictions and their held-out scores from a completed run."""
    root = Path(run_dir).resolve()
    config = yaml.safe_load((root / 'config.yml').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Run is incomplete; finish its experiments before plotting')
    prediction_mode = config.get('model', {}).get('prediction_mode', 'fspl_residual')
    scene = scene_from_config(config, scene_id)
    scene_dir = root / scene_id
    selected = set(summary.get('methods', [])) | set(summary.get('baselines', {}))
    methods = [method for method in METHOD_ORDER if method in selected]
    with np.load(scene_dir / 'split.npz', allow_pickle=False) as split:
        train = split['train'].copy()
    original = np.asarray(scene.target_pl_db, dtype=np.float64).reshape(-1).copy()
    mask = np.asarray(scene.valid_mask, dtype=bool).reshape(-1)
    original[~mask] = np.nan

    # E4 alone observes the source band; FSPL alone observes no PL labels.
    target_measured = bool(selected & {'B1_knn', 'B2_mlp', 'B3_nerf2', 'E2_scratch', 'E3_transfer'})
    measured_band, measurement_count = None, 0
    sparse = np.full(original.shape, np.nan)
    sparse_title = 'No training measurements (FSPL)'
    if target_measured or 'E4_zero_shot' in selected:
        measured_band = config['data'].get('target_band', 'f3') if target_measured else config['data'].get('source_band', 'f2')
        measured_pl = scene.target_pl_db if target_measured else scene.source_pl_db
        sparse = sparse_measurements(measured_pl, train, mask)
        measurement_count = len(train)
        percent = measurement_count / int(mask.sum()) * 100.0
        sparse_title = f'Sparse {measured_band} measurements ({percent:.1f}%)'
    panels = [
        {'method': 'target', 'title': 'Original target PL', 'values': original},
        {'method': 'sparse', 'title': sparse_title, 'values': sparse},
    ]
    with tempfile.TemporaryDirectory(prefix='icassp-figure-') as tmp:
        for method in methods:
            dest = scene_dir / method if method in WESN_METHODS else scene_dir / 'baselines' / method
            score = json.loads((dest / 'metrics.json').read_text())
            if method == 'E1_fspl':
                parameters = json.loads((dest / 'parameters.json').read_text())
                baseline = FSPL(parameters['tx_m'], parameters['frequency_ghz'])
                prediction = np.full(len(scene.rx_m), np.nan)
                prediction[mask] = baseline.predict(scene.rx_m[mask])
            elif method == 'B1_knn':
                parameters = json.loads((dest / 'parameters.json').read_text())
                with np.load(dest / 'observations.npz', allow_pickle=False) as observations:
                    baseline = KNN(**parameters).fit(observations['rx_m'], observations['path_loss_db'])
                prediction = np.full(len(scene.rx_m), np.nan)
                prediction[mask] = baseline.predict(scene.rx_m[mask])
            else:
                prediction = _neural_prediction(dest / 'final.pt', Path(tmp) / f'{method}.npz',
                                                device, method, prediction_mode)
            values = np.asarray(prediction, dtype=np.float64).reshape(-1).copy()
            values[~mask] = np.nan
            rmse = float(score['rmse_db'])
            panels.append({'method': method,
                           'title': f'{_method_label(method, prediction_mode)}\nTest RMSE {rmse:.2f} dB',
                           'values': values, 'rmse_db': rmse})
    metadata = {'scene_id': scene_id, 'run_dir': str(root), 'seed': config['train']['seed'],
                'wesn_prediction_mode': prediction_mode if selected.intersection(WESN_METHODS) else None,
                'target_frequency_ghz': config['data']['target_frequency_ghz'],
                'height': scene.height, 'width': scene.width, 'n_valid': int(mask.sum()),
                'n_training_measurements': measurement_count, 'n_allocated_training_positions': len(train),
                'measurement_band': measured_band, 'measurement_fraction': measurement_count / int(mask.sum()),
                'evaluation_limited': summary.get('evaluation_limited', False)}
    return scene, panels, metadata


def render_figure(scene, panels, output_path, metadata=None, *, vmin=None, vmax=None, dpi=300):
    """Save a shared-scale figure and a compact JSON sidecar."""
    output = Path(output_path)
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists():
        raise ValueError(f'Output already exists: {output}')
    original = np.asarray(panels[0]['values']).reshape(-1)
    valid_values = original[np.isfinite(original)]
    low = float(valid_values.min() if vmin is None else vmin)
    high = float(valid_values.max() if vmax is None else vmax)
    if low == high:
        high = low + 1.0
    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4.2 * columns, 4.0 * rows), constrained_layout=True)
    mask = np.asarray(scene.valid_mask, dtype=bool).reshape(scene.height, scene.width)
    background = np.ones((scene.height, scene.width, 4), dtype=np.float64)
    background[~mask] = (.72, .72, .72, 1.)
    extent = (-scene.pixel_size_m / 2, scene.width_m + scene.pixel_size_m / 2,
              scene.height_m + scene.pixel_size_m / 2, -scene.pixel_size_m / 2)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad((1., 1., 1., 0.))
    norm = Normalize(vmin=low, vmax=high)
    for ax, panel in zip(axes.flat, panels):
        values = np.asarray(panel['values']).reshape(scene.height, scene.width)
        ax.imshow(background, origin='upper', extent=extent, interpolation='nearest')
        image = ax.imshow(np.ma.masked_invalid(values), origin='upper', extent=extent,
                          interpolation='nearest', cmap=cmap, norm=norm)
        ax.set_title(panel['title'])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(axes.flat)[:len(panels)], label='Path loss (dB)', shrink=.85)
    suffix = ' — limited test subset' if (metadata or {}).get('evaluation_limited', False) else ''
    fig.suptitle(f'{scene.scene_id}: coordinate/frequency path-loss maps{suffix}', fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=dpi, metadata={'Creator': 'TL_ELM scripts/visualize.py'})
    finally:
        plt.close(fig)
    report = dict(metadata or {})
    report.update({'vmin_db': low, 'vmax_db': high, 'dpi': dpi,
                   'layout_rows': rows, 'layout_columns': columns,
                   'excluded_receiver_color': 'gray', 'unmeasured_sparse_color': 'white',
                   'panels': [{key: value for key, value in panel.items() if key != 'values'} for panel in panels]})
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def visualize(run_dir, scene_id, output_path, device='auto', **plot_options):
    output = Path(output_path)
    if output.exists() or output.with_suffix(output.suffix + '.json').exists():
        raise ValueError(f'Output already exists: {output}')
    scene, panels, metadata = collect_panels(run_dir, scene_id, device)
    return render_figure(scene, panels, output_path, metadata, **plot_options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    report = visualize(args.run_dir, args.scene, args.output, args.device,
                       vmin=args.vmin, vmax=args.vmax, dpi=args.dpi)
    print(f'Saved {args.output}; {len(report["panels"])} panels, '
          f'{report["n_training_measurements"]} sparse measurements.')


if __name__ == '__main__':
    main()
