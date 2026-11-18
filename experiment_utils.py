"""Shared runtime, reproducibility and scoring utilities for ICASSP experiments."""

import hashlib
from pathlib import Path
import platform
import random

import numpy as np
import torch


METRIC_FIELDS = ("rmse_db", "mae_db", "train_time_sec", "test_time_sec")


def file_fingerprint(root, paths):
    """Content fingerprint independent of the machine's absolute checkout path."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def reproducibility_metadata(project_root, dataset_root, scenes, source_band, target_band, device):
    """Record code/data identity and runtime versions without environment secrets."""
    project_root, dataset_root = Path(project_root), Path(dataset_root)
    sources = [project_root / name for name in ('main.py', 'propagation.py', 'experiment_utils.py')]
    for directory in ('Models', 'baselines', 'datasets'):
        sources.extend((project_root / directory).glob('*.py'))
    data = [dataset_root / 'positions.csv', dataset_root / 'building_details.csv']
    for scene in sorted(scenes):
        for band in (source_band, target_band):
            data.append(dataset_root / scene / f'{band}_path_loss.png')
    selected_device = torch.device(device)
    return {'source_sha256': file_fingerprint(project_root, sources),
            'dataset_sha256': file_fingerprint(dataset_root, data),
            'python_version': platform.python_version(), 'platform': platform.system(),
            'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
            'device': str(selected_device), 'cuda_runtime_version': torch.version.cuda,
            'cuda_device_name': (torch.cuda.get_device_name(selected_device)
                                 if selected_device.type == 'cuda' else None),
            'cuda_bitwise_determinism_guaranteed': False}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scene_seed(seed, scene_id):
    return (seed + int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16)) % (2**32)


def metrics(prediction, target):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {"rmse_db": float(np.sqrt(np.mean(error**2))),
            "mae_db": float(np.mean(np.abs(error))), "n_points": int(error.size)}


def aggregate_results(rows):
    """Average each of the four metrics equally over scenes."""
    result = {}
    for approach in sorted({row["approach"] for row in rows}):
        selected = [row for row in rows if row["approach"] == approach]
        result[approach] = {"n_scenes": len(selected),
                            **{name: float(np.mean([row[name] for row in selected]))
                               for name in METRIC_FIELDS}}
    return result


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(value)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm):
    """Clip gradients, retrying the norm in float64 if float32 overflows.

    The usual PyTorch path handles normal gradients. Large but finite
    gradients can overflow its sum of squares; only that case is retried.
    Actual NaN/Inf gradients still stop training before an optimizer update.
    """
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False)
    except RuntimeError:
        gradients = [p.grad for p in parameters if p.grad is not None]
        norms = torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients])
        total_norm = torch.linalg.vector_norm(norms)
        if not bool(torch.isfinite(total_norm)):
            raise
        coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.)
        for gradient in gradients:
            gradient.copy_((gradient.double() * coefficient).to(gradient.dtype))
        return total_norm


def sync_if_cuda(device):
    """Finish pending GPU work before a performance timing boundary."""
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
