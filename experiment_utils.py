"""Shared runtime, reproducibility and scoring utilities for ICASSP experiments."""

import hashlib
import random

import numpy as np
import torch


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
    if not error.size or not np.isfinite(error).all():
        raise ValueError("Cannot score empty or nonfinite predictions")
    return {"rmse_db": float(np.sqrt(np.mean(error**2))),
            "mae_db": float(np.mean(np.abs(error))),
            "bias_db": float(np.mean(error)), "n_points": int(error.size)}


def aggregate_results(rows):
    """Primary result is equal scene weighting, never pixel-count weighting."""
    result = {}
    for approach in sorted({row["approach"] for row in rows}):
        selected = [r for r in rows if r["approach"] == approach]
        entry = {"n_scenes": len(selected),
                 "n_buildings": len({r["building_id"] for r in selected})}
        for name in ("rmse_db", "mae_db", "bias_db"):
            values = np.array([r[name] for r in selected])
            entry["scene_mean_" + name] = float(values.mean())
            entry["scene_std_" + name] = float(values.std())
        building_rmse = [np.mean([r["rmse_db"] for r in selected if r["building_id"] == b])
                         for b in sorted({r["building_id"] for r in selected})]
        entry["building_mean_rmse_db"] = float(np.mean(building_rmse))
        entry["pooled_rmse_db"] = float(np.sqrt(
            sum(r["rmse_db"]**2 * r["n_points"] for r in selected) /
            sum(r["n_points"] for r in selected)))
        result[approach] = entry
    return result


def as_geometry(array, device):
    return torch.as_tensor(np.ascontiguousarray(array.transpose(2, 0, 1)[None]),
                           dtype=torch.float32, device=device)


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(value)
