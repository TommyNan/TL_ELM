"""Frequency features and Friis path loss in metres, GHz and dB.

FSPL uses the true positive Tx/Rx distance, without a distance floor.
Coincident locations are excluded by the dataset loader.
"""
from __future__ import annotations

import math
import numpy as np
import torch


SPEED_OF_LIGHT_M_S = 299_792_458.0
FREQUENCY_REFERENCE_GHZ = 1.0
WESN_PREDICTION_MODES = ("fspl_residual", "direct_pl")


def require_wesn_prediction_mode(mode):
    if mode not in WESN_PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {WESN_PREDICTION_MODES}")
    return mode


def numpy_fspl_db(rx_m, tx_m, frequency_ghz):
    """Friis loss at the actual Euclidean distance, with no clipping."""
    rx = np.asarray(rx_m, dtype=np.float64)
    tx = np.asarray(tx_m, dtype=np.float64)
    distance = np.linalg.norm(rx - tx, axis=-1)
    if np.any(distance <= 0):
        raise ValueError("FSPL requires positive distance; exclude coincident Tx/Rx receivers")
    return (20 * np.log10(distance) + 20 * np.log10(frequency_ghz)
            + 20 * math.log10(4 * math.pi * 1e9 / SPEED_OF_LIGHT_M_S))


def _torch_frequencies(frequency_ghz, reference):
    """Broadcast a scalar or one frequency per receiver, preserving autograd."""
    return torch.as_tensor(frequency_ghz, dtype=reference.dtype,
                           device=reference.device).reshape(-1).expand(len(reference))


def torch_frequency_feature(frequency_ghz, reference):
    """Log-frequency feature [B,1], using the fixed 1 GHz reference."""
    return torch.log(_torch_frequencies(frequency_ghz, reference) / FREQUENCY_REFERENCE_GHZ)[:, None]


def torch_fspl_db(rx_m, tx_m, frequency_ghz):
    """Differentiable Friis loss, accepting scalar frequency or [B]."""
    frequency = _torch_frequencies(frequency_ghz, rx_m)
    distance = (rx_m - tx_m).norm(dim=-1)
    if bool((distance <= 0).any()):
        raise ValueError("FSPL requires positive distance; exclude coincident Tx/Rx receivers")
    return (20 * torch.log10(distance) + 20 * torch.log10(frequency)
            + 20 * math.log10(4 * math.pi * 1e9 / SPEED_OF_LIGHT_M_S))
