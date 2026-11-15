"""Pointwise MLP regressor of complete target-band path loss.

Inputs: local Rx xyz(3), Tx xyz(3) and log carrier frequency(1), seven values per receiver.
Outputs: returns full path loss in dB.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from propagation import torch_frequency_feature


class MLP(nn.Module):
    """Predict [B] PL dB using the same call signature as RadioMapField."""

    def __init__(self, model_params):
        super().__init__()
        self.width_m = float(model_params['width_m'])
        self.height_m = float(model_params['height_m'])
        hidden = int(model_params.get('hidden_dim', 128))
        depth = int(model_params.get('num_layers', 4))

        self.register_buffer('extent_m', torch.tensor([self.width_m, self.height_m, 1.0]))
        layers = [nn.Linear(7, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden, hidden), nn.ReLU()))
        layers.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, rx_m, tx_m, frequency_ghz):
        """Return the unbounded final affine output in path-loss dB units."""
        frequency_feature = torch_frequency_feature(frequency_ghz, rx_m)
        features = torch.cat((
            rx_m / self.extent_m, tx_m / self.extent_m, frequency_feature,
        ), dim=-1)
        return self.network(features).squeeze(-1)
