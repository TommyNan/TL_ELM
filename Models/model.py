"""
NeRF2-inspired radio map field.

The attenuation field is a WESN over ordered samples on receiver-centred rays.
A radiance MLP supplies complex source terms conditioned on transmitter,
frequency and sparse observation features. Finite-volume compositing gives
E_r = sum_j T_j (1-exp(-sigma_j*ds_j)) a_j exp(i*phi_j),
P = mean_r |E_r|^2. Prediction is FSPL(d,f) + calibration - 10 log10(P).

This is a learned path-loss surrogate with a free-space prior, not an exact
reproduction of original NeRF2 or a complex-channel reconstruction. All ray
lengths are finite and scene-bounded; the data supplies power labels only.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from Models.wesn import WESN


class RadioMapField(nn.Module):
    """Predict path loss in dB from local metre coordinates and material maps.

    ``geometry`` is [1, 2, H, W], raw reflectance/transmittance dB. Image row
    increases with local y. Context is [B, context_dim]; zeros mean no observed
    channels. Callers must never encode query labels in context.
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.width_m = float(model_params['width_m'])
        self.height_m = float(model_params['height_m'])
        self.num_samples = int(model_params.get('num_samples', 16))
        self.num_rays = int(model_params.get('num_rays', 8))
        hidden = int(model_params.get('hidden_dim', 32))
        self.num_freq = int(model_params.get('num_freq', 4))
        self.context_dim = int(model_params.get('context_dim', 0))
        self.attenuation_kind = model_params.get('attenuation', 'wesn')
        win_len = int(model_params.get('win_len', 4))
        if min(self.width_m, self.height_m) <= 0:
            raise ValueError('scene extents must be positive')
        if self.num_samples < 1 or self.num_rays < 1 or hidden < 2 or self.num_freq < 0:
            raise ValueError('invalid ray, sample, hidden or encoding size')
        if self.context_dim < 0 or win_len < 0 or win_len > self.num_samples:
            raise ValueError('invalid context dimension or WESN window length')
        self.register_buffer('extent_m', torch.tensor([self.width_m, self.height_m, max(self.width_m, self.height_m)]))
        angles = torch.arange(self.num_rays, dtype=torch.float32) * (2 * math.pi / self.num_rays)
        self.register_buffer('directions', torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], -1))
        self.register_buffer('sample_fraction', (torch.arange(self.num_samples, dtype=torch.float32) + .5) / self.num_samples)
        self.register_buffer('encoding_frequencies', 2. ** torch.arange(self.num_freq, dtype=torch.float32))
        positional_dim = 3 * (1 + 2 * self.num_freq)
        # Geometry and frequency belong to the attenuation field; observations
        # condition the source/radiance branch rather than modifying the scene.
        field_input_dim = positional_dim + 2 + 1
        self.feature_dim = hidden
        if self.attenuation_kind == 'wesn':
            self.attenuation = WESN(
                hidden_dim=hidden,
                connectivity=float(model_params.get('connectivity', .5)),
                leaky=float(model_params.get('leaky', 1.)),
                spectral_radius=float(model_params.get('spectral_radius', .9)),
                win_len=win_len,
                output_dim=hidden + 1,
            )
            # Existing WESN creates its readout and fixed reservoir lazily.
            # Materialize now so optimizers and strict checkpoint loading see
            # every tensor, including the persistent non-trainable reservoir.
            with torch.no_grad():
                self.attenuation(torch.zeros(1, self.num_samples, field_input_dim))
        elif self.attenuation_kind == 'mlp':
            self.attenuation = nn.Sequential(nn.Linear(field_input_dim, hidden), nn.SiLU(),
                                             nn.Linear(hidden, hidden + 1))
        else:
            raise ValueError("attenuation must be 'wesn' or 'mlp'")
        self.radiance = nn.Sequential(
            nn.Linear(hidden + 3 + 3 + 1 + self.context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )
        self.calibration_db = nn.Parameter(torch.tensor(15.))

    def _encode(self, xyz):
        if self.num_freq == 0:
            return xyz
        phases = xyz.unsqueeze(-1) * self.encoding_frequencies * math.pi
        return torch.cat((xyz, phases.sin().flatten(-2), phases.cos().flatten(-2)), -1)

    def _rays(self, rx_m):
        """Midpoint quadrature ending at the rectangular scene boundary."""
        dirs = self.directions[None, :, :].expand(len(rx_m), -1, -1)
        origin = rx_m[:, None, :].expand_as(dirs)
        xy_dir = dirs[..., :2]
        xy_origin = origin[..., :2]
        extent = self.extent_m[:2]
        numerator = torch.where(xy_dir >= 0, extent - xy_origin, -xy_origin)
        # For exactly axis-parallel rays, the other axis limits the distance.
        denominator = torch.where(xy_dir.abs() > 1e-6, xy_dir, torch.ones_like(xy_dir))
        limits = numerator / denominator
        limits = torch.where(xy_dir.abs() > 1e-6, limits, torch.full_like(limits, float('inf')))
        ray_length = limits.min(-1).values.clamp_min(.01)
        points = origin.unsqueeze(-2) + dirs.unsqueeze(-2) * (ray_length[..., None] * self.sample_fraction)[..., None]
        return points, dirs, ray_length

    def forward(self, rx_m, tx_m, frequency_ghz, geometry, context=None):
        """Return [B] path loss dB. rx_m/tx_m are [B,3] in local metres."""
        if rx_m.ndim != 2 or rx_m.shape[-1] != 3 or tx_m.shape != rx_m.shape:
            raise ValueError('rx_m and tx_m must have shape [B,3]')
        if geometry.ndim != 4 or geometry.shape[:2] != (1, 2):
            raise ValueError('geometry must have shape [1,2,H,W]')
        batch = len(rx_m)
        frequency = torch.as_tensor(frequency_ghz, device=rx_m.device, dtype=rx_m.dtype)
        if frequency.numel() == 1:
            frequency = frequency.reshape(1).expand(batch)
        if frequency.shape != (batch,) or not bool(torch.all(frequency > 0)):
            raise ValueError('frequency_ghz must be positive scalar or [B]')
        if context is None:
            context = rx_m.new_zeros(batch, self.context_dim)
        if context.shape != (batch, self.context_dim):
            raise ValueError('context has wrong shape')
        points, dirs, length = self._rays(rx_m)
        normalized = points / self.extent_m
        grid = normalized[..., :2].reshape(1, batch * self.num_rays, self.num_samples, 2) * 2 - 1
        materials = F.grid_sample(geometry, grid, mode='bilinear', padding_mode='border', align_corners=True)
        materials = materials[0].permute(1, 2, 0).reshape(batch, self.num_rays, self.num_samples, 2) / 60.
        freq_feature = torch.log(frequency / 3.5).view(batch, 1, 1, 1).expand(-1, self.num_rays, self.num_samples, -1)
        field_input = torch.cat((self._encode(normalized), materials, freq_feature), -1)
        field = self.attenuation(field_input.reshape(batch * self.num_rays, self.num_samples, -1))
        field = field.reshape(batch, self.num_rays, self.num_samples, -1)
        sigma = F.softplus(field[..., 0])
        ds = length[..., None] / (max(self.width_m, self.height_m) * self.num_samples)
        optical_depth = sigma * ds
        transmittance = torch.exp(-(optical_depth.cumsum(-1) - optical_depth))
        alpha = -torch.expm1(-optical_depth)
        tx_feature = (tx_m / self.extent_m).view(batch, 1, 1, 3).expand(-1, self.num_rays, self.num_samples, -1)
        view_feature = dirs.unsqueeze(-2).expand(-1, -1, self.num_samples, -1)
        observed = context.view(batch, 1, 1, self.context_dim).expand(-1, self.num_rays, self.num_samples, -1)
        radiance = self.radiance(torch.cat((field[..., 1:], tx_feature, view_feature, freq_feature, observed), -1))
        amplitude = F.softplus(radiance[..., 0])
        phase = math.pi * torch.tanh(radiance[..., 1])
        weighted_amp = transmittance * alpha * amplitude
        # Real form of the complex integral avoids unnecessary complex autograd.
        real = (weighted_amp * phase.cos()).sum(-1)
        imag = (weighted_amp * phase.sin()).sum(-1)
        power = (real.square() + imag.square()).mean(-1)
        distance_m = (tx_m - rx_m).norm(dim=-1).clamp_min(.25)
        fspl_db = 20 * torch.log10(distance_m) + 20 * torch.log10(frequency) + 32.4478
        return fspl_db + self.calibration_db - 10 * torch.log10(power + 1e-8)
