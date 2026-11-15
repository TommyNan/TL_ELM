"""Standalone direct-PL NeRF2-inspired baseline.

Reference: https://github.com/XPengZhao/NeRF2/blob/main/model.py
The original attenuation defaults are eight 256-wide hidden layers with an
input skip after zero-based layer 4. This file contains its network, coordinate
encoding, finite rays and power renderer.

Only Tx/Rx coordinates and carrier frequency are inputs. Latent power is a
predicted received/transmitted ratio G; PL = -10 log10(G), supervised directly
by PL labels without FSPL, a residual target, output scaling or calibration.

"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from propagation import torch_frequency_feature


class SkipMLP(nn.Module):
    """Pointwise attenuation MLP with the original NeRF2 skip placement.

    NeRF2 concatenates the original encoded input *after* hidden layer 4
    (zero-based), so hidden layer 5 receives ``hidden_dim + input_dim``.
    Source: https://github.com/XPengZhao/NeRF2/blob/main/model.py .
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, skips=(4,)):
        super().__init__()
        self.skips = tuple(sorted(set(skips)))
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] +
            [nn.Linear(hidden_dim + (input_dim if i - 1 in self.skips else 0), hidden_dim)
             for i in range(1, num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        hidden = inputs
        for i, layer in enumerate(self.linears):
            hidden = F.relu(layer(hidden))
            if i in self.skips:
                hidden = torch.cat((inputs, hidden), dim=-1)
        return self.output_layer(hidden)


class NeRF2Field(nn.Module):
    """Standalone coordinate/frequency ray model of complete path loss dB.

    Network topology, coordinate encoding, ray construction and compositing
    are implemented here, independently of the project's WESN model.
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.width_m = float(model_params['width_m'])
        self.height_m = float(model_params['height_m'])
        self.num_samples = int(model_params.get('num_samples', 16))
        self.num_rays = int(model_params.get('num_rays', 8))
        hidden = int(model_params.get('hidden_dim', 256))
        self.num_freq = int(model_params.get('num_freq', 4))
        self.attenuation_kind = 'mlp'

        self.register_buffer('extent_m', torch.tensor([self.width_m, self.height_m, 1.0]))
        angles = torch.arange(self.num_rays, dtype=torch.float32) * (2 * math.pi / self.num_rays)
        self.register_buffer('directions', torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], -1))
        self.register_buffer('sample_fraction', (torch.arange(self.num_samples, dtype=torch.float32) + .5) / self.num_samples)
        self.register_buffer('encoding_frequencies', 2. ** torch.arange(self.num_freq, dtype=torch.float32))
        positional_dim = 3 * (1 + 2 * self.num_freq)
        # The same coordinate/frequency information is available in every
        # neural model; material maps and measured-label context are excluded.
        field_input_dim = positional_dim + 1
        self.feature_dim = hidden
        self.attenuation = SkipMLP(
            field_input_dim, hidden, hidden + 1,
            num_layers=int(model_params.get('mlp_num_layers', 8)),
            skips=model_params.get('mlp_skips', [4]),
        )
        self.radiance = nn.Sequential(
            nn.Linear(hidden + 3 + 3 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )

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

    def _render_loss_db(self, rx_m, tx_m, frequency_ghz):
        """Integrate predicted power and return its negative dB logarithm.

        The power ratio is supervised directly by complete PL labels. Power-
        only labels do not identify a unique latent phase or complex channel.
        """
        batch = len(rx_m)
        frequency_feature = torch_frequency_feature(frequency_ghz, rx_m)
        frequency_feature = frequency_feature.view(batch, 1, 1, 1).expand(
            -1, self.num_rays, self.num_samples, -1)
        points, dirs, length = self._rays(rx_m)
        normalized = points / self.extent_m
        field_input = torch.cat((self._encode(normalized), frequency_feature), dim=-1)
        field = self.attenuation(field_input.reshape(batch * self.num_rays, self.num_samples, -1))
        field = field.reshape(batch, self.num_rays, self.num_samples, -1)
        sigma = F.softplus(field[..., 0])
        ds = length[..., None] / (max(self.width_m, self.height_m) * self.num_samples)
        optical_depth = sigma * ds
        log_transmittance = -torch.cat((torch.zeros_like(optical_depth[..., :1]),
                                        optical_depth.cumsum(-1)[..., :-1]), dim=-1)
        # Machine tiny only handles a representational zero in log/complex
        # arithmetic. It is not an absolute received-power floor such as 1e-8.
        tiny = torch.finfo(optical_depth.dtype).tiny
        log_alpha = torch.log((-torch.expm1(-optical_depth)).clamp_min(tiny))
        tx_feature = (tx_m / self.extent_m).view(batch, 1, 1, 3).expand(-1, self.num_rays, self.num_samples, -1)
        view_feature = dirs.unsqueeze(-2).expand(-1, -1, self.num_samples, -1)
        radiance = self.radiance(torch.cat((field[..., 1:], tx_feature, view_feature, frequency_feature), -1))
        log_amplitude = radiance[..., 0]
        phase = math.pi * torch.tanh(radiance[..., 1])
        log_weighted_amp = log_transmittance + log_alpha + log_amplitude
        shift = log_weighted_amp.max(dim=-1, keepdim=True).values
        scaled_amp = torch.exp(log_weighted_amp - shift)
        # Coherent sum within each ray, followed by an incoherent mean of ray
        # powers, as in this fixed power-map renderer. Max shifting preserves
        # both very small and large relative gains without exponent overflow.
        real = (scaled_amp * phase.cos()).sum(-1)
        imag = (scaled_amp * phase.sin()).sum(-1)
        log_ray_power = 2 * shift.squeeze(-1) + torch.log(
            (real.square() + imag.square()).clamp_min(tiny))
        log_relative_power = torch.logsumexp(log_ray_power, dim=-1) - math.log(self.num_rays)
        # Natural-log power to dB: this is the unit conversion, not an
        # adjustable output multiplier or label normalization.
        return -(10.0 / math.log(10.0)) * log_relative_power

    def forward(self, rx_m, tx_m, frequency_ghz):
        """Return the direct [B] complete path-loss prediction, in dB."""
        return self.predict_path_loss_db(rx_m, tx_m, frequency_ghz)

    def predict_path_loss_db(self, rx_m, tx_m, frequency_ghz):
        """Return -10 log10(predicted power ratio), without an FSPL prior."""
        return self._render_loss_db(rx_m, tx_m, frequency_ghz)

