"""NeRF-inspired ray field and WESN radio-map model with two PL modes.

The attenuation field is a WESN over ordered samples on receiver-centred rays.
A radiance MLP supplies log-relative-amplitude and phase terms conditioned on
transmitter coordinates and carrier frequency. Finite-volume compositing gives
E_r = sum_j T_j (1-exp(-sigma_j*ds_j)) a_j exp(i*phi_j),
P = mean_r |E_r|^2. The shared renderer returns the score -10 log10(P).
Direct mode fits complete PL in dB. Residual mode adds the learned dB
correction to FSPL: PL = FSPL + score. Both modes use the original PL labels.
The rendered latent power is not a calibrated physical power ratio.

Power is evaluated in a shifted log domain, so there is no arbitrary
calibration offset or absolute power epsilon limiting the dB residual range.
The predict_path_loss_db API returns complete physical PL dB. This is a
model-guided surrogate, not an exact reproduction of NeRF2 or a CSI estimate.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from Models.wesn import WESN
from propagation import torch_frequency_feature, torch_fspl_db


def _log_alpha(raw_density, ds, optical_depth):
    """Compute log(1-exp(-softplus(raw)*ds)) without a huge 1/alpha gradient.

    In the small-depth limit log(alpha) equals log(depth) to float32 accuracy.
    Safe inputs in the unselected branches also keep their backward pass finite.
    The existing machine-tiny alpha threshold is preserved in the log domain.
    """
    cutoff = -20.0
    log_sigma = torch.where(raw_density < cutoff, raw_density,
                            F.softplus(raw_density.clamp_min(cutoff)).log())
    log_depth = log_sigma + ds.log()
    small = log_depth < cutoff
    regular_depth = torch.where(small, torch.ones_like(optical_depth), optical_depth)
    log_alpha = torch.where(small, log_depth, (-torch.expm1(-regular_depth)).log())
    return log_alpha.clamp_min(math.log(torch.finfo(raw_density.dtype).tiny))


def _log_power(real, imag, tiny):
    """Accumulate the small coherent remainder in double precision.

    A finite float32 remainder can have a finite log-power but an overflowing
    intermediate derivative through square/log. Keep the original tiny floor.
    """
    return (real.to(torch.float64).square() + imag.to(torch.float64).square()).clamp_min(tiny).log()


class SkipMLP(nn.Module):
    """Pointwise attenuation MLP with the original NeRF2 skip placement.

    NeRF2 concatenates the original encoded input *after* hidden layer 4
    (zero-based), so hidden layer 5 receives ``hidden_dim + input_dim``.
    Source: https://github.com/XPengZhao/NeRF2/blob/main/model.py .
    The output head here provides one density plus radiance features for our
    power-only renderer; it is not the original complex-channel output head.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, skips=(4,)):
        super().__init__()
        if num_layers < 1 or any(i < 0 or i >= num_layers - 1 for i in skips):
            raise ValueError('Use positive mlp_num_layers and skips before its last layer')
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


class RayField(nn.Module):
    """Shared ray construction and learned power integration.

    Scene width/height set the coordinate normalization and ray bounds.
    Encoded coordinates and log carrier frequency condition the learned field.
    No material map, observation context or per-query label is accepted.
    """
    def __init__(self, model_params: dict):
        super().__init__()
        self.width_m = float(model_params['width_m'])
        self.height_m = float(model_params['height_m'])
        self.num_samples = int(model_params.get('num_samples', 16))
        self.num_rays = int(model_params.get('num_rays', 8))
        hidden = int(model_params.get('hidden_dim', 32))
        self.num_freq = int(model_params.get('num_freq', 4))
        self.attenuation_kind = model_params.get('attenuation', 'wesn')
        win_len = int(model_params.get('win_len', 4))
        if self.num_samples < 1 or self.num_rays < 1 or hidden < 2 or self.num_freq < 0:
            raise ValueError('invalid ray, sample, hidden or encoding size')
        if win_len < 0 or win_len > self.num_samples:
            raise ValueError('invalid WESN window length')
        self.register_buffer('extent_m', torch.tensor([self.width_m, self.height_m, max(self.width_m, self.height_m)]))
        angles = torch.arange(self.num_rays, dtype=torch.float32) * (2 * math.pi / self.num_rays)
        self.register_buffer('directions', torch.stack([angles.cos(), angles.sin(), torch.zeros_like(angles)], -1))
        self.register_buffer('sample_fraction', (torch.arange(self.num_samples, dtype=torch.float32) + .5) / self.num_samples)
        self.register_buffer('encoding_frequencies', 2. ** torch.arange(self.num_freq, dtype=torch.float32))
        positional_dim = 3 * (1 + 2 * self.num_freq)
        # The same coordinate/frequency information is available in every
        # neural model; material maps and measured-label context are excluded.
        field_input_dim = positional_dim + 1
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
            self.attenuation = SkipMLP(
                field_input_dim, hidden, hidden + 1,
                num_layers=int(model_params.get('mlp_num_layers', 8)),
                skips=model_params.get('mlp_skips', [4]),
            )
        else:
            raise ValueError("attenuation must be 'wesn' or 'mlp'")
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

    def _render_score(self, rx_m, tx_m, frequency_ghz):
        """Integrate latent power and return its negative logarithmic score.

        RadioMapField interprets the score as complete PL or a signed FSPL
        correction, both in dB. PL labels do not identify unique phase or CSI.
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
        # Machine tiny protects representational zeros, not an adjustable
        # received-power floor. Evaluate its logarithm without reciprocal overflow.
        tiny = torch.finfo(optical_depth.dtype).tiny
        log_alpha = _log_alpha(field[..., 0], ds, optical_depth)
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
        log_ray_power = 2 * shift.squeeze(-1).to(torch.float64) + _log_power(real, imag, tiny)
        log_relative_power = torch.logsumexp(log_ray_power, dim=-1) - math.log(self.num_rays)
        # Natural-log power to the dB regression output.
        return (-(10.0 / math.log(10.0)) * log_relative_power).to(field.dtype)


class RadioMapField(RayField):
    """WESN complete PL regression, with an optional analytical FSPL prior.

    ``fspl_residual`` adds FSPL and a learned correction in dB;
    ``direct_pl`` regresses complete PL in dB and never evaluates FSPL.
    """

    def __init__(self, model_params):
        super().__init__(model_params)
        self.prediction_mode = model_params.get('prediction_mode', 'fspl_residual')

    def forward(self, rx_m, tx_m, frequency_ghz):
        if self.prediction_mode == 'fspl_residual':
            prior = torch_fspl_db(rx_m, tx_m, frequency_ghz)
            return prior + self._render_score(rx_m, tx_m, frequency_ghz)
        return self._render_score(rx_m, tx_m, frequency_ghz)

    def predict_path_loss_db(self, rx_m, tx_m, frequency_ghz):
        """Return complete PL in dB without inverse scaling."""
        return self(rx_m, tx_m, frequency_ghz)

    def predict_residual_db(self, rx_m, tx_m, frequency_ghz):
        """Return the signed FSPL correction, only in fspl_residual mode."""
        if self.prediction_mode != 'fspl_residual':
            raise ValueError('predict_residual_db is unavailable in direct_pl mode, which predicts full PL')
        return self._render_score(rx_m, tx_m, frequency_ghz)
