"""Invertible PL min-max transforms using known, band-specific dataset ranges."""
from __future__ import annotations

import numpy as np
import torch


class BandMinMax:
    """Scale complete PL to normalized units and restore predictions to dB.

    Statistics are supplied by the dataset scan and saved with the model.
    Predictions are not clipped to [0, 1]. Missing statistics preserve the
    original dB interface for models constructed without normalization.
    """

    def __init__(self, statistics=None):
        self.statistics = statistics
        bands = list(statistics['bands'].values()) if statistics is not None else []
        self.frequencies = np.array([band['frequency_ghz'] for band in bands])
        self.ranges = np.array([[band['min_db'], (band['max_db'] - band['min_db']) or 1.0]
                                for band in bands])

    @property
    def enabled(self):
        return self.statistics is not None

    def _bounds(self, values, frequency_ghz):
        """Select each frequency's minimum/span and broadcast over label axes."""
        if torch.is_tensor(values) and torch.is_tensor(frequency_ghz):
            frequency = frequency_ghz.to(dtype=values.dtype, device=values.device)
            ranges = values.new_tensor(self.ranges)
            matched = torch.isclose(frequency[..., None], frequency.new_tensor(self.frequencies))
            index = matched.long().argmax(-1)
        else:
            frequency = np.asarray(frequency_ghz, dtype=np.float64)
            ranges = self.ranges
            matched = np.isclose(frequency[..., None], self.frequencies)
            index = matched.argmax(-1)
        if not matched.any(-1).all():
            raise ValueError('No PL min-max statistics for one or more supplied frequencies')

        bounds = ranges[index]
        if frequency.ndim == 0:
            return float(bounds[0]), float(bounds[1])
        if torch.is_tensor(values):
            bounds = torch.as_tensor(bounds, dtype=values.dtype, device=values.device)
        else:
            bounds = bounds.astype(np.result_type(np.asarray(values).dtype, np.float32))
        shape = frequency.shape + (1,) * (values.ndim - frequency.ndim)
        return bounds[..., 0].reshape(shape), bounds[..., 1].reshape(shape)

    def normalize(self, values, frequency_ghz):
        if not self.enabled:
            return values
        minimum, span = self._bounds(values, frequency_ghz)
        return (values - minimum) / span

    def denormalize(self, values, frequency_ghz):
        if not self.enabled:
            return values
        minimum, span = self._bounds(values, frequency_ghz)
        return values * span + minimum

    def denormalize_residual(self, values, frequency_ghz):
        """A residual is scaled by the band span, without an intercept."""
        if not self.enabled:
            return values
        _, span = self._bounds(values, frequency_ghz)
        return values * span
