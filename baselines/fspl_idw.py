"""Baselines: free-space path loss plus local inverse-distance residual interpolation.

Only the observations passed to ``fit`` are used. In the ICASSP experiment,
these are all target-band labels at the common 20% training coordinates.
Prediction interpolates the observed excess loss above Free Space PathLoss (FSPL) in dB and adds
the FSPL at each query.

By default interpolation uses the nearest 32 observations (or all
available observations if fewer); setting ``k`` to the observation count gives
global IDW. Queries are batched to bound memory use.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _coordinates(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite array with shape [N, 3]")
    return values


def fspl_db(rx_m, tx_m, frequency_ghz):
    """FSPL in dB, with metres and GHz; distances are clamped to 0.25 m."""
    rx = _coordinates(rx_m, "rx_m")
    tx = np.asarray(tx_m, dtype=np.float64)
    frequency = float(frequency_ghz)
    if tx.shape != (3,) or not np.isfinite(tx).all():
        raise ValueError("tx_m must be finite with shape [3]")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency_ghz must be positive and finite")
    distance_m = np.maximum(np.linalg.norm(rx - tx, axis=1), 0.25)
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency) + 32.447783


class FSPLIDW:
    """Fit excess path loss at measured locations, then interpolate new points.

    ``k`` is the number of nearest measured locations used at each query;
    ``power`` is the inverse-distance exponent. All distances use
    the supplied 3D metre coordinates.
    """

    def __init__(self, k=32, power=2.0, batch_size=4096):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if (isinstance(batch_size, bool) or
                not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(float(power)) or float(power) <= 0:
            raise ValueError("power must be positive and finite")
        self.k = int(k)
        self.power = float(power)
        self.batch_size = int(batch_size)
        self._tree = None

    def fit(self, rx_m, path_loss_db, tx_m, frequency_ghz):
        rx = _coordinates(rx_m, "rx_m")
        labels = np.asarray(path_loss_db, dtype=np.float64)
        if len(rx) == 0:
            raise ValueError("At least one measured location is required")
        if labels.shape != (len(rx),) or not np.isfinite(labels).all():
            raise ValueError("path_loss_db must be finite with shape [N]")
        residuals = labels - fspl_db(rx, tx_m, frequency_ghz)
        self.tx_m = np.asarray(tx_m, dtype=np.float64).copy()
        self.frequency_ghz = float(frequency_ghz)
        locations, inverse, counts = np.unique(rx, axis=0, return_inverse=True,
                                               return_counts=True)
        self.observation_rx_m = locations.copy()
        self.residuals_db = np.bincount(inverse, weights=residuals) / counts
        self.n_observations = len(rx)
        self._tree = cKDTree(self.observation_rx_m)
        return self

    def predict(self, query_rx_m):
        if self._tree is None:
            raise RuntimeError("Call fit before predict")
        queries = _coordinates(query_rx_m, "query_rx_m")
        predicted_residual = np.empty(len(queries), dtype=np.float64)
        k = min(self.k, len(self.observation_rx_m))
        for start in range(0, len(queries), self.batch_size):
            query = queries[start:start + self.batch_size]
            distances, indices = self._tree.query(query, k=k)
            distances = distances.reshape(len(query), k)
            indices = indices.reshape(len(query), k)
            exact = distances[:, 0] == 0
            values = np.empty(len(query), dtype=np.float64)
            values[exact] = self.residuals_db[indices[exact, 0]]
            remaining = ~exact
            if remaining.any():
                # Scale by each query's nearest distance before exponentiation
                # to keep weights finite for very close measured locations.
                d = distances[remaining]
                weights = (d[:, :1] / d) ** self.power
                weights /= weights.sum(axis=1, keepdims=True)
                values[remaining] = np.sum(weights * self.residuals_db[indices[remaining]], axis=1)
            predicted_residual[start:start + len(query)] = values
        return fspl_db(queries, self.tx_m, self.frequency_ghz) + predicted_residual
