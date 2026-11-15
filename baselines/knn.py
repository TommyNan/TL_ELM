"""Independent spatial KNN regression of measured path loss in dB.

Predictions are the uniform mean of the k nearest training PL labels using
Euclidean distances between 3D metre coordinates.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _coordinates(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite array with shape [N, 3]")
    return values


class KNN:
    """Regress raw PL dB from the nearest training receiver coordinates.

    ``k`` defaults to five; fewer available observations are all used. Queries
    are batched to bound memory use. Repeated training coordinates remain
    distinct observations, as in ordinary uniform KNN regression. An exact
    coordinate match does not override the k-neighbor mean.
    """

    def __init__(self, k=5, batch_size=4096):
        self.k = int(k)
        self.batch_size = int(batch_size)
        self._tree = None

    def fit(self, rx_m, path_loss_db):
        rx = _coordinates(rx_m, "rx_m")
        labels = np.asarray(path_loss_db, dtype=np.float64)
        if not len(rx):
            raise ValueError("At least one measured location is required")
        if labels.shape != (len(rx),) or not np.isfinite(labels).all():
            raise ValueError("path_loss_db must be finite with shape [N]")
        locations, measured = rx.copy(), labels.copy()
        tree = cKDTree(locations, copy_data=True)
        locations.setflags(write=False)
        measured.setflags(write=False)
        self.observation_rx_m = locations
        self.path_loss_db = measured
        self.n_observations = len(locations)
        self._tree = tree
        return self

    def predict(self, query_rx_m):
        if self._tree is None:
            raise RuntimeError("Call fit before predict")
        queries = _coordinates(query_rx_m, "query_rx_m")
        prediction = np.empty(len(queries), dtype=np.float64)
        k = min(self.k, self.n_observations)
        for start in range(0, len(queries), self.batch_size):
            query = queries[start:start + self.batch_size]
            _, indices = self._tree.query(query, k=k)
            indices = indices.reshape(len(query), k)
            prediction[start:start + len(query)] = self.path_loss_db[indices].mean(axis=1)
        return prediction