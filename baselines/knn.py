"""Independent spatial KNN regression of measured path loss in dB.

Predictions are the uniform mean of the k nearest training PL labels using
Euclidean distances between 3D metre coordinates. This model neither subtracts
nor adds FSPL and has no inverse-distance weights or material inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


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
        if self.k < 1 or self.batch_size < 1:
            raise ValueError("k and batch_size must be positive")
        self._tree = None

    def fit(self, rx_m, path_loss_db):
        locations = np.asarray(rx_m, dtype=np.float64).copy()
        measured = np.asarray(path_loss_db, dtype=np.float64).copy()
        if not len(locations):
            raise ValueError("At least one measured location is required")
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
        queries = np.asarray(query_rx_m, dtype=np.float64)
        prediction = np.empty(len(queries), dtype=np.float64)
        k = min(self.k, self.n_observations)
        for start in range(0, len(queries), self.batch_size):
            query = queries[start:start + self.batch_size]
            _, indices = self._tree.query(query, k=k)
            indices = indices.reshape(len(query), k)
            prediction[start:start + len(query)] = self.path_loss_db[indices].mean(axis=1)
        return prediction
