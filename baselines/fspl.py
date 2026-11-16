"""Label-free Free Space Path Loss reference, in metres/GHz/dB.

This baseline uses only the supplied transmitter, receiver and carrier
frequency. It does not fit an offset or use any path-loss measurements.
"""
import numpy as np

from propagation import numpy_fspl_db


class FSPL:
    """Predict the shared Friis formula at each receiver without fitting labels."""

    def __init__(self, tx_m, frequency_ghz):
        tx = np.asarray(tx_m, dtype=np.float64).copy()
        frequency = float(frequency_ghz)
        if tx.shape != (3,):
            raise ValueError("tx_m must have shape [3]")
        tx.setflags(write=False)
        self.tx_m = tx
        self.frequency_ghz = frequency

    def predict(self, rx_m):
        return numpy_fspl_db(rx_m, self.tx_m, self.frequency_ghz)
