import numpy as np


def onecycle_cover(distance_matrix):
    """Return a deterministic one-cycle pairing order for guided mixup."""
    n = int(distance_matrix.shape[0])
    if n <= 1:
        return np.arange(n)
    return np.roll(np.arange(n), -1)
