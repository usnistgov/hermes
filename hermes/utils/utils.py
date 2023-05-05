"""Misc Utils."""
from numpy import ndarray
import numpy as np


def default_ndarray():
    """Default factory for ndarray."""
    return np.array([])


def find_new_locations(w: ndarray, m: ndarray) -> ndarray:  # pylint: disable=C0103
    """Find new locations to collect measures on."""
    return np.array([x for x in w if x not in m])
