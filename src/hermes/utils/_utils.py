"""Misc Utils."""
from typing import Any

import numpy as np
from numpy import ndarray


def default_ndarray():
    """Default factory for ndarray."""
    return np.array([])


def find_new_locations(w: ndarray, m: ndarray) -> ndarray:  # pylint: disable=C0103
    """Find new locations to collect measures on."""
    return np.array([x for x in w if x not in m])


def _check_attr(obj: Any, attr: str) -> None:
    """Raise error if attr is None."""
    if getattr(obj, attr) is None:
        raise AttributeError(f"instance does not have {attr}")
