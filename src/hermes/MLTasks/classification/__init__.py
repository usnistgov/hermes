"""Classifiers"""

import logging

from ._classification import Classification
from .predict_saved import SavedModel

GPC_INSTALLED = False

logger = logging.getLogger("hermes")
try:
    from ._classification import (
        GPC_INSTALLED,
        HeteroscedasticGPC,
        HomoscedasticGPC,
        SparceHeteroscedasticGPC,
        SparceHomoscedasticGPC,
    )
except ImportError:
    logger.warning("GPFlow not installed, GPC classifiers will not be available.")
else:
    GPC_INSTALLED = True

if GPC_INSTALLED:
    __all__ = [
        "Classification",
        "GPC_INSTALLED",
        "HomoscedasticGPC",
        "SparceHomoscedasticGPC",
        "HeteroscedasticGPC",
        "SparceHeteroscedasticGPC",
        "SavedModel",
    ]
else:
    __all__ = ["Classification", "SavedModel"]
