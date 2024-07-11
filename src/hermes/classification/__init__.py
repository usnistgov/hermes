"""Classifiers"""

import logging

from ._classification import GPC_INSTALLED, Classification
from .predict_saved import SavedModel

if GPC_INSTALLED:
    from ._classification import (
        HeteroscedasticGPC,
        HomoscedasticGPC,
        SparceHeteroscedasticGPC,
        SparceHomoscedasticGPC,
    )


logger = logging.getLogger("hermes")
# try:

# except ImportError:
#     logger.warning("GPFlow not installed, GPC classifiers will not be available.")
# else:
#     GPC_INSTALLED = True

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
