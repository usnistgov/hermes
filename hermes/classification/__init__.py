"""Classifiers"""

import logging

from ._classification import Classification

GPC = False

logger = logging.getLogger("hermes")
try:
    from ._classification import (
        GPC,
        HeteroscedasticGPC,
        HomoscedasticGPC,
        SparceHeteroscedasticGPC,
        SparceHomoscedasticGPC,
    )
except ImportError:
    logger.warning("GPFlow not installed, GPC classifiers will not be available.")
else:
    GPC = True

if GPC:
    __all__ = [
        "Classification",
        "GPC",
        "HomoscedasticGPC",
        "SparceHomoscedasticGPC",
        "HeteroscedasticGPC",
        "SparceHeteroscedasticGPC",
    ]
else:
    __all__ = ["Classification"]
