"""Top-level package for Hermes Python."""

from hermes.distance import BaseDistance, EuclidianDistance, CosineDistance, PNormDistance

__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"
__all__ = [
        "BaseDistance",
        "EuclidianDistance",
        "CosineDistance",
        "PNormDistance",
        "compute_distance",
        ]