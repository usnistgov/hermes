from dataclasses import dataclass
from hermes.base import BaseDS

import numpy as np


@dataclass
class BaseSimilarity(BaseDS):
    """Base Class for Similarity measures."""

    pass


@dataclass
class SquaredExponential(BaseSimilarity):
    """Convert distances to similarities by:
       sigma^2 * exp(-distance^2/(2*lengscale^2))"""
    
    distance_matrix: np.ndarray = field(default_factory=_default_ndarray)

    #Default variance and lengthscales
    variance:float = 1.0
    lengthscale:float = 1.0
        
    def calculate(cls):
        similarities = cls.variance * np.exp(-(cls.distance_matrix/cls.lengthscale)**2)
        return similarities