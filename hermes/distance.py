"""Distance classes and methods."""
from dataclasses import dataclass

import numpy as np

# TODO dynamic time warping dtw-python 1.3.0 ASK
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances

from hermes.base import BaseDS

# from orix.quaternion.orientation import Misorientation
# from orix.quaternion import symmetry


@dataclass
class BaseDistance(BaseDS):
    """Base class for distance types."""


# TODO utility function for ops on X, Y if not same dim
@dataclass
class EuclidianDistance(BaseDistance):
    """Euclidian Distance. L2Norm."""

    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="euclidean")
        return distance_matrix


# TODO classes for imported sklearn
"""
From sklearn valid tp's are: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’],
From scipy valid tp's are: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
"""


@dataclass
class CosineDistance(BaseDistance):
    """Cosine Distance."""

    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="cosine")
        return distance_matrix


@dataclass
class CityBlockDistance(BaseDistance):
    """Sklearn wrapper."""

    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="cityblock")
        return distance_matrix


@dataclass
class L1Distance(BaseDistance):
    """Sklearn wrapper."""

    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="l1")
        return distance_matrix


@dataclass
class L2Distance(BaseDistance):
    """Sklearn wrapper."""

    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="l2")
        return distance_matrix


# TODO ASK questions abut this


@dataclass
class PNormDistance(BaseDistance):
    """PNorm Distance. Unless otherwise specified, p = 2 - equivalent to Euclidian Distance."""

    P: float = 2.0

    def calculate(self, X, Y=None):
        """Calculate the L-Norm with degree P.
        P = 2 is equivalent to Euclidean
        P = 1 is equivalent to Manhattan aka Taxi Cab aka City Block"""
        if Y is None:
            Y = X
        difference = X - Y.T
        if self.P == 0:
            raise ZeroDivisionError("Division by Zero: Undefined")
        elif self.P == np.inf:
            stack = np.dstack((X, Y))
            distance = np.max(np.abs(stack), axis=2)
        else:
            print("else")
            exponentiation = difference**self.P
            sums = np.sum(exponentiation, axis=1)
            distance = sums ** (1 / self.P)
        print(f"{difference}\n")
        print(distance)
        distance_matrix = distance.reshape(X.shape[0], Y.shape[0])
        return distance_matrix


@dataclass
class WassersteinDistance(BaseDistance):
    """Wrapper for Wasserstein Distance from scipy."""

    @classmethod
    def calculate(cls, X, Y=None, X_weights=None, Y_weights=None):
        return wasserstein_distance(X, Y, X_weights, Y_weights)


@dataclass
class HaversineDistance(BaseDistance):
    """Wrapper for Haversine Distance from sklearn."""

    @classmethod
    def calculate(cls, X, Y=None):
        return haversine_distances(X, Y)


# TODO Orientation distance
