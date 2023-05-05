"""Distance classes and methods."""
from dataclasses import dataclass

import numpy as np

# TODO dynamic time warping dtw-python 1.3.0 ASK
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances

from hermes.Base import BaseDS

# from orix.quaternion.orientation import Misorientation
# from orix.quaternion import symmetry


@dataclass
class BaseDistance(BaseDS):
    """Base class for distance types."""


def sklearnwrapper(metric: str):
    """Decorator for sklearn distances wrappers."""

    def wrapper(cls):
        def calculate(cls, X, Y=None):  # type: ignore
            distance_matrix = pairwise_distances(X, Y, metric=metric)
            return distance_matrix

        cls.calculate = classmethod(calculate)
        return cls

    return wrapper


@sklearnwrapper("euclidian")
class EuclidianDistance(BaseDistance):
    """Euclidian Distance. L2Norm."""


# From sklearn valid tp"s are: ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
# From scipy valid tp's are: ["braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]


@sklearnwrapper("cosine")
class CosineDistance(BaseDistance):
    """Cosine Distance."""


@sklearnwrapper("cityblock")
class CityBlockDistance(BaseDistance):
    """CityBlock Distance."""


@sklearnwrapper("l1")
class L1Distance(BaseDistance):
    """L1 Distance."""


@sklearnwrapper("l2")
class L2Distance(BaseDistance):
    """Sklearn wrapper."""


_l = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


@sklearnwrapper("braycurtis")
class BrayCurtisDistance(BaseDistance):
    """Sklearn wrapper."""


@sklearnwrapper("canberra")
class CanberraDistance(BaseDistance):
    """Sklearn wrapper."""


@sklearnwrapper("chebyshev")
class ChebyshevDistance(BaseDistance):
    """Sklearn wrapper."""


@sklearnwrapper("correlation")
class CorrelationDistance(BaseDistance):
    """Sklearn wrapper."""


@sklearnwrapper("dice")
class DiceDistance(BaseDistance):
    """Sklearn wrapper."""


@sklearnwrapper("hamming")
class HammingDistance(BaseDistance):
    """Sklearn wrapper."""


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
