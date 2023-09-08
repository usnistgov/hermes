# pylint: disable=C0103, disable=W0221
"""Distance classes and methods."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# TODO dynamic time warping dtw-python 1.3.0 ASK
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances

from hermes.base import BaseDS

# from orix.quaternion.orientation import Misorientation
# from orix.quaternion import symmetry

from hermes.utils import _default_ndarray


@dataclass
class BaseDistance(BaseDS):
    """Base class for distance types."""

    def calculate(self, X: np.ndarray, Y: Optional[np.ndarray], metric: str):
        """Calculate distance."""
        distance_matrix = pairwise_distances(X, Y, metric)
        return distance_matrix

    def __str__(self) -> str:
        if self.Y:  # type: ignore
            return f"{self.__class__.__name__}(X={self.X.shape}, Y={self.Y.shape})"
        return f"{self.__class__.__name__}(X={self.X.shape}, Y=None)"

    def __repr__(self) -> str:
        if self.Y:  # type: ignore
            return f"{self.__class__.__name__}(X={self.X.shape}, Y={self.Y.shape})"
        return f"{self.__class__.__name__}(X={self.X.shape}, Y=None)"


@dataclass
class EuclideanDistance(BaseDistance):
    """Euclidean Distance. L2Norm."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Euclidean Distance."""
        return super().calculate(self.X, self.Y, "euclidean")

    def __repr__(self) -> str:
        return super().__repr__()


# From sklearn valid tp"s are: ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
# From scipy valid tp's are: ["braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]


@dataclass
class CosineDistance(BaseDistance):
    """Cosine Distance."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Cosine Distance."""
        return super().calculate(self.X, self.Y, "cosine")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class CityBlockDistance(BaseDistance):
    """CityBlock Distance."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate CityBlock Distance."""
        return super().calculate(self.X, self.Y, "cityblock")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class L1Distance(BaseDistance):
    """L1 Distance."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate L1 Distance."""
        return super().calculate(self.X, self.Y, "l1")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class L2Distance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate L2 Distance."""
        return super().calculate(self.X, self.Y, "l2")

    def __repr__(self) -> str:
        return super().__repr__()


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


@dataclass
class BrayCurtisDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate BrayCurtis Distance."""
        return super().calculate(self.X, self.Y, "braycurtis")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class CanberraDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Canberra Distance."""
        return super().calculate(self.X, self.Y, "canberra")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class ChebyshevDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Chebyshev Distance."""
        return super().calculate(self.X, self.Y, "chebyshev")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class CorrelationDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Correlation Distance."""
        return super().calculate(self.X, self.Y, "correlation")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class DiceDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Dice Distance."""
        return super().calculate(self.X, self.Y, "dice")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class HammingDistance(BaseDistance):
    """Sklearn wrapper."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate Hamming Distance."""
        return super().calculate(self.X, self.Y, "hamming")

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class PNormDistance(BaseDistance):
    """PNorm Distance. Unless otherwise specified, p = 2 - equivalent to Euclidian Distance."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)
    P: float = 2.0

    def calculate(self):
        """Calculate the L-Norm with degree P.
        P = 2 is equivalent to Euclidean
        P = 1 is equivalent to Manhattan aka Taxi Cab aka City Block"""
        if self.Y is None:
            self.Y = self.X
        difference = self.X - self.Y.T
        if self.P == 0:
            raise ZeroDivisionError("Division by Zero: Undefined")
        if self.P == np.inf:
            stack = np.dstack((self.X, self.Y))
            distance = np.max(np.abs(stack), axis=2)
        else:
            print("else")
            exponentiation = difference**self.P
            sums = np.sum(exponentiation, axis=1)
            distance = sums ** (1 / self.P)
        print(f"{difference}\n")
        print(distance)
        distance_matrix = distance.reshape(self.X.shape[0], self.Y.shape[0])
        return distance_matrix


@dataclass
class WassersteinDistance(BaseDistance):
    """Wrapper for Wasserstein Distance from scipy."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)
    X_weights: Optional[np.ndarray] = field(init=False, default=None)
    Y_weights: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate WassersteinDistance."""
        return wasserstein_distance(self.X, self.Y, self.X_weights, self.Y_weights)


@dataclass
class HaversineDistance(BaseDistance):
    """Wrapper for Haversine Distance from sklearn."""

    X: np.ndarray = field(default_factory=_default_ndarray)
    Y: Optional[np.ndarray] = field(init=False, default=None)

    def calculate(self):
        """Calculate HaversineDistance."""
        return haversine_distances(self.X, self.Y)


# TODO Orientation distance
