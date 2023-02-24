from dataclasses import dataclass
from hermes.distance.base import BaseDS

import numpy as np
from sklearn.metrics.pairwise import haversine_distances, cosine_distances

# TODO dynamic time warping dtw-python 1.3.0
from scipy.stats import wasserstein_distance  # TODO
from sklearn.metrics import pairwise_distances

from orix.quaternion.orientation import Misorientation
from orix.quaternion import symmetry


@dataclass
class BaseDistance(BaseDS):
    def _needs_locations(self):
        return self.needs_locations


# TODO utility function for ops on X, Y if not same dim
@dataclass
class EuclidianDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="euclidean")
        return distance_matrix


# TODO classes for imported sklearn


@dataclass
class CosineDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric="cosine")
        return distance_matrix


@dataclass
class PNormDistance(BaseDistance):
    # Unless otherwise specifed P is set to 2 (Equivelent to Euclidean Distance)
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
            print("ekse")
            exponentiation = difference**self.P
            sums = np.sum(exponentiation, axis=1)
            distance = sums ** (1 / self.P)
        print(f"{difference}\n")
        print(distance)
        distance_matrix = distance.reshape(X.shape[0], Y.shape[0])
        return distance_matrix


# TODO Orientation distance
