from numpy import ndarray
from functools import singledispatch
from sklearn.metrics.pairwise import haversine_distances, cosine_distances
from typing import Optional
from ..distance import HaversineDistance

# do it with only one ndarray


@singledispatch
def compute_similarity(tp: similarity.abc, array: ndarray):
    # default computation
    return similarity


@compute_similarity.register  # only one method for two arrays
def _(tp: similarity.xyz, locations: ndarray, measurements: ndarray):
    return similarity


@singledispatch
def compute_distance(tp: distance.abc, array: ndarray):
    # default computation
    return distance


@compute_distance.register  # only one method for two arrays
def _(tp: HaversineDistance, X: ndarray, Y: Optional[ndarray]=None):
    return haversine_distances(X, Y)


@compute_distance.register  # only one method for two arrays
def _(tp: NewDistance, X: ndarray, Y: Optional[ndarray]=None):
    return NewDistance.compute(X, Y)
# from hermes.measurements import euclidian add precomputed
