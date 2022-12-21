from numpy import ndarray
from functools import singledispatch
from typing import Optional, Any
from ..distance import BaseDistance, EuclidianDistance, CosineDistance, PNormDistance, OrientationDistance
# do it with only one ndarray



@singledispatch
def compute_distance(tp: Any):
    # default computation
    raise ValueError("Please specify a distance type")


@compute_distance.register  # only one method for two arrays
def _(tp: EuclidianDistance, X: ndarray, Y: Optional[ndarray]=None):
    return EuclidianDistance.calculate(X, Y)

@compute_distance.register  # only one method for two arrays
def _(tp: CosineDistance, X: ndarray, Y: Optional[ndarray]=None):
    return CosineDistance.calculate(X, Y)

@compute_distance.register  # only one method for two arrays
def _(tp: PNormDistance, X: ndarray, Y: Optional[ndarray]=None, P:Optional[int]=None):
    if not P:
        return PNormDistance.calculate(X, Y)
    return PNormDistance.calculate(X, Y, P)
# from hermes.measurements import euclidian add precomputed
