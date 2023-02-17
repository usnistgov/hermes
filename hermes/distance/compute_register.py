from numpy import ndarray
from functools import singledispatch
from typing import Optional, Type
from . import BaseDistance, EuclidianDistance, CosineDistance, PNormDistance
# do it with only one ndarray



# @singledispatch
# def compute_distance(type_: Any):
#     # default computation
#     raise ValueError("Please specify a distance type")


def compute_distance(type_: Type[BaseDistance], X: ndarray, Y: Optional[ndarray]=None):
    return type_.calculate(X, Y)
 # type: ignore
# @compute_distance.register  # only one method for two arrays
# def _(type_: EuclidianDistance, X: ndarray, Y: Optional[ndarray]=None):
#     return EuclidianDistance.calculate(X, Y)

# @compute_distance.register  # only one method for two arrays
# def _(type_: CosineDistance, X: ndarray, Y: Optional[ndarray]=None):
#     return CosineDistance.calculate(X, Y)

# @compute_distance.register  # only one method for two arrays
# def _(type_: PNormDistance, X: ndarray, Y: Optional[ndarray]=None, P:Optional[int]=None):
#     return PNormDistance.calculate(X, Y)
# # from hermes.measurements import euclidian add precomputed
