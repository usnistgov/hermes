from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import haversine_distances, cosine_distances
from sklearn.metrics import pairwise_distances

from orix.quaternion.orientation import Misorientation
from orix.quaternion import symmetry 

@dataclass
class BaseDistance:
    pass

@dataclass
class EuclidianDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric = "euclidean")
        return distance_matrix
    pass

class CosineDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        distance_matrix = pairwise_distances(X, Y, metric = "cosine")
        return distance_matrix
    pass 

class PNormDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None, P=2):
        '''Calculate the L-Norm with degree P.
        P = 2 is equivelent to Euclidean
        P = 1 is equivelent to Manhattan aka Taxi Cab aka City Block'''
        if Y is None:
            Y = X 
        difference = X - Y 
        if P == 0:
            raise ValueError("Divide by Zero: Undifined")
        elif P == np.inf:
            stack = np.dstack((X,Y))
            distance = np.max(np.abs(stack), axis = 2)
        else:
            exponentiation = difference**P 
            sums = np.sum(exponentiation, axis = 1)
            distance = sums**(1/P)
        distance_matrix = distance.reshape(X.shape[0], Y.shape[0])
        return distance_matrix
    pass

class OrientationDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None, sym_x, sym_y = None, convert_from_Euler = False):
        ''' Calcualte the Missorientation between 2 orientations considering the crystal symmetry.
        sym_x and sym_y are the Orix symmetry objects (e.g. Oh) for the measurements in X and Y, resepectively.
        X and Y are quaternions (must have shape = nx4)'''
        if Y is None:
            Y = X
        if (X.shape[1] != 4) | (Y.shape[1] != 4):
            raise ValueError("Data must be in Quaternions (therefore shape must be nx4")
        
        distance_matrix = Misorientation((~X).outer(Y)).set_symmetry(sym_x,sym_y).angle.data
        return distance_matrix
    pass 

class HaversineDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        haversine_distances(X, Y)

class CosineDistance(BaseDistance):
    @classmethod
    def calculate(cls, X, Y=None):
        cosine_distances(X, Y)

# wrapper earthdistance, cosine
