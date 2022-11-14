from dataclasses import dataclass
from sklearn.metrics.pairwise import haversine_distances, cosine_distances


@dataclass
class BaseDistance:
    pass


@dataclass
class EuclidianDistance(BaseDistance):
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
