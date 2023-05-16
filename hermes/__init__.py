"""Top-level package for Hermes Python."""

# from hermes.distance import (
#     BaseDistance,
#     EuclideanDistance,
#     CosineDistance,
#     PNormDistance,
# )
# from hermes.similarity import (
#     BaseSimilarity,
#     SquaredExponential,
# )
# from hermes.clustering import (
#     Cluster,
#     SpectralClustering,
#     ContiguousCommunityDiscovery,
#     ContiguousCluster,
#     ContiguousFixedKClustering,
# )
from . import distance


__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = ["distance"]
# __all__ = [
#     "BaseDistance",
#     "EuclideanDistance",
#     "CosineDistance",
#     "PNormDistance",
#     "BaseSimilarity",
#     "SquaredExponential",
#     "Cluster",
#     "SpectralClustering",
#     "ContiguousCommunityDiscovery",
#     "ContiguousCluster",
#     "ContiguousFixedKClustering",
# ]
