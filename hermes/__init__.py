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
from . import archive
from . import base
from . import classification
from . import clustering
from . import distance
from . import instruments
from . import pipelines
from . import schemas
from . import similarity
from . import utils



__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = ["archive",
           "base",
           "classification",
           "clustering",
           "distance"
           "instruments",
           "pipelines",
           "schemas",
           "similarity",
           "utils"]
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
