"""Top-level package for Hermes Python."""
from . import base

#For archiving results
from . import archive

#For comparison measures
from . import distance
from . import similarity

#For data analysis tasks
from . import clustering
from . import classification
from . import acquire

#For controling instruments
from . import instruments

#For builing autonomous loops
from . import pipelines

#For various convience functions
from . import utils


__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = [
    "base",
    "archive",
    "distance",
    "similarity",
    "clustering",
    "classification",
    "acquire",
    "instruments",
    "pipelines",
    "utils",
]
