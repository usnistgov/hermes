"""Top-level package for Hermes Python."""

# For various convience functions
# For builing autonomous loops
# For controling instruments
# For data analysis tasks
# For comparison measures
# For archiving results
from . import (
    acquire,
    archive,
    base,
    clustering,
    joint,
    distance,
    instruments,
    loopcontrols,
    similarity,
    utils,
)

__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = [
    "base",
    "archive",
    "distance",
    "similarity",
    "clustering",
    "joint",
    "acquire",
    "instruments",
    "loopcontrols",
    "utils",
]
