"""Top-level package for Hermes Python."""

# For various convience functions
# For building autonomous loops
# For controling instruments
# For data analysis tasks
# For comparison measures
# For archiving results
from hermes.bailiwick import PhaseID, SpectralProbability
from hermes.MLTasks import acquisition, classification, clustering, distance, similarity

__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = [
    "archive",
    "distance",
    "similarity",
    "clustering",
    "classification",
    "acquisition",
    "instruments",
    "pipelines",
    "PhaseID",
    "SpectralProbability",
    "utils",
]
