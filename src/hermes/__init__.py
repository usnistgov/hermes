"""Top-level package for Hermes Python."""

# For various convience functions
# For building autonomous loops
# For controling instruments
# For data analysis tasks
# For comparison measures
# For archiving results
from hermes import (acquisition, 
                    archive, 
                    classification, 
                    clustering, 
                    distance, 
                    instruments, 
                    joint, 
                    latent_space, 
                    # multitask,
                    pipelines, 
                    similarity, 
                    utils
                    )


__author__ = """Austin McDannald, Brian DeCost, Camilo Velez"""
__email__ = "camilo.velezramirez@nist.gov"
__version__ = "0.1.0"

__all__ = [
    "acquisition",
    "archive",
    "classification",
    "clustering",
    "distance",
    "instruments",
    "joint",
    "latent_space",
    # "multitask",
    "pipelines",
    "similarity",
    "utils",
]
