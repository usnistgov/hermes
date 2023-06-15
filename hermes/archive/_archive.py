from dataclasses import dataclass

import numpy as np


@dataclass
class Archiver:
    """Base class for archiving"""


@dataclass
class JSON(Archiver):
    """Class for writing JSON's"""

@dataclass
class CombiMappingModels(JSON):
    """For archiving the models analyze combi wafers with:
    Instruement
    Clustering Model
    Classification Model"""
    """stub"""