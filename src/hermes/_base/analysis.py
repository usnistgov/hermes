"""Base Class definition for intrinsic data analysis.
"""
from dataclasses import dataclass


@dataclass
class Analysis:
    """Base level class for analyising the properties of the data.
    All inputs are treated as features."""
