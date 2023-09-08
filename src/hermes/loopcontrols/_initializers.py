"""Classes of Autonomous Loop Initializers"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Initializer:
    """Base class of initializers"""


@dataclass
class RandomStart(Initializer):
    domain: np.ndarray
    start_measurements: int

    def initialize(self):
        indexes = np.arange(0, self.domain.shape[0])
        permute = np.random.permutation(indexes)

        next_indexes = permute[0 : self.start_measurements]

        return next_indexes
