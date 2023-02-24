from dataclasses import dataclass
from hermes.distance.base import BaseDS

import numpy as np


@dataclass
class BaseSimilarity(BaseDS):
    def _needs_locations(self):
        return self.needs_locations
