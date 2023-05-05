"""Definition of Data Pipeline Class."""
from dataclasses import dataclass
from typing import Type, Optional

import numpy as np

from hermes.Base import BaseDataPipeline, BaseArchive
from hermes.Base.analysis import Analysis
from hermes.distance import BaseDistance
from hermes.similarity import BaseSimilarity


@dataclass
class DataPipeline(BaseDataPipeline):
    """Data Pipeline Class."""

    def __init__(
        self,
        locations: np.ndarray,
        distance: Type[BaseDistance],
        similarity: Type[BaseSimilarity],
        analysis: list[Type[Analysis]],  # ordered
        archive: Type[BaseArchive],  # JSON, SQLlite, Cordra
    ):
        pass
