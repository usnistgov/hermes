# pylint: disable=R0913
"""Definition of Data Pipeline Class."""
from dataclasses import dataclass
from typing import ForwardRef, Optional, Type

import numpy as np
from pydantic.dataclasses import dataclass as typesafe_dataclass

from hermes.acquire import Acquisition

# from hermes.base.analysis import Analysis
from hermes.archive import Archiver
from hermes.classification import Classification
from hermes.clustering import Cluster
from hermes.distance import BaseDistance
from hermes.instruments import Instrument

# from hermes.loopcontrols import BaseLoopControl, Initializer
from hermes.similarity import BaseSimilarity

# @dataclass
# class DataPipeline(BaseDataPipeline):
#     """Data Pipeline Class."""


#     def __init__(
#         self,
#         locations: np.ndarray,
#         distance: Type[BaseDistance],
#         similarity: Type[BaseSimilarity],
#         analysis: list[Type[Analysis]],  # ordered
#         archive: Type[BaseArchive],  # JSON, SQLlite, Cordra
#     ):
#         pass
Pipeline = ForwardRef("Pipeline")
Convergence = ForwardRef("Convergence")


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    # validate_assignment = True


@typesafe_dataclass(config=_Config)
class Pipeline:
    """Custom Pipeline Class."""

    instrument: Optional[Type[Instrument]] = None
    domain: Optional[np.ndarray] = None


class ClusterClassification(Pipeline):
    """Cluster-Classification Pipeline."""

    cluster_method: Type[Cluster] = None
    classification_method: Type[Classification] = None
    archiver: Type[Archiver] = None

    def __init__(self, *args, **kwargs):
        print("ClusterClassification init")
        print(f"args->{args}")
        return super().__init__(*args, **kwargs)


# subclass into: active learning, clustering classifi
# TODO modify cluster: so that we only do cluster_method.cluster()
# clustering classification: for this subclass of pipelines alwasy this order,
# output: args for acquisition ( this makes it AL)
#
#

# @typesafe_dataclass
# class PhaseMappingPipeline(Pipeline):
