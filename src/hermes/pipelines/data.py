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
from hermes.loopcontrols import BaseLoopControl, Initializer
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


@typesafe_dataclass
class Pipeline:
    """Custom Pipeline Class."""

    instrument: Optional[Type[Instrument]] = None
    domain: Optional[np.ndarray] = None
    init_method: Optional[Type[Initializer]] = None
    data_analysis: Optional[Pipeline] = None
    al_loops: Optional[int] = None
    convergence: Optional[Convergence] = None
    cluster_method: Optional[Type[Cluster]] = None
    classification_method: Optional[Type[Classification]] = None
    acquisition_method: Optional[Type[Acquisition]] = None
    archiver = Optional[Type[Archiver]] = None


# subclass into: active learning, clustering classifi
# TODO modify cluster: so that we only do cluster_method.cluster()
# clustering classification: for this subclass of pipelines alwasy this order,
# output: args for acquisition ( this makes it AL)
#
#

mypipelinePipeline(instrument=)
mypipeline.method="active learning"
mypipeline.data_analysis=[ClusterClassification(), Acquire(), Archive()]
# @typesafe_dataclass
# class PhaseMappingPipeline(Pipeline):
