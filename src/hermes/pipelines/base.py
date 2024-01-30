# pylint: disable=R0913
"""Definition of Data Pipeline Class."""
from typing import ForwardRef, Optional, Type

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from hermes.archive import Archiver
from hermes.classification import Classification as ClassificationMethod
from hermes.clustering import Cluster as ClusterMethod
from hermes.instruments import Instrument

# Pipeline = ForwardRef("Pipeline")
# Convergence = ForwardRef("Convergence")
RegressionMethod = ForwardRef("RegressionMethod")


_Config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


@dataclass(config=_Config)
class Pipeline:
    """Custom Pipeline Class."""

    instrument: Optional[Type[Instrument]] = None
    domain: Optional[np.ndarray] = None


@dataclass(config=_Config)
class Cluster(Pipeline):
    """Cluster Pipeline."""

    cluster_method: Optional[Type[ClusterMethod]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class Classification(Pipeline):
    """Classification Pipeline."""

    classification_method: Optional[Type[ClassificationMethod]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class ClusterClassification(Pipeline):
    """Cluster-Classification Pipeline."""

    cluster_method: Optional[Type[Cluster]] = None
    classification_method: Optional[Type[Classification]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class Regression(Pipeline):
    """Regression Pipeline."""

    regression_method: Optional[Type[RegressionMethod]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class ClusterClassificationRegression(Pipeline):
    """Cluster-Classification-Regression Pipeline."""

    cluster_method: Optional[Type[ClusterMethod]] = None
    classification_method: Optional[Type[ClassificationMethod]] = None
    regression_method: Optional[Type[RegressionMethod]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class ClassificationRegression(Pipeline):
    """Classification-Regression Pipeline."""

    classification_method: Optional[Type[ClassificationMethod]] = None
    regression_method: Optional[Type[RegressionMethod]] = None
    archiver: Optional[Type[Archiver]] = None


@dataclass(config=_Config)
class ClusterRegression(Pipeline):
    """Cluster-Regression Pipeline."""

    cluster_method: Optional[Type[ClusterMethod]] = None
    regression_method: Optional[Type[RegressionMethod]] = None
    archiver: Optional[Type[Archiver]] = None


# subclass into: active learning, clustering classifi
# TODO modify cluster: so that we only do cluster_method.cluster()
# clustering classification: for this subclass of pipelines alwasy this order,
# output: args for acquisition ( this makes it AL)
#
#

# @dataclass
# class PhaseMappingPipeline(Pipeline):
