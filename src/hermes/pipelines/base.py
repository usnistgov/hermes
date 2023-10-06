# pylint: disable=R0913
"""Definition of Data Pipeline Class."""
from typing import ForwardRef, Optional, Type

import numpy as np
from pydantic.dataclasses import dataclass as typesafe_dataclass

from hermes.archive import Archiver
from hermes.classification import Classification as ClassificationMethod
from hermes.clustering import Cluster as ClusterMethod
from hermes.instruments import Instrument

# Pipeline = ForwardRef("Pipeline")
# Convergence = ForwardRef("Convergence")
RegressionMethod = ForwardRef("RegressionMethod")


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    validate_assignment = True


@typesafe_dataclass(config=_Config)
class Pipeline:
    """Custom Pipeline Class."""

    instrument: Optional[Type[Instrument]] = None
    domain: Optional[np.ndarray] = None


@typesafe_dataclass(config=_Config)
class Cluster(Pipeline):
    """Cluster Pipeline."""

    cluster_method: Type[ClusterMethod] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class Classification(Pipeline):
    """Classification Pipeline."""

    classification_method: Type[ClassificationMethod] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class ClusterClassification(Pipeline):
    """Cluster-Classification Pipeline."""

    cluster_method: Type[Cluster] = None
    classification_method: Type[Classification] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class Regression(Pipeline):
    """Regression Pipeline."""

    regression_method: Type[RegressionMethod] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class ClusterClassificationRegression(Pipeline):
    """Cluster-Classification-Regression Pipeline."""

    cluster_method: Type[ClusterMethod] = None
    classification_method: Type[ClassificationMethod] = None
    regression_method: Type[RegressionMethod] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class ClassificationRegression(Pipeline):
    """Classification-Regression Pipeline."""

    classification_method: Type[ClassificationMethod] = None
    regression_method: Type[RegressionMethod] = None
    archiver: Type[Archiver] = None


@typesafe_dataclass(config=_Config)
class ClusterRegression(Pipeline):
    """Cluster-Regression Pipeline."""

    cluster_method: Type[ClusterMethod] = None
    regression_method: Type[RegressionMethod] = None
    archiver: Type[Archiver] = None


# subclass into: active learning, clustering classifi
# TODO modify cluster: so that we only do cluster_method.cluster()
# clustering classification: for this subclass of pipelines alwasy this order,
# output: args for acquisition ( this makes it AL)
#
#

# @typesafe_dataclass
# class PhaseMappingPipeline(Pipeline):
