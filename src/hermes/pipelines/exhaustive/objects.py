"""Active Learning pipelines."""

from dataclasses import field
from typing import Optional, Type

from pydantic.dataclasses import dataclass as typesafe_dataclass

import hermes.pipelines.base as base


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    validate_assignment = True


@typesafe_dataclass(config=_Config)
class Exhaustive(base.Pipeline):
    """Metaclass for Exhaustive (For-loop)."""

    n: int = 1
    data_analysis: Type[base.Pipeline] = field(init=False)
    parallel: bool = False


# add comments to indicate below is for convenience


@typesafe_dataclass(config=_Config)
class ExhaustiveClusterClassification(Exhaustive):
    """Active Learning ClusterClassification Class."""

    data_analysis: Optional[Type[base.ClusterClassification]] = None


@typesafe_dataclass(config=_Config)
class ExhaustiveRegression(Exhaustive):
    """Active Learning Regression Class."""

    data_analysis: Optional[Type[base.Regression]] = None


@typesafe_dataclass(config=_Config)
class ExhaustiveCluster(Exhaustive):
    """Active Learning Cluster Class."""

    data_analysis: Optional[Type[base.Cluster]] = None


@typesafe_dataclass(config=_Config)
class ExhaustiveClassification(Exhaustive):
    """Active Learning Classification Class."""

    data_analysis: Optional[Type[base.Classification]] = None


@typesafe_dataclass(config=_Config)
class ExhaustiveClusterClassificationRegression(Exhaustive):
    """Active Learning ClusterClassificationRegression Class."""

    data_analysis: Optional[Type[base.ClusterClassificationRegression]] = None


@typesafe_dataclass(config=_Config)
class ExhaustiveClusterRegression(Exhaustive):
    """Active Learning ClusterRegression Class."""

    data_analysis: Optional[Type[base.ClusterRegression]] = None
