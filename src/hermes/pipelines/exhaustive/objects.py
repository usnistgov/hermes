"""Active Learning pipelines."""

from typing import Optional, Type

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from hermes.pipelines import base

_Config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


@dataclass(config=_Config)
class Exhaustive(base.Pipeline):
    """Metaclass for Exhaustive (For-loop)."""

    n: int = 1
    data_analysis: Optional[Type[base.Pipeline]] = None
    parallel: bool = False


# add comments to indicate below is for convenience


@dataclass(config=_Config)
class ExhaustiveClusterClassification(Exhaustive):
    """Active Learning ClusterClassification Class."""

    data_analysis: Optional[Type[base.ClusterClassification]] = None


@dataclass(config=_Config)
class ExhaustiveRegression(Exhaustive):
    """Active Learning Regression Class."""

    data_analysis: Optional[Type[base.Regression]] = None


@dataclass(config=_Config)
class ExhaustiveCluster(Exhaustive):
    """Active Learning Cluster Class."""

    data_analysis: Optional[Type[base.Cluster]] = None


@dataclass(config=_Config)
class ExhaustiveClassification(Exhaustive):
    """Active Learning Classification Class."""

    data_analysis: Optional[Type[base.Classification]] = None


@dataclass(config=_Config)
class ExhaustiveClusterClassificationRegression(Exhaustive):
    """Active Learning ClusterClassificationRegression Class."""

    data_analysis: Optional[Type[base.ClusterClassificationRegression]] = None


@dataclass(config=_Config)
class ExhaustiveClusterRegression(Exhaustive):
    """Active Learning ClusterRegression Class."""

    data_analysis: Optional[Type[base.ClusterRegression]] = None
