"""Active Learning pipelines."""

from typing import Optional, Type

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from hermes.archive import Archiver
from hermes.loopcontrols import Initializer
from hermes.pipelines import base

_Config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

@dataclass(config=_Config)
class AL(base.Pipeline):
    """Metaclass for AL."""

    init_method: Optional[Type[Initializer]] = None
    archive_method: Optional[Type[Archiver]] = None
    data_analysis: Optional[Type[base.Pipeline]] = None
    # TODO own data archive that takes/returns dict


@dataclass(config=_Config)
class ALClusterClassification(AL):
    """Active Learning ClusterClassification Class."""

    data_analysis: Optional[Type[base.ClusterClassification]] = None


@dataclass(config=_Config)
class ALRegression(AL):
    """Active Learning Regression Class."""

    data_analysis: Optional[Type[base.Regression]] = None


@dataclass(config=_Config)
class ALCluster(AL):
    """Active Learning Cluster Class."""

    data_analysis: Optional[Type[base.Cluster]] = None


@dataclass(config=_Config)
class ALClusterRegression(ALCluster):
    """Active Learning ClusterRegression Class."""

    data_analysis: Optional[Type[base.ClusterRegression]] = None


@dataclass(config=_Config)
class ALClassificationRegression(AL):
    """Active Learning ClassificationRegression Class."""

    data_analysis: Optional[Type[base.ClassificationRegression]] = None


@dataclass(config=_Config)
class ALClusterClassificationRegression(AL):
    """Active Learning ClusterClassificationRegression Class."""

    data_analysis: Optional[Type[base.ClusterClassificationRegression]] = None
