"""Pipelines for Classification."""

from pydantic.dataclasses import dataclass

from hermes.pipelines.base import Classification

_Config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
# TODO _Config out


@dataclass
class ClassificationPipeline(Classification):
    """Classification Pipeline."""

    pass
