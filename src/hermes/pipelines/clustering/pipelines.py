"""Pipelines for Clustering."""

from pydantic.dataclasses import dataclass

from hermes.pipelines.base import Classfication, Clustering


@dataclass
class ClusteringPipeline(Clustering):
    """Classification Pipeline."""

    pass
