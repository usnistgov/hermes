"""Active Learning pipelines."""

from dataclasses import field
from typing import Optional, Type

from pydantic.dataclasses import dataclass as typesafe_dataclass

import hermes.pipelines.base as base
from hermes.archive import Archiver
from hermes.loopcontrols import Initializer


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    validate_assignment = True


class AL(base.Pipeline):
    """Metaclass for AL."""

    init_method: Type[Initializer] = None
    archive_method: Type[Archiver] = None
    data_analysis: Type[base.Pipeline] = field(init=False)
    # TODO own data archive that takes/returns dict


# class ALClusterClassification(AL):
#     """Active Learning ClusterClassification Class."""

#     data_analysis: Optional[Type[base.ClusterClassification]] = None


# class ALMeta(Pipeline):
#     """Metaclass for AL."""

#     init_method: Type[Initializer] = None
#     archive_method: Type[Archiver] = None
#     data_analysis: Type[base.Pipeline] = field(init=False)

# @typesafe_dataclass(config=_Config)
# class ClusterClassification(ALMeta, base.ClusterClassification):
#     """Active Learning ClusterClassification Class."""

#     pass


# cluster_method = hermes.clustering.RBPots(
#     measurements=measurements,
#     measurements_distance_type=hermes.distance.CosineDistance(),
#     measurements_similarity_type=hermes.similarity.SquaredExponential(lengthscale=0.01),
#     locations=locations,
#     resolution=0.2,
# )
# classification_method = hermes.classification.HeteroscedasticGPC(
#     locations=locations,
#     labels=cluster_method.labels,
#     domain=domain,
#     probabilities=cluster_method.probabilities,
#     measured_indexes=indexes,
#     indexes=indexes,
# )
