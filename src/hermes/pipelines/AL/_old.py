"""Active Learning pipelines."""

from abc import ABC, abstractmethod
from typing import Optional, Type

import numpy as np
from pydantic.dataclasses import dataclass as typesafe_dataclass

from hermes.pipelines.base import Pipeline


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    # validate_assignment = True


# @typesafe_dataclass(config=_Config)
# class AL:
#     """Active Learning Pipeline Class."""

#     data_analysis: Optional[Type[Pipeline]] = None

#     @classmethod
#     def __getattribute__(cls, name):
#         """Get attribute."""
#         print("getattribute", name)
#         if name == "ClusterClassification":
#             print("ClusterClassification")
#             return cls(data_analysis=Pipeline())
#         return object.__getattribute__(cls, name)

from hermes.pipelines.base import ClusterClassification, Pipeline


class ALMeta(type):
    """Metaclass for AL."""

    def __getattr__(cls, name):
        """Get attribute."""
        print("getattribute", name)
        if name == "ClusterClassification":
            print("ClusterClassification")
            print(cls.mro)
            # da = ClusterClassification()
            # return cls(data_analysis=())
            return 0
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


print(issubclass(ClusterClassification, Pipeline))


@typesafe_dataclass(config=_Config)
class AL(metaclass=ALMeta):
    """Active Learning Pipeline Class."""

    data_analysis: Optional[Type[Pipeline]] = None


from pathlib import Path

from hermes.archive import Archiver
from hermes.classification import Classification
from hermes.clustering import RBPots

SIM_LOAD_DIR = Path(__file__).parents[4].joinpath("tests", "resources")
WAFER_COORDS_FILE = "XY_Coordinates_177.txt"
WAFER_COMPOSITION_FILE = "CombiView_Format_GeSbTe_Composition.txt"
WAFER_XRD_FILE = (
    "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt"
)

import hermes
from hermes.instruments import CHESSQM2Beamline

instrument = CHESSQM2Beamline(
    simulation=True,
    wafer_directory=SIM_LOAD_DIR,
    wafer_coords_file=WAFER_COORDS_FILE,
    wafer_composition_file=WAFER_COMPOSITION_FILE,
    wafer_xrd_file=WAFER_XRD_FILE,
    sample_name="This is a great name",
    diffraction_space_bins=10000,
)


# domain = instrument.xy_locations.to_numpy()
domain = instrument.compositions
indexes = np.arange(0, domain.shape[0])

# Choose the initial locations
# start_measurements = 10
start_measurements = domain.shape[0]
initialization_method = hermes.loopcontrols.RandomStart(domain, start_measurements)
next_indexes = initialization_method.initialize()
print("next_indexes =", next_indexes)
next_locations = domain[next_indexes]
measured_indexes = np.array([])
locations = np.array([]).reshape(-1, domain.shape[1])
measurements = np.array([]).reshape(-1, instrument.diffraction_space_bins)


next_measurements = instrument.move_and_measure(next_indexes)
measured_indexes = np.append(measured_indexes, next_indexes)
locations = np.append(locations, next_locations, axis=0)
measurements = np.append(measurements, next_measurements, axis=0)

cluster_method = hermes.clustering.RBPots(
    measurements=measurements,
    measurements_distance_type=hermes.distance.CosineDistance(),
    measurements_similarity_type=hermes.similarity.SquaredExponential(lengthscale=0.01),
    locations=locations,
    resolution=0.2,
)
classification_method = hermes.classification.HeteroscedasticGPC(
    locations=locations,
    labels=cluster_method.labels,
    domain=domain,
    probabilities=cluster_method.probabilities,
    measured_indexes=indexes,
    indexes=indexes,
)
pipe1 = AL.ClusterClassification(
    cluster_method=cluster_method,
    classification_method=classification_method,
    archiver=Archiver,
)


2
