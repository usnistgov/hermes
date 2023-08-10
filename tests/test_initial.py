# type: ignore
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import hermes

SIM_LOAD_DIR = Path(__file__).parent.joinpath("resources")
WAFER_COORDS_FILE = "XY_Coordinates_177.txt"
WAFER_COMPOSITION_FILE = "CombiView_Format_GeSbTe_Composition.txt"
WAFER_XRD_FILE = (
    "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt"
)

instrument = hermes.instruments.CHESSQM2Beamline(
    simulation=True,
    wafer_directory=SIM_LOAD_DIR,
    wafer_coords_file=WAFER_COORDS_FILE,
    wafer_composition_file=WAFER_COMPOSITION_FILE,
    wafer_xrd_file=WAFER_XRD_FILE,
    sample_name="This is a great name",
)


def test_create_QM2():
    instrument = hermes.instruments.CHESSQM2Beamline(
        simulation=True,
        wafer_directory=SIM_LOAD_DIR,
        wafer_coords_file=WAFER_COORDS_FILE,
        wafer_composition_file=WAFER_COMPOSITION_FILE,
        wafer_xrd_file=WAFER_XRD_FILE,
    )
    assert instrument.simulation == True


@pytest.fixture
def QM2():
    instrument = hermes.instruments.CHESSQM2Beamline(
        simulation=True,
        wafer_directory=SIM_LOAD_DIR,
        wafer_coords_file=WAFER_COORDS_FILE,
        wafer_composition_file=WAFER_COMPOSITION_FILE,
        wafer_xrd_file=WAFER_XRD_FILE,
    )
    return instrument


@dataclass
class RandomStart:
    domain: np.ndarray
    start_measurements: int

    def initialize(self):
        indexes = np.arange(0, self.domain.shape[0])
        permute = np.random.permutation(indexes)

        next_indexes = permute[0 : self.start_measurements]

        return next_indexes


def test_init_method(QM2):
    init_method = RandomStart(QM2.composition_domain_2d, 10)
    test = init_method.initialize()
    assert test.size == 10


@pytest.fixture(scope="session")
def TestArray(QM2):
    init_method = RandomStart(QM2.composition_domain_2d, 10)
    test = init_method.initialize()
    return test


def test_domains(QM2):
    assert isinstance(QM2.composition_domain_2d, np.ndarray)


def test_domains_3d(QM2):
    assert isinstance(QM2.composition_domain[1], np.ndarray)


def test_prepare_al(QM2):
    domain_2d = QM2.composition_domain_2d
    domain_3d = QM2.composition_domain[1]

    # Choose the initial locations
    start_measurements = 11
    initialization_method = RandomStart(QM2.composition_domain_2d, start_measurements)
    next_indexes = initialization_method.initialize()
    next_locations = domain_2d[next_indexes]

    # Initialize containers for locations and measurements:
    locations = np.array([]).reshape(-1, domain_2d.shape[1])
    measurements = np.array([]).reshape(-1, QM2.sim_two_theta_space.shape[0])
    assert True


@pytest.fixture
def domain_2d(QM2):
    domain_2d = QM2.composition_domain_2d
    return domain_2d


@pytest.fixture
def domain_3d(QM2):
    domain_3d = QM2.composition_domain[1]
    return domain_3d


def test_al_loops(QM2, domain_2d, domain_3d):
    start_measurements = 11
    initialization_method = RandomStart(QM2.composition_domain_2d, start_measurements)
    next_indexes = initialization_method.initialize()
    next_locations = domain_2d[next_indexes]
    domain = QM2.xy_locations.to_numpy()
    # Get the indexes in the domain:
    indexes = np.arange(0, domain.shape[0])

    # Initialize containers for locations and measurements:
    locations = np.array([]).reshape(-1, domain_2d.shape[1])
    measurements = np.array([]).reshape(-1, QM2.sim_two_theta_space.shape[0])
    for n in range(2):
        next_measurements = QM2.move_and_measure(domain_3d[next_indexes])

        locations = np.append(locations, next_locations, axis=0)
        measurements = np.append(measurements, next_measurements, axis=0)

        cluster_method = hermes.clustering.RBPots(
            measurements=measurements,
            measurements_distance_type=hermes.distance.CosineDistance(),
            measurements_similarity_type=hermes.similarity.SquaredExponential(
                lengthscale=0.1
            ),
            locations=locations,
            resolution=0.2,
        )
        # cluster_method.form_graph()
        cluster_method.cluster()
        cluster_method.get_local_membership_prob()

        # TODO use qm2 example

        classification_method = hermes.classification.HeteroscedasticGPC(
            locations=locations,
            labels=cluster_method.labels,
            domain=domain,
            probabilities=cluster_method.probabilities,
            measured_indexes=indexes,
            indexes=indexes,
        )

        classification_method.train()
        classification_method.predict_unmeasured()

        acquisition_method = hermes.acquire.PureExplore(
            classification_method.unmeasured_locations,
            classification_method.mean_unmeasured,
            classification_method.var_unmeasured,
        )

        next_locations = acquisition_method.calculate()
        next_indexes = classification_method.return_index(next_locations)
        assert True
