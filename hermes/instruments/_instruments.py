"""Instrument classes."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass as typesafedataclass

from hermes.utils import _check_attr

import pyspec

@dataclass
class Instrument:
    """Base level class for communicating with instruments."""


@dataclass
class Diffractometer(Instrument):
    """Class for diffractometer instruments"""


config1 = {"allow_arbitrary_types": True}


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True


@typesafedataclass(config=_Config)
class PowderDiffractometer(Diffractometer):
    """Class for Powder (aka 1D) diffractometer instruments.
    Typically expect the sample to be a combinatorial wafer (each location has a known, pre-determined composition),
    but ignore the composition information for general samples."""

    # Location for example data used in simulation mode:
    wafer_directory: Path
    # Location for wafer coordinates file (tab delimited .txt)
    wafer_coords_file: Path  # relative to wafer_directory
    # Location for wafer composition file (tab delimited .txt)
    wafer_composition_file: Path  # relative to wafer_directory
    # Location for XRD measurements file (tab delimited .txt)
    wafer_xrd_file: Path  # relative to wafer_directory

    xy_locations: Optional[pd.DataFrame] = field(init=False, default=None)
    compositions: Optional[pd.DataFrame] = field(init=False, default=None)
    xrd_measurements: Optional[pd.DataFrame] = field(init=False, default=None)

    def load_sim_data(self):
        """Load simulated data."""
        self.xy_locations = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_coords_file)
        )
        self.compositions = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_composition_file)
        )

        self.xrd_measurements = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_xrd_file)
        )

    def simulated_move_and_measure(self, compositions_locations):
        """Move (in composition-space) to new locations
        and return the XRD measurements."""

        # # If the data for simulation mode hasn't been loaded, load it.
        # if not self.xy_locations:
        #     self.load_sim_data()

        _check_attr(self, "compositions")
        _check_attr(self, "xrd_measurements")
        indexes = []
        for comp in compositions_locations:
            index = self.compositions[self.compositions.to_numpy() == comp].index[0]
            indexes.append(index)

        measurements = self.xrd_measurements.iloc[indexes, :].to_numpy()
        return measurements

    @property
    def sim_wafer_coords(self):
        """Get all of the possible coordinates of the sample"""
        _check_attr(self, "xy_locations")
        return self.xy_locations.to_numpy()

    @property
    def sim_composition_domain(self):
        """Get the entire domain in composition space."""
        _check_attr(self, "compositions")
        components = self.compositions.columns.to_list()
        fractions = self.compositions.to_numpy()
        return components, fractions

    @property
    def sim_two_theta_space(self):
        """Get the 2Theta values of the XRD measurements in degrees"""
        _check_attr(self, "xrd_measurements")
        two_theta = self.xrd_measurements.columns.to_numpy().astype(float)
        return two_theta

    @property
    def compositions_2d(self):
        """Converting the compostions from the 3D simplex to a 2D triangle
        NOTE: the triangle is smaller than the simplex by a factor of sqrt(2)."""
        _check_attr(self, "compositions")
        # In 3D space
        A_3d = np.array([1, 0, 0])
        B_3d = np.array([0, 1, 0])
        C_3d = np.array([0, 0, 1])

        # In 2D space
        A_2d = np.array([0, 0])  # A at the origin
        B_2d = np.array([1, 0])  # B at the x-axis = 1 point
        C_2d = np.array(
            [0.5, 0.5 * np.sqrt(3)]
        )  # C at the top of an equilateral triangle with the base along x of length 1.

        points = self.compositions.to_numpy()  # Read in the 3D compostions
        # Multiply 2D coordinates with the compositions for each component
        points_A = points[:, 0].reshape(-1, 1) * A_2d.reshape(1, -1)
        points_B = points[:, 1].reshape(-1, 1) * B_2d.reshape(1, -1)
        points_C = points[:, 2].reshape(-1, 1) * C_2d.reshape(1, -1)
        # Sum the coordinates for each component
        points_2d = points_A + points_B + points_C

        return points_2d
    
    
    # def compositions_2d_to_index(self, locations_2d):




@typesafedataclass(config=_Config)
class CHESSQM2Beamline(PowderDiffractometer):
    """Class for the QM2 diffractometer at CHESS"""

    simulation: bool = False

    def __post_init_post_parse__(self):
        if self.simulation:
            self.load_sim_data()

    def load_wafer_file(self):
        """Load the wafer file."""
        self.xy_locations = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_coords_file)
        )
        self.compositions = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_composition_file)
        )
        self.xrd_measurements = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_xrd_file)
        )

    def move_and_measure(self, compositions_locations):
        """Move (in composition-space) to new locations
        and return the XRD measurements."""

        if self.simulation:
            measurements = self.simulated_move_and_measure(compositions_locations)

        else:
            print(pyspec.__version__)
            raise NotImplementedError

            # Convert compostion to wafer coordinates
            # For each location:
            # Move to wafer coordinates
            # Measure
            # Reduce
            # Concatenate measurements

        return measurements

    @property
    def wafer_coords(self):
        """Get all of the possible coordinates of the sample"""
        _check_attr(self, "xy_locations")
        return self.xy_locations.to_numpy()

    @property
    def composition_domain(self):
        """Get the entire domain in composition space."""
        _check_attr(self, "compositions")
        components = self.compositions.columns.to_list()
        fractions = self.compositions.to_numpy()
        return components, fractions

    @property
    def composition_domain_2d(self):
        """Converting the compostions from the 3D simplex to a 2D triangle
        NOTE: the triangle is smaller than the simplex by a factor of sqrt(2)."""

        _check_attr(self, "compositions")
        # In 3D space
        A_3d = np.array([1, 0, 0])
        B_3d = np.array([0, 1, 0])
        C_3d = np.array([0, 0, 1])

        # In 2D space
        A_2d = np.array([0, 0])  # A at the origin
        B_2d = np.array([1, 0])  # B at the x-axis = 1 point
        C_2d = np.array(
            [0.5, 0.5 * np.sqrt(3)]
        )  # C at the top of an equilateral triangle with the base along x of length 1.

        points = self.composition_domain[1]  # Read in the 3D compostions
        # Multiply 2D coordinates with the compositions for each component
        points_A = points[:, 0].reshape(-1, 1) * A_2d.reshape(1, -1)
        points_B = points[:, 1].reshape(-1, 1) * B_2d.reshape(1, -1)
        points_C = points[:, 2].reshape(-1, 1) * C_2d.reshape(1, -1)
        # Sum the coordinates for each component
        points_2d = points_A + points_B + points_C

        return points_2d

    @property
    def two_theta_space(self):
        """Get the 2Theta values of the XRD measurements in degrees"""
        _check_attr(self, "xrd_measurements")
        two_theta = self.xrd_measurements.columns.to_numpy().astype(float)
        return two_theta

    # @property
    # def q_space(self):
    #     lambda = self.wavelength
    #     #some conversion here

    #     return q

    # @dataclass
    # class BrukerD8(PowderDiffractometer):
    #     #something
