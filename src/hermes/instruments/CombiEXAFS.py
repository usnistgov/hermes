import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass as typesafedataclass

from hermes.utils import _check_attr

from ._instruments import DiffractometerForCombi

logger = logging.getLogger("hermes")

class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True

@typesafedataclass(config=_Config)
class CombiEXAFS(CombiPointByPoint):
    """Class for _____________

    Attributes
    ----------
    simulation : bool
        Flag for simulation mode. 
        Default is False
        
    sample_name : str
        Name of the sample to use for saving files. 
        
    reduction_dir : Path
        Location in the filesystem for a directory where the reduced data will be stored.
        
    Properties
    ----------
    diffraction_space()
        Values of the diffraction space (e.g q values or 2theta values) for each bin of the measurement. 
    
    Methods
    -------
     move_and_measure(self, indexes)
        Method to move to the location or composition specified by the index,
        and acquire XRD measurement there. 
    """   

    # Location for measurements file 
    wafer_xrd_file: Optional[Path] = None #TODO change for EXAFS
    absorber: Optional[np.ndarray] = None
    R_bins: Optional[np.ndarray] = None
    complex_component: Optional[list] = ["real", "imaginary"]

    xrd_measurements: Optional[Union[pd.DataFrame, np.ndarray]] = field(
        init=False, default=None
    )  # TODO: behavior of this when sim

    def load_sim_data(self):
        self.xrd_measurements = None
        if self.diffraction_space_bins is not None:
            self.xrd_measurements = np.array([]).reshape(
                -1, self.diffraction_space_bins
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
    def sim_two_theta_space(self):
        """Get the 2Theta values of the XRD measurements in degrees"""
        _check_attr(self, "xrd_measurements")
        two_theta = self.xrd_measurements.columns.to_numpy().astype(float)
        return two_theta

@typesafedataclass(config=_Config)
class EXAFSBeamline(CombiEXAFS):
    """Class for _____________

    Attributes
    ----------
    simulation : bool
        Flag for simulation mode. 
        Default is False
        
    sample_name : str
        Name of the sample to use for saving files. 
        
    reduction_dir : Path
        Location in the filesystem for a directory where the reduced data will be stored.
        
    Properties
    ----------
    diffraction_space()
        Values of the diffraction space (e.g q values or 2theta values) for each bin of the measurement. 
    
    Methods
    -------
     move_and_measure(self, indexes)
        Method to move to the location or composition specified by the index,
        and acquire XRD measurement there. 
    """
    

    simulation: bool = False

    sample_name: str = "some_sample_name"

    reduced_sample_dir: Path = "some/defult/path"

    def __post_init__(self):
        # load xy coordinates and compositions for discrete library sample
        self.load_wafer_data()

        if self.simulation:
            # load simulated XRD measurements
            self.load_sim_data()

    def load_sim_data(self):
        self.xrd_measurements = None
        if self.diffraction_space_bins is not None:
            self.xrd_measurements = np.array([]).reshape(
                -1, self.diffraction_space_bins
            )
        self.xrd_measurements = pd.read_table(
            self.wafer_directory.joinpath(self.wafer_xrd_file)
        )

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

    def move_and_measure(self, indexes) -> np.ndarray:
        """Move (in composition-space) to new locations
        and return the XRD measurements."""

        # print(f"{compositions_locations=}")

        if self.simulation:
            measurements = self.simulated_move_and_measure(indexes)

        else:
            
            #Start a container for the measurements
            # with shape [samples, absorber, R, complex_component]
            measurements = np.array([]).reshape(-1, self.absorber, self.R_bins, 2)

            # For each location:
            for idx, row in self.xy_locations.loc[indexes].iterrows():

                #TODO: Some thing!!!
                # measurement = Do some thing!!!

                measurements = np.concatenate(
                    (measurements, np.array(measurement).reshape(1, self.absorber, self.R_bins, 2)), axis=0
                )

        return measurements
