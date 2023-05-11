import numpy as np
import pandas as pd
from dataclasses import dataclass



@dataclass
class Instrument:
    """Base level class for communicating with instruments."""
    
@dataclass
class Diffractometer(Instrument):
    """Class for diffractometer instruments"""
    
@dataclass
class Powder_Diffractometer(Diffractometer):
    """Class for Powder (aka 1D) diffractometer instruments.
    Typically expect the sample to be a combinatorial wafer (each location has a known, pre-determined composition), 
    but ignore the composition information for general samples."""
    
    #Location for example data used in simulation mode:
    wafer_directory:str = None
    #Location for wafer coordinates file (tab delimited .txt)
    wafer_coords_file:str = None
    #Location for wafer composition file (tab delimited .txt)
    wafer_composition_file:str = None
    #Location for XRD measurements file (tab delimited .txt)
    wafer_xrd_file:str = None
        
    def load_sim_data(self):
        self.xy_locations = pd.read_table(self.wafer_directory+self.wafer_coords_file)
        self.compositions = pd.read_table(self.wafer_directory+self.wafer_composition_file)

        self.xrd_measurements = pd.read_table(self.wafer_directory+self.wafer_xrd_file)
        
    def simulated_move_and_measure(self, compositions_locations):
        """Move (in composition-space) to new locations
        and return the XRD measurements."""
        
        #If the data for simulation mode hasn't been loaded, load it. 
        if ~hasattr(self, "xy_locations"):
            self.load_sim_data()
        
        indexes = []
        for comp in compositions_locations:
            index = self.compositions[self.compositions.to_numpy()==comp].index[0]
            indexes.append(index)
            
        measurements = self.xrd_measurements.iloc[indexes,:].to_numpy()
        return measurements
    
    def simulated_move_and_measure(self, compositions_locations):
        """Move (in composition-space) to new locations
        and return the XRD measurements."""
        
        #If the data for simulation mode hasn't been loaded, load it. 
        if ~hasattr(self, "xy_locations"):
            self.load_sim_data()
        
        indexes = []
        for comp in compositions_locations:
            index = self.compositions[self.compositions.to_numpy()==comp].index[0]
            indexes.append(index)
            
        measurements = self.xrd_measurements.iloc[indexes,:].to_numpy()
        return measurements
    
    @property
    def sim_wafer_coords(self):
        """Get all of the possible coordinates of the sample"""
        return self.xy_locations.to_numpy()
    
    @property
    def sim_composition_domain(self):
        """Get the entire domain in composition space."""
        components = self.compositions.columns.to_list()
        fractions = self.compositions.to_numpy()
        return components, fractions
    
    @property
    def sim_two_theta_space(self):
        """Get the 2Theta values of the XRD measurements in degrees"""
        two_theta = self.xrd_measurements.columns.to_numpy().astype(float)
        return two_theta
        
        
    
    
@dataclass
class CHESS_QM2_Beamline(Powder_Diffractometer):
    """Class for the QM2 diffractometer at CHESS"""
    
    simulation: bool = False 
        
    def load_wafer_file(self):
        self.xy_locations = pd.read_table(self.wafer_directory+self.wafer_coords_file)
        self.compositions = pd.read_table(self.wafer_directory+self.wafer_composition_file)
        self.xrd_measurements = pd.read_table(self.sim_load_dir+self.wafer_xrd_file)
        
    def move_and_measure(self, compositions_locations):
        """Move (in composition-space) to new locations
        and return the XRD measurements."""
        
        if self.simulation:
            measurements = self.simulated_move_and_measure(compositions_locations)
            
        else:
            print("not implemented yet")
            
            #Convert compostion to wafer coordinates
            #For each location:
                #Move to wafer coordinates
                #Measure
                #Reduce
                #Concatenate measurements
        
        return measurements
    
    @property
    def wafer_coords(self):
        """Get all of the possible coordinates of the sample"""
        return self.xy_locations.to_numpy()
    
    @property
    def composition_domain(self):
        """Get the entire domain in composition space."""
        components = self.compositions.columns.to_list()
        fractions = self.compositions.to_numpy()
        return components, fractions
    
    @property
    def two_theta_space(self):
        """Get the 2Theta values of the XRD measurements in degrees"""
        two_theta = self.xrd_measurements.columns.to_numpy().astype(float)
        return two_theta
        
    