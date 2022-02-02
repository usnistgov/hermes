# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:59:16 2022

@author: asm6
"""

import numpy as np

import pandas as pd


def read_SEM_measurement_csv(Filename):
    #Read the csv into a pd.DataFrame
    Measurements = pd.read_csv(Filename,
                          names=(['X', 'Y', 'Mean_grainsize', 'Median_Grainsize', 'Mean_ln_grainsize', 'NONE']))
    #Get rid of the of the 'NONE' column
    Measurements = Measurements.iloc[:,0:-1]
    
    #Format the coordinates from text into floats
    X = Measurements.iloc[:,0]
    X = list(X)
    X = np.array([x.replace('[', '') for x in X])
    X = X.astype(np.float32())
    Y = Measurements.iloc[:,1]
    Y = list(Y)
    Y = np.array([y.replace(']', '') for y in Y])
    Y = Y.astype(np.float32())
    
    #Rewrite the coordiates as floats in the DataFrame
    Measurements.iloc[:,0]=X
    Measurements.iloc[:,1]=Y
    
    return Measurements

def read_wafer_coordinate_csv(Filename):
    #Read the csv into a pd.DataFrame
    #Skipping the 30-line preamble
    Locations = pd.read_csv(Filename,
                        names=(['ID', 'X', 'Y']),
                        skiprows=(30))
    #Extract the coordinates as a (entry by dimension) numpy array 
    test_locations = np.concatenate((Locations['X'].to_numpy().reshape(-1,1),
                                 Locations['Y'].to_numpy().reshape(-1,1)),
                                axis=1)
    return test_locations

def generate_dense_grid_in_circle(sparce_grid, new_1d_density):
    #'sparce_grid' is an (entry by dimension) numpy array
    # for locations on a circle, centered at the origin
    
    #Find the radius of the circle
    R = np.max(np.sqrt(sparce_grid[:,0]**2 + sparce_grid[:,1]**2))
    
    #Create the points along the X and Y axis
    # dense_x = np.linspace(np.min(sparce_grid[:,0]),np.max(sparce_grid[:,0]), new_1d_density)
    # dense_y = np.linspace(np.min(sparce_grid[:,1]),np.max(sparce_grid[:,1]), new_1d_density)
    dense_x = np.linspace(-R,R, new_1d_density)
    dense_y = np.linspace(-R,R, new_1d_density)
    
    #Generate a square grid
    dense_x_grid, dense_y_grid = np.meshgrid(dense_x, dense_y)



    #Calculate a mask for the square grid points inside the circle
    mask = dense_x_grid**2 + dense_y_grid**2 <= R**2

    #Mask the squre grid
    dense_x_grid = dense_x_grid[mask].reshape(-1,1)
    dense_y_grid = dense_y_grid[mask].reshape(-1,1)
    dense_locations = np.concatenate((dense_x_grid, dense_y_grid), axis = 1)
    
    return dense_locations

