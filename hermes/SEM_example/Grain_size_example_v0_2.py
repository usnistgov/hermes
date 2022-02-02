# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:08:30 2022

@author: asm6
"""

from GP_train_predict_acquire import Train_GPR, Predict_GPR, Acquire_GPR

from SEM_grain_size_data_wrangle import (read_SEM_measurement_csv, 
                                        read_wafer_coordinate_csv, 
                                        generate_dense_grid_in_circle)

from matplotlib import pyplot as plt
import numpy as np


#Read in the SEM measurements into DataFrame
Measurements = read_SEM_measurement_csv('Grain size measurements.txt')

#Read in the die locations on the wafer
test_locations = read_wafer_coordinate_csv('Grid_locations_177map.csv')

#Generate dense test locations for veiwing purposes
dense_locations = generate_dense_grid_in_circle(test_locations, 201)

#Train GPR
#NOTE: in wafer coordinates!
measurement_locations = Measurements.iloc[:,0:2].to_numpy().reshape(-1,2)
observations = Measurements.iloc[:,2].to_numpy().reshape(-1,1)
model = Train_GPR(measurement_locations, observations)

#Predict GPR
predict_mean, predict_var = Predict_GPR(model, test_locations)

dense_mean, dense_var = Predict_GPR(model, dense_locations)

acq_locations, acq_loc_indext = Acquire_GPR(model, test_locations, 2)


#Plot
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.scatter(dense_locations[:,0], dense_locations[:,1], c=dense_mean, cmap= plt.plasma(),
           vmin=np.min(dense_mean), vmax=np.max(dense_mean))
plt.colorbar(label = 'mean')
plt.scatter(measurement_locations[:,0], measurement_locations[:,1], c='r', marker='*')
plt.scatter(acq_locations[:,0], acq_locations[:,1], c='g', marker='*')
plt.axis('square')


plt.subplot(122)
plt.scatter(dense_locations[:,0], dense_locations[:,1], c=dense_var, cmap= plt.plasma(),
           vmin=np.min(dense_var), vmax=np.max(dense_var))
plt.colorbar(label = 'var')
plt.scatter(measurement_locations[:,0], measurement_locations[:,1], c='r', marker='*')
plt.scatter(acq_locations[:,0], acq_locations[:,1], c='g', marker='*')
plt.axis('square')
plt.show()



