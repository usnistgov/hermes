# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:32:29 2021

@author: asm6
"""

import numpy as np
import gpflow
from Train_Predict_Plot_GPC import predict_class

def acquire_next_batch(active_model, input_locations, measured_index, batch_size):
    ''' Use the trained model kernel to acquire a batch of points to measure
    '''
    
    #Find all the locations in the input that HAVEN'T been measured
    unmeasured_index = np.setdiff1d(np.arange(0, input_locations.shape[0]), measured_index)
    
    #Predict the variance at all the unmeasured locations
    _, batch_var, _, _ = predict_class(active_model, input_locations[unmeasured_index])
    
    #Use pure exploration as the primary acquisition function
    alpha = batch_var
    
    #Find the first sample in the batch
    next_sample_index = np.argmax(alpha) #arg in the list of UNMEASURED samples
    next_sample = unmeasured_index[next_sample_index] #arg in the list of all locations
    
    #Start a container for batch of new locations to acquire
    points = np.array((next_sample)).reshape(-1,1)
    
    #Start a container for the distance measurement matrix
    K = np.zeros((len(unmeasured_index),0))
    
    #Find the rest of the points in the batch
    for e in range(batch_size):
        ''' Use the model's kernel to measure the similarity 
        of the latest point in the batch to the rest of the 
        available points (i.e. unmeasured)'''
        K1 = active_model.kernel.K(input_locations[unmeasured_index],
                                   input_locations[points[-1].reshape(-1,1)].reshape(-1,input_locations.shape[1]))
        
        #Stack that column to the distance matrix
        K = np.hstack((K, K1))
        
        #Add the similarities at each location
        K_sum = np.sum(K, axis=1).reshape(-1,1)
        
        #Find the largest similarity
        K_sum_max = np.max(K_sum)
        
        #Normalize the similarities to 0 to 1
        alpha_modifier = K_sum/K_sum_max
        
        '''Modify the acqusition landscape by the 
        similarities to the points already in this batch.
        Points that are very similar to other points in the batch 
        should be avoided (have a lower acquisition value).'''
        alpha_batch = alpha - alpha_modifier
        
        #Find the next point in the batch
        next_batchpoint_index = np.argmax(alpha_batch) #arg in the list of UNMEASURED locations
        next_batchpoint = unmeasured_index[next_batchpoint_index] #arg in the list of all locations
        
        #Add this point to the list of points in the batch
        points = np.vstack((points, next_batchpoint))
        
    return points
        
        
        
    
    
    