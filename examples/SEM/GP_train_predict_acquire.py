# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:14:51 2022

@author: asm6
"""

import numpy as np

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import gpflow
from gpflow.ci_utils import ci_niter


def Train_GPR(train_locations, train_observations,
              RBF_L = 8.0, RBF_var =0.00001 , const=0.001, noise_var=0.00001,
              optimizer = 9):
    ''' Trains a Gaussian Process Regressor using the input locations 
    in 'train_locations' and the obersvations in 'train_observations', 
    and returns the trained GPR model.'''
    
    X = tf.convert_to_tensor(train_locations, dtype=tf.float64) #shape should be (entries, dimensions)
    Y = tf.convert_to_tensor(train_observations, dtype=tf.float64)
    data = (X,Y)
    #RBF Kernel + Constant Kernel 
    kernel = gpflow.kernels.RBF(lengthscales=RBF_L, variance=RBF_var) \
            + gpflow.kernels.Constant(variance=const)
    
    #Model
    m = gpflow.models.GPR(data, kernel,
                          noise_variance=noise_var)
    
    #### Train the GPC ####
    opt = gpflow.optimizers.Scipy()
    
    opt_list = ['Nelder-Mead' , 'Powell', 'CG','BFGS','Newton-CG', 'L-BFGS-B','TNC', 
            'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    opt_logs = opt.minimize(
    m.training_loss_closure(), m.trainable_variables,method=opt_list[9], options=dict(maxiter=ci_niter(10000)))
    
    return m


def Predict_GPR(m, test_locations):
    ''' Uses the trained model to predict the function at the test locations'''
    X = tf.convert_to_tensor(test_locations, dtype=tf.float64)
    
    mean, var = m.predict_f(X)
    
    return mean.numpy(), var.numpy()

def Acquire_GPR(m, test_locations, batchsize):
    ''' Calculates the acquisition function for the trained state of the model
    uses the current state of the kernel to modify the acquisiton landscape for 
    subsequent batch points '''
    
    #Find the dimensions of the location coordinates
    #'test_locations' should be #_of_points by #_of_dimensions
    dim = test_locations.shape[1]
    #Start an array of the indexes of the locations
    loc_index = np.arange(test_locations.shape[0])
    
    _, batch_var = Predict_GPR(m, test_locations)
    
    #acquisition function 
    alpha = batch_var
    
    alpha_batch = np.array(alpha)
    
    next_sample_index = np.argmax(alpha)
    next_sample = test_locations[next_sample_index]
    points = np.array((next_sample_index)).reshape(-1,1)
    
    #Initialize container for distance matrix
    K = np.zeros((len(test_locations),0))
    
    #Find the next points in the batch
    for e in range(batchsize):
        '''Use the kernel of the model to cacluate the similary of the 
        most recent batch point to all the other possible locations.'''
        K1 = m.kernel.K(test_locations, test_locations[points[-1]].reshape(-1,dim))
        
        #stack K and K1
        K = np.hstack((K, K1))
        
        #Sum across all of the rows (each of the locations)
        K_sum = np.sum(K, axis = 1).reshape(-1,1)
        
        #Normalize
#         alpha_modifier = K_sum/(np.max(K_sum))
#         alpha_modifier = (K_sum-np.min(K_sum))/(np.max(K_sum)-np.min(K_sum))
        alpha_modifier = (np.array(K1) - np.min(K1))/(np.max(K1)-np.min(K1))
        
        #Modify the acquisition landscape
#         alpha_batch = alpha - alpha_modifier
#         alpha_batch = alpha*(1 - alpha_modifier)
#         alpha_batch = alpha + (np.max(alpha)-np.min(alpha))*(1 - alpha_modifier)
#         print(alpha_batch.shape)
        alpha_batch = alpha_batch + (np.max(alpha_batch)-np.min(alpha_batch))*(1 - alpha_modifier)
        
        #Find the points not in the batch already
        diff_index = np.setdiff1d(loc_index, points.flatten())
        
        #Find the next batch point
        next_batchpoint_index = np.argmax(alpha_batch[diff_index])
        points = np.vstack((points,next_batchpoint_index))
        
    batch_locations = test_locations[points.flatten()]
    
    return batch_locations, points
        

