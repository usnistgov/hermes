from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic.dataclasses import dataclass as typesafedataclass


from hermes.base import Analysis

from .heterscedastic_gpc import HeteroscedasticRobustMax, HeteroscedasticMultiClass

import numpy as np

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import gpflow
from gpflow.ci_utils import ci_niter


class Classification(Analysis):
    """Base level class for classification - predicting labels of data from known examples"""

    #Training data
    locations: np.array # Locations of the oberservations
    labels: np.array #labels in the form of an Nx1 matrix, where N is the number of observations.
    
    #Test data
    domain: np.array #The set of all possible locations to measure




class GPC(Classification):
    """A class for all Gaussian Processes for clasification."""
    ### Set up the GPC ####
    #RBF Kernel
    lengthscales = 1.0
    variance = 1.0
    kernel = gpflow.kernels.RBF(lengthscales = lengthscales, 
                                variance = variance) 

class HomoscedasticGPC(GPC):
    """A class for GPC's where the uncertainty on the labels is the same everywhere."""
    def train(self):
        #Number of classes 
        C = np.unique(self.labels)
        #Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1,1))

        data = (self.locations.astype('float'), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  # Multiclass likelihood


        m = gpflow.models.VGP(
            data = data,
            kernel=self.kernel,
            likelihood=likelihood,
            num_latent_gps=C,)
        
        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(), m.trainable_variables,
            method ='tnc', options=dict(maxiter=ci_niter(1000)))
        
        return m
    
class SparceHomoscedasticGPC(GPC):
    """A class for Sparce GPC's where the uncertainty on the labels is the same everywhere."""
    def train(self):
        #Number of classes 
        C = np.unique(self.labels)
        #Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1,1))

        data = (self.locations.astype('float'), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  # Multiclass likelihood

        M = int(0.4*Y.shape[0]) #Number of inducing points
        Z1 = np.random.permutation(inputs) #Generate a random list of input locations
        Z = Z1[:M, :].copy() #Take the first M locations of Z1 to initialize the inducing points


        m = gpflow.models.SVGP(
            kernel,
            likelihood,
            Z,
            num_latent_gps=C,)
        
        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(), m.trainable_variables,
            method ='tnc', options=dict(maxiter=ci_niter(1000)))
        
        return m

class HeteroscedasticGPC(GPC):
    """A class for GPC's where the training data has known uncertainty.
    Specifically, at every observation there is a probabilistic assignment of the labels."""

    #Probabilistic labeling
    probabilities: np.array # NxC matrix, where C is the number of clusters - rows must sum to 1.

    #Train the models
    def train(self):

        #Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1,1))
        #Tensor of the probabilities
        Sigma_y = tf.convert_to_tensor(self.probabilities)
        #Number of clusters
        C = len(self.probabilities[0,:])

        #Package training data
        data = (self.locations.astype('float'), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = HeteroscedasticRobustMax(C, Sigma_y)  # Robustmax inverse link function
        likelihood = HeteroscedasticMultiClass(C, invlink=invlink)  # Multiclass likelihood


        m = gpflow.models.VGP(
            data = data,
            kernel=self.kernel,
            likelihood=likelihood,
            num_latent_gps=C,)
        
        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(), m.trainable_variables,
            method ='tnc', options=dict(maxiter=ci_niter(1000)))
        
        return m


class SparceHeteroscedasticGPC(GPC):
    """A class for sparce GPC's where the training data has known uncertainty.
    Specifically, at every observation there is a probabilistic assignment of the labels."""

    #Probabilistic labeling
    probabilities: np.array # NxC matrix, where C is the number of clusters - rows must sum to 1.



    #Train the models
    def train(self):

        #Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1,1))
        #Tensor of the probabilities
        Sigma_y = tf.convert_to_tensor(self.probabilities)
        #Number of clusters
        C = len(self.probabilities[0,:])

        #Package training data
        data = (self.locations.astype('float'), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = HeteroscedasticRobustMax(C, Sigma_y)  # Robustmax inverse link function
        likelihood = HeteroscedasticMultiClass(C, invlink=invlink)  # Multiclass likelihood

        M = int(0.4*Y.shape[0]) #Number of inducing points
        Z1 = np.random.permutation(inputs) #Generate a random list of input locations
        Z = Z1[:M, :].copy() #Take the first M locations of Z1 to initialize the inducing points


        m = gpflow.models.SVGP(
            self.kernel,
            likelihood,
            Z,
            num_latent_gps=C,)
        
        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(), m.trainable_variables,
            method ='tnc', options=dict(maxiter=ci_niter(1000)))
        
        return m