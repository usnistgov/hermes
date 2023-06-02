from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic.dataclasses import dataclass as typesafedataclass


from hermes.base import Analysis

from .heteroscedastic_gpc import HeteroscedasticRobustMax, HeteroscedasticMultiClass

import numpy as np

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import gpflow

@dataclass
class Classification(Analysis):
    """Base level class for classification - predicting labels of data from known examples"""

    #Training data
    locations: np.array # Locations of the oberservations
    labels: np.array #labels in the form of an Nx1 matrix, where N is the number of observations.
    
    #Test data
    domain: np.array #The set of all possible locations to measure
    
    #Unmeasured_Locations
    @property
    def unmeasured_locations(self):
        """Find all the locations in the domain that haven't been measured."""

        measured_set = set(map(tuple, self.locations))
        domain_set = set(map(tuple, self.domain))

        unmeasured = np.array(list(domain_set - measured_set))
        return unmeasured

    model: Any

# class NN(Classification):
#     model: pytorch = None

@dataclass
class GPC(Classification):
    """A class for all Gaussian Processes for clasification."""
    ### Set up the GPC ####
    #RBF Kernel
    lengthscales = 1.0
    variance = 1.0

    kernel = gpflow.kernels.RBF(lengthscales = lengthscales, 
                                variance = variance) 
    
   
    def predict(self):
        """Predict the model accross the domain."""

        mean, var = self.model.predict_y(self.domain)
        self.mean = mean
        self.var = var

    def acquire(self):
        """Acquire the next point(s) to measure."""

        mean, var = self.model.predict_y(self.unmeasured_locations)
        

@dataclass
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
            method ='tnc', 
            #options=dict(maxiter=1000)
            )
        
        self.model = m

    
@dataclass
class SparceHomoscedasticGPC(GPC):
    """A class for Sparce GPC's where the uncertainty on the labels is the same everywhere."""
    def train(self):
        """Use the training data to train the model."""


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


        model = gpflow.models.SVGP(
            kernel,
            likelihood,
            Z,
            num_latent_gps=C,)
        
        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            model.training_loss_closure(), model.trainable_variables,
            method ='tnc', options=dict(maxiter=1000))
        
        self.model = model



@dataclass
class HeteroscedasticGPC(GPC):
    """A class for GPC's where the training data has known uncertainty.
    Specifically, at every observation there is a probabilistic assignment of the labels."""

    #Probabilistic labeling
    probabilities: np.array # NxC matrix, where C is the number of clusters - rows must sum to 1.

    # def __init__(self, probabilities):
    #     self.probabilities = probabilities
    #     super().__init__(**kwargs)

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
            method ='TNC', 
            # options=dict(maxiter=1000)
            )
        
        self.model = m
    


@dataclass
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
            method ='tnc', options=dict(maxiter=1000))
        
        self.model = m
    
