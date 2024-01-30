# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:26:32 2021

@author: asm6
"""
import numpy as np

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import gpflow
from gpflow.ci_utils import ci_niter

from Heteroscedastic_GPC import HeteroscedasticRobustMax, HeteroscedasticMultiClass

from matplotlib import pyplot as plt


def train_HSGPC_classifier(inputs, labels, probabilities):
  # Inputs are the measurement locations, and labels+probabilities
  # Ouputs are the trained model

  #Tensor of the lables
  Y = tf.convert_to_tensor(labels.reshape(-1,1))
  #Tensor of the probabilities
  Sigma_y = tf.convert_to_tensor(probabilities)
  #Number of clusters
  C = len(probabilities[0,:])

  data = (inputs.astype('float'), Y)

  ### Set up the GPC ####
  #RBF Kernel
  kernel = gpflow.kernels.RBF() 

  # Robustmax Multiclass Likelihood
  invlink = HeteroscedasticRobustMax(C, Sigma_y)  # Robustmax inverse link function
  likelihood = HeteroscedasticMultiClass(C, invlink=invlink)  # Multiclass likelihood


  m = gpflow.models.VGP(
      data = data,
      kernel=kernel,
      likelihood=likelihood,
      num_latent_gps=C,)
  
  #### Train the GPC ####
  opt = gpflow.optimizers.Scipy()

  opt_logs = opt.minimize(
      m.training_loss_closure(), m.trainable_variables,
      method ='tnc', options=dict(maxiter=ci_niter(1000)))
  
  return m

def train_GPC_classifier(inputs, labels, C):
  # Inputs are the measurement locations, and HARD labels, and number of clusters
  # Ouputs are the trained model

  #Tensor of the lables
  Y = tf.convert_to_tensor(labels.reshape(-1,1))

  data = (inputs.astype('float'), Y)

  ### Set up the GPC ####
  #RBF Kernel
  kernel = gpflow.kernels.RBF() 

  # Robustmax Multiclass Likelihood
  invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
  likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  # Multiclass likelihood


  m = gpflow.models.VGP(
      data = data,
      kernel=kernel,
      likelihood=likelihood,
      num_latent_gps=C,)
  
  #### Train the GPC ####
  opt = gpflow.optimizers.Scipy()

  opt_logs = opt.minimize(
      m.training_loss_closure(), m.trainable_variables,
      method ='tnc', options=dict(maxiter=ci_niter(1000)))
  
  return m

def train_SHSGPC_classifier(inputs, labels, probabilities):
  # Inputs are the measurement locations, and labels+probabilities
  # Ouputs are the trained model

  #Tensor of the lables
  Y = tf.convert_to_tensor(labels.reshape(-1,1))
  #Tensor of the probabilities
  Sigma_y = tf.convert_to_tensor(probabilities)
  #Number of clusters
  C = len(probabilities[0,:])

  data = (inputs.astype('float'), Y)

  ### Set up the GPC ####
  #RBF Kernel
  kernel = gpflow.kernels.RBF() 

  # Robustmax Multiclass Likelihood
  invlink = HeteroscedasticRobustMax(C, Sigma_y)  # Robustmax inverse link function
  likelihood = HeteroscedasticMultiClass(C, invlink=invlink)  # Multiclass likelihood
  
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
      m.training_loss_closure(data), m.trainable_variables,
      method ='tnc', options=dict(maxiter=ci_niter(1000)))
  
  return m

def train_SGPC_classifier(inputs, labels, C):
  # Inputs are the measurement locations, and HARD labels, and number of clusters
  # Ouputs are the trained model

  #Tensor of the lables
  Y = tf.convert_to_tensor(labels.reshape(-1,1))

  data = (inputs.astype('float'), Y)

  ### Set up the GPC ####
  #RBF Kernel
  kernel = gpflow.kernels.RBF() 

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
      m.training_loss_closure(data), m.trainable_variables,
      method ='tnc', options=dict(maxiter=ci_niter(1000)))
  
  return m

def predict_class(m, test_locations):
  # inputs are the trained model, and the test locations
  # Outputs are mean and variance of the model

  mean, Var = m.predict_y(test_locations)

  predicted_classes = np.zeros((test_locations[:,0].size))

  for i in range(test_locations[:,0].size):
      predicted_classes[i] = np.argmax(mean[i])

  total_var = np.sum(Var, axis = 1).reshape(-1,1)

  return predicted_classes, total_var, mean, Var

def plot_each_class_fit(test_locations, mean, Var, C):
  plt.figure(figsize = (18,8))

  plt.subplot(121)
  plt.scatter(test_locations[:,0].reshape(-1,1), test_locations[:,1].reshape(-1,1), c = mean[:,C].numpy().reshape(-1,1))
  plt.axis("scaled")
  plt.colorbar()
  plt.title("Mean of GPC Fit to Class {}".format(C))

  plt.subplot(122)
  plt.scatter(test_locations[:,0].reshape(-1,1), test_locations[:,1].reshape(-1,1), c = Var[:,C].numpy().reshape(-1,1))
  plt.axis("scaled")
  plt.colorbar()
  plt.title("Mean of GPC Fit to Class {}".format(C))

def plot_fit(test_locations, mean, Var):
  plt.figure(figsize = (18,8))

  plt.subplot(121)
  plt.scatter(test_locations[:,0].reshape(-1,1), test_locations[:,1].reshape(-1,1), c = mean.reshape(-1,1))
  plt.axis("scaled")
  plt.colorbar()
  plt.title("Total Mean of GPC Fit")

  plt.subplot(122)
  plt.scatter(test_locations[:,0].reshape(-1,1), test_locations[:,1].reshape(-1,1), c = Var.reshape(-1,1))
  plt.axis("scaled")
  plt.colorbar()
  plt.title("Total Var of GPC Fit")