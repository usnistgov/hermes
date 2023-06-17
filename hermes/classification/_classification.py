import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from pydantic.dataclasses import dataclass as typesafedataclass

from hermes.base import Analysis

from .heteroscedastic_gpc import HeteroscedasticMultiClass, HeteroscedasticRobustMax

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import gpflow


@dataclass
class Classification(Analysis):
    """Base level class for classification - predicting labels of data from known examples"""

    # Book-keeping
    indexes: np.ndarray # Indexes of all the possible 
    measured_indexes: np.ndarray # Indexes that have been measured

    # Training data
    locations: np.ndarray  # Locations of the oberservations
    labels: np.ndarray  # labels in the form of an Nx1 matrix, where N is the number of observations.

    # Test data
    domain: np.ndarray  # The set of all possible locations to measure

    # Unmeasured_Locations
    @property
    def unmeasured_indexes(self):
        """Find all the indexes in the domain that haven't been measured."""

        measured_set = set(map(tuple, self.measured_indexes))
        domain_set = set(map(tuple, self.indexes))

        unmeasured = np.array(list(domain_set - measured_set))
        return unmeasured
    
    @property
    def unmeasured_locations(self):
        """Find all the indexes in the domain that haven't been measured."""
        unmeas_locations = self.domain[self.unmeasured_indexes]
        return unmeas_locations

    # The Model
    model: Optional[Any] = field(init=False, default=None)

    # Indexes
    def return_index(self, locations):
        # find the indexes of the domain that coorispond to the locations
        # Useful to convert locations requested by acquisition functions to an index in the entire domain.
        indexes = []
        for i in range(locations.shape[0]):
            index = np.argmax(np.prod(self.domain == locations[i, :], axis = 1))
            indexes.append(index[0][0])

        return indexes


@dataclass
class GPC(Classification):
    """A class for all Gaussian Processes for clasification."""

    ### Set up the GPC ####
    # RBF Kernel
    lengthscales = 1.0
    variance = 1.0

    kernel = gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance)

    def predict(self):
        """Predict the model accross the domain."""

        mean, var = self.model.predict_y(self.domain)
        self.mean = mean
        self.var = var

    def predict_unmeasured(self):
        """Predict the model on the unmeasured locations of the domain"""
        # Predict the classes in the unmeasured locations
        self.mean_unmeasured, var_s = self.model.predict_y(self.unmeasured_locations)

        # Sum the variances across the classes
        self.var_unmeasured = np.sum(var_s, axis=1).reshape(-1, 1)

    # def ML_as_service_predict(self):
    #     #Invoke the model from the ML as service run:
    #     model_name = self.model_name

    #     #some http command to call the model_name

    #     self.mean = mean
    #     self.var = var



@dataclass
class HomoscedasticGPC(GPC):
    """A class for GPC's where the uncertainty on the labels is the same everywhere."""

    def train(self):
        # Number of classes
        C = np.unique(self.labels)
        # Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))

        data = (self.locations.astype("float"), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(
            C, invlink=invlink
        )  # Multiclass likelihood

        m = gpflow.models.VGP(
            data=data,
            kernel=self.kernel,
            likelihood=likelihood,
            num_latent_gps=C,
        )

        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(),
            m.trainable_variables,
            method="tnc",
            # options=dict(maxiter=1000)
        )

        self.model = m



@dataclass
class SparceHomoscedasticGPC(GPC):
    """A class for Sparce GPC's where the uncertainty on the labels is the same everywhere."""

    def train(self):
        """Use the training data to train the model."""

        # Number of classes
        C = np.unique(self.labels)
        # Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))

        data = (self.locations.astype("float"), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(
            C, invlink=invlink
        )  # Multiclass likelihood

        M = int(0.4 * Y.shape[0])  # Number of inducing points
        Z1 = np.random.permutation(inputs)  # Generate a random list of input locations
        Z = Z1[
            :M, :
        ].copy()  # Take the first M locations of Z1 to initialize the inducing points

        model = gpflow.models.SVGP(
            kernel,
            likelihood,
            Z,
            num_latent_gps=C,
        )

        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            model.training_loss_closure(),
            model.trainable_variables,
            method="tnc",
            options=dict(maxiter=1000),
        )

        self.model = model


@dataclass
class HeteroscedasticGPC(GPC):
    """A class for GPC's where the training data has known uncertainty.
    Specifically, at every observation there is a probabilistic assignment of the labels.
    """

    # Probabilistic labeling
    probabilities: np.ndarray  # NxC matrix, where C is the number of clusters - rows must sum to 1.

    # def __init__(self, probabilities):
    #     self.probabilities = probabilities
    #     super().__init__(**kwargs)

    # Train the models
    def train(self):
        # Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))
        # Tensor of the probabilities
        Sigma_y = tf.convert_to_tensor(self.probabilities)
        # Number of clusters
        C = len(self.probabilities[0, :])

        # Package training data
        data = (self.locations.astype("float"), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = HeteroscedasticRobustMax(
            C, Sigma_y
        )  # Robustmax inverse link function
        likelihood = HeteroscedasticMultiClass(
            C, invlink=invlink
        )  # Multiclass likelihood

        m = gpflow.models.VGP(
            data=data,
            kernel=self.kernel,
            likelihood=likelihood,
            num_latent_gps=C,
        )

        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(),
            m.trainable_variables,
            method="TNC",
            # options=dict(maxiter=1000)
        )

        self.model = m

    # def ML_service_train(self):
    #     # Tensor of the lables
    #     Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))
    #     # Tensor of the probabilities
    #     Sigma_y = tf.convert_to_tensor(self.probabilities)
    #     # Number of clusters
    #     C = len(self.probabilities[0, :])

    #     # Package training data
    #     data = (self.locations.astype("float"), Y)

    #     ### Set up the GPC ####
    #     # Robustmax Multiclass Likelihood
    #     invlink = HeteroscedasticRobustMax(
    #         C, Sigma_y
    #     )  # Robustmax inverse link function
    #     likelihood = HeteroscedasticMultiClass(
    #         C, invlink=invlink
    #     )  # Multiclass likelihood

    #     m = gpflow.models.VGP(
    #         data=data,
    #         kernel=self.kernel,
    #         likelihood=likelihood,
    #         num_latent_gps=C,
    #     )

    #     #### Train the GPC ####
    #     opt = gpflow.optimizers.Scipy()

    #     opt_logs = opt.minimize(
    #         m.training_loss_closure(),
    #         m.trainable_variables,
    #         method="TNC",
    #         # options=dict(maxiter=1000)
    #     )

    #     #Send model to ML as service
    #     ## some http command:

    #     model_name = ###
    #     self.model_name = model_name
    #     self.model = m

@dataclass
class SparceHeteroscedasticGPC(GPC):
    """A class for sparce GPC's where the training data has known uncertainty.
    Specifically, at every observation there is a probabilistic assignment of the labels.
    """

    # Probabilistic labeling
    probabilities: np.array  # NxC matrix, where C is the number of clusters - rows must sum to 1.

    # Train the models
    def train(self):
        # Tensor of the lables
        Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))
        # Tensor of the probabilities
        Sigma_y = tf.convert_to_tensor(self.probabilities)
        # Number of clusters
        C = len(self.probabilities[0, :])

        # Package training data
        data = (self.locations.astype("float"), Y)

        ### Set up the GPC ####
        # Robustmax Multiclass Likelihood
        invlink = HeteroscedasticRobustMax(
            C, Sigma_y
        )  # Robustmax inverse link function
        likelihood = HeteroscedasticMultiClass(
            C, invlink=invlink
        )  # Multiclass likelihood

        M = int(0.4 * Y.shape[0])  # Number of inducing points
        Z1 = np.random.permutation(inputs)  # Generate a random list of input locations
        Z = Z1[
            :M, :
        ].copy()  # Take the first M locations of Z1 to initialize the inducing points

        m = gpflow.models.SVGP(
            self.kernel,
            likelihood,
            Z,
            num_latent_gps=C,
        )

        #### Train the GPC ####
        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(
            m.training_loss_closure(),
            m.trainable_variables,
            method="tnc",
            options=dict(maxiter=1000),
        )

        self.model = m
