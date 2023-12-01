# pylint: disable=W0201
"""Classification methods for Hermes."""
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, Union

import h5py
import numpy as np
import tensorflow as tf

from hermes.base import Analysis

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

logger = logging.getLogger("hermes")

GPC_INSTALLED = False
try:
    import gpflow

    from .heteroscedastic_gpc import HeteroscedasticMultiClass, HeteroscedasticRobustMax
except ModuleNotFoundError:
    logger.warning("GPFlow is not installed.")
else:
    GPC_INSTALLED = True


@dataclass
class Classification(Analysis):
    """
    Base level class for classification.

    Used to predict labels of data from known examples.

    Attributes
    ----------
    unmeasured_indexes
    unmeasured_locations
    indexes : np.ndarray
        Indexes of all the possible locations to measure.
    measured_indexes : np.ndarray
        Indexes that have been measured.
    locations : np.ndarray
        Locations of the observations.
    labels : np.ndarray
        Labels in the form of an Nx1 matrix, where N is the number of observations.
    domain : np.ndarray
        The set of all possible locations to measure.
    model: Any
    # TODO

    Methods
    -------
    return_index(locations)
        Find the indexes of the domain that correspond to the locations.

    """

    # Book-keeping
    indexes: np.ndarray  # Indexes of all the possible
    measured_indexes: np.ndarray  # Indexes that have been measured

    # Training data
    locations: np.ndarray  # Locations of the oberservations
    labels: np.ndarray  # labels in the form of an Nx1 matrix, where N is the number of observations.

    # Test data
    domain: np.ndarray  # The set of all possible locations to measure

    def __post_init__(self):
        """Check measured_indexes are 1d"""
        if not self.indexes.ndim == 1:
            raise ValueError("invalid dimensions for indexes, must be 1d")

    # Unmeasured_Locations
    @property
    def unmeasured_indexes(self):
        """All indexes in the domain that haven't been measured."""

        measured_set = set(self.measured_indexes)
        domain_set = set(self.indexes)

        unmeasured = np.array(list(domain_set - measured_set))
        return unmeasured

    @property
    def unmeasured_locations(self):
        """All locations in the domain that haven't been measured."""
        unmeas_locations = self.domain[self.unmeasured_indexes]
        return unmeas_locations

    # The Model
    model: Optional[Any] = field(init=False, default=None)

    # Indexes
    def return_index(self, locations) -> list:
        """
        Find the indexes of the domain that correspond to the locations.

        Useful to convert locations requested by
        acquisition functions to an index in the entire domain.
        """
        indexes = []
        for i in range(locations.shape[0]):
            index = np.argmax(np.prod(self.domain == locations[i, :], axis=1))
            indexes.append(index)

        return indexes


if GPC_INSTALLED:

    @dataclass
    class GPC(Classification):
        """
        Base class for all Gaussian Processes for classification.

        Attributes
        ----------
        lengthscale : float, default=1.0
            Lengthscale of the kernel.
        variance : float, default=1.0
            Variance of the kernel.
        kernel : gpflow.kernels.Kernel, default=gpflow.kernels.RBF
            The kernel of the Gaussian Process.
        params : dict
            The parameters of the model.


        Methods
        -------
        predict()
            Predict the model accross the domain.
        predict_unmeasured()
            Predict the model on the unmeasured locations of the domain.
        save(path)
            Save the model to a file.
        load(path)
            Load model from a file.
        load_params(other)
            Load hyperparameters from another model.
        save_params(path)
            Save the parameters of the model to a HDF5 file.

        """

        ### Set up the GPC ####
        # RBF Kernel
        lengthscales = 1.0
        variance = 1.0  # TODO ask probelamtic when redifining kernel

        kernel = gpflow.kernels.RBF(lengthscales=lengthscales, variance=variance)

        def predict(self):
            """Predict the model accross the domain."""

            mean, var = self.model.predict_y(self.domain)
            self.mean = mean
            self.var = var

        def predict_unmeasured(self) -> None:
            """Predict the model on the unmeasured locations of the domain"""
            # Predict the classes in the unmeasured locations
            self.mean_unmeasured, var_s = self.model.predict_y(
                self.unmeasured_locations
            )

            # Sum the variances across the classes
            self.var_unmeasured = np.sum(var_s, axis=1).reshape(-1, 1)

        def save(self, path: str) -> None:
            """Save the model to a file."""
            # from https://gpflow.github.io/GPflow/2.9.0/notebooks/getting_started/saving_and_loading.html#TensorFlow-saved_model
            self.model.compiled_predict_f = tf.function(
                lambda Xnew: self.model.predict_f(Xnew, full_cov=False),
                input_signature=[
                    tf.TensorSpec(shape=[None, self.domain.shape[1]], dtype=tf.float64)
                ],
            )
            self.model.compiled_predict_y = tf.function(
                lambda Xnew: self.model.predict_y(Xnew, full_cov=False),
                input_signature=[
                    tf.TensorSpec(shape=[None, self.domain.shape[1]], dtype=tf.float64)
                ],
            )
            tf.saved_model.save(self.model, path)

            # def ML_as_service_predict(self):

        def load_params(
            self, other: Union[Type[Classification], gpflow.models.GPModel]
        ) -> None:
            """Load hyperparameters from another model."""
            if isinstance(other, GPC):
                other_model = other.model
            elif isinstance(other, gpflow.models.GPModel):
                other_model = other
            else:
                raise TypeError(
                    "other must be a classification method or gpflow.models.GPModel"
                )
            params = gpflow.utilities.parameter_dict(other_model)
            gpflow.utilities.multiple_assign(self.model, params)

        #     #Invoke the model from the ML as service run:
        #     model_name = self.model_name

        #     #some http command to call the model_name

        #     self.mean = mean
        #     self.var = var

        @classmethod
        def load(cls, path: Union[str, Path]) -> "GPC":
            """Load model from a file."""
            loaded_ = tf.saved_model.load(path)
            # work on each param

        @property
        def params(self):
            """Return the parameters of the model."""
            return gpflow.utilities.parameter_dict(self.model)

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "kernel":
                super().__setattr__(name, value)
                self._generate_model()  # every time the kernel is changed, the model must be regenerated
                return
            return super().__setattr__(name, value)

        def save_params(self, path: str):
            """Save the parameters of the model to a HDF5 file."""
            file = h5py.File(path, "w")
            file.create_group("kernel")
            file.create_dataset("kernel/lengthscales", data=self.kernel.lengthscales)
            file.create_dataset("kernel/variance", data=self.kernel.variance)
            file.create_dataset("kernel/name", data=self.kernel.__class__.__name__)
            file.create_dataset("q_mu", data=self.params[".q_mu"])
            file.create_dataset("q_sqrt", data=self.params[".q_sqrt"])
            file.create_dataset(
                "epsilon", data=self.params[".likelihood.invlink.epsilon"]
            )
            file.create_dataset("num_data", data=self.params[".num_data"])
            file.create_dataset("kernel_variance", data=self.params[".kernel.variance"])
            file.create_dataset(
                "kernel_lengthscales", data=self.params[".kernel.lengthscales"]
            )
            file.close()

    # TODO
    # fix kernel variance and everything not q_mu and q_sqrt
    # then train the model
    # to reconstruct q_mu and q_sqrt
    @dataclass
    class HomoscedasticGPC(GPC):
        """A class for GPC's where the uncertainty on the labels is the same everywhere."""

        def __post_init__(self):
            self._generate_model()

        def _generate_model(self):
            # Number of classes
            C = np.unique(self.labels)
            # Tensor of the lables
            Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))

            self._data = (self.locations.astype("float"), Y)

            ### Set up the GPC ####
            # Robustmax Multiclass Likelihood
            invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
            self._likelihood = gpflow.likelihoods.MultiClass(
                C, invlink=invlink
            )  # Multiclass likelihood

            m = gpflow.models.VGP(
                data=self._data,
                kernel=self.kernel,
                likelihood=self._likelihood,
                num_latent_gps=C,
            )
            self.model = m

        def train(self):
            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt_logs = opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                # options=dict(maxiter=1000)
            )
            # self.model = self.model

    @dataclass
    class SparceHomoscedasticGPC(GPC):
        """A class for Sparce GPC's where the uncertainty on the labels is the same everywhere."""

        def __post_init__(self):
            self._generate_model()

        def _generate_model(self):
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
            Z1 = np.random.permutation(
                data[0]
            )  # Generate a random list of input locations
            Z = Z1[
                :M, :
            ].copy()  # Take the first M locations of Z1 to initialize the inducing points
            model = gpflow.models.SVGP(
                self.kernel,
                likelihood,
                Z,
                num_latent_gps=C,
            )
            self.model = model

        def train(self):
            """Use the training data to train the model."""

            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt_logs = opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                options=dict(maxiter=1000),
            )

            # self.model = model

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
        def __post_init__(self):
            self._generate_model()

        def _generate_model(self) -> None:
            # Tensor of the lables
            Y = tf.convert_to_tensor(self.labels.reshape(-1, 1))
            # Tensor of the probabilities
            sigma_y = tf.convert_to_tensor(self.probabilities)
            # Number of clusters
            _C = len(self.probabilities[0, :])

            # Package training data
            self._data = (self.locations.astype("float"), Y)

            ### Set up the GPC ####
            # Robustmax Multiclass Likelihood
            invlink = HeteroscedasticRobustMax(
                _C, sigma_y
            )  # Robustmax inverse link function
            self._likelihood = HeteroscedasticMultiClass(
                _C, invlink=invlink
            )  # Multiclass likelihood

            m = gpflow.models.VGP(
                data=self._data,
                kernel=self.kernel,
                likelihood=self._likelihood,
                num_latent_gps=_C,
            )
            self.model = m

        @classmethod
        def _generate_model_from_trained(
            cls,
            kernel: gpflow.kernels.Kernel,
            sigma_y: tf.Tensor,
            C: int,
            data: tuple,
        ) -> gpflow.models.VGP:
            ### Set up the GPC ####
            # Robustmax Multiclass Likelihood
            invlink = HeteroscedasticRobustMax(
                C, sigma_y
            )  # Robustmax inverse link function
            _likelihood = HeteroscedasticMultiClass(
                C, invlink=invlink
            )  # Multiclass likelihood

            m = gpflow.models.VGP(
                data=data,  # type: ignore
                kernel=kernel,
                likelihood=_likelihood,
                num_latent_gps=C,
            )

            gpflow.utilities.set_trainable(m.kernel.variance, False)
            gpflow.utilities.set_trainable(m.kernel.lengthscales, False)
            # "train" model. goal: get same q_mu and q_sqrt as before
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(
                m.training_loss_closure(),
                m.trainable_variables,
                method="TNC",
            )
            return m

        # Train the models
        def train(self):
            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt_logs = opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="TNC",
                # options=dict(maxiter=1000)
            )

            # self.model = m

        def save_trained(self, path: Union[str, Path]) -> None:
            with h5py.File(str(path), "w") as file:
                file.create_group("kernel")
                file.create_dataset(
                    "kernel/variance",
                    data=self.model.kernel.variance.numpy(),
                )
                file.create_dataset(
                    "kernel/lengthscales",
                    data=self.model.kernel.lengthscales.numpy(),
                )
                file.create_dataset(
                    "kernel/name",
                    data=self.model.kernel.__class__.__name__,
                )
                file.create_group("data")
                file.create_dataset("data/x", data=self._data[0])
                file.create_dataset("data/y", data=self._data[1])
                file.create_dataset("probabilities", data=self.probabilities)

        @classmethod
        def load(cls, path: Union[str, Path]) -> gpflow.models.VGP:
            """Load trained model from a file."""
            with h5py.File(str(path), "r") as file:
                kernel_name = file["kernel/name"][()].decode()
                kernel_variance = file["kernel/variance"][()]
                kernel_lengthscales = file["kernel/lengthscales"][()]
                data = (file["data/x"][()], file["data/y"][()])
                probabilities = file["probabilities"][()]

            kernel = getattr(gpflow.kernels, kernel_name)(
                lengthscales=kernel_lengthscales, variance=kernel_variance
            )
            sigma_y = tf.convert_to_tensor(probabilities)
            C = len(probabilities[0, :])
            return cls._generate_model_from_trained(
                kernel=kernel,
                sigma_y=probabilities,
                C=C,
                data=data,
            )

            # work on each param

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

        def __post_init__(self):
            self._generate_model()

        def _generate_model(self):
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
            Z1 = np.random.permutation(
                data[0]
            )  # Generate a random list of input locations
            Z = Z1[
                :M, :
            ].copy()  # Take the first M locations of Z1 to initialize the inducing points

            m = gpflow.models.SVGP(
                self.kernel,
                likelihood,
                Z,
                num_latent_gps=C,
            )
            self.model = m

        # Train the models
        def train(self):
            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt_logs = opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                options=dict(maxiter=1000),
            )

            # self.model = m

            # TODO Kernel not RBF.
