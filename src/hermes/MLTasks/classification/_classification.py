# pylint: disable=W0201
"""Classification methods for Hermes."""
import logging
import warnings
from dataclasses import field
from pathlib import Path
from typing import Any, Optional, Type, Union

import h5py  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from hermes._base import Analysis

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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Classification(Analysis):
    """
    Base level class for classification.

    Used to predict labels of data from known examples.

    Attributes
    ----------
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
    model: gpflow.models.VGP
    # TODO, make comments of how they get initialized

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
    labels: np.ndarray  # labels in the form of an Nx1 matrix,
    # where N is the number of observations.

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
    def unmeasured_locations(self) -> np.ndarray:
        """All locations in the domain that haven't been measured."""
        if len(self.unmeasured_indexes) > 0:
            unmeas_locations = self.domain[self.unmeasured_indexes]
        else:
            unmeas_locations = np.array([])
        return unmeas_locations

    # The Model
    # model: Optional[Any] = field(init=False, default=None)
    model: gpflow.models.VGP = field(init=False)

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
        lengthscales_prior = 1.0
        variance_prior = 1.0

        kernel = gpflow.kernels.RBF(
            lengthscales=lengthscales_prior, variance=variance_prior
        )
        model: gpflow.models.VGP = field(init=False)

        def predict(self):
            """Predict the model accross the domain."""

            mean, var = self.model.predict_y(self.domain)
            self.mean = mean
            self.var = var

        def predict_unmeasured(self) -> None:
            """Predict the model on the unmeasured locations of the domain."""
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

        @property
        def params(self):
            """Return the parameters of the model."""
            return gpflow.utilities.parameter_dict(self.model)

        def _generate_model(self) -> None:
            # placeholder for mypy
            pass

        def __setattr__(self, name: str, value: Any) -> None:
            if (
                name == "kernel"
            ):  # every time the kernel is changed, the model must be regenerated
                super().__setattr__(name, value)
                if (
                    hasattr(self, "model") and self.model is not None
                ):  # if model has been generated already
                    self._generate_model()
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

    @dataclass
    class HomoscedasticGPC(GPC):
        """A class for GPC's where the uncertainty on the labels is the same everywhere."""

        def __post_init__(self):
            self._generate_model()

        def _generate_model(self) -> None:
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
            """Train GPC using scipy optimizer and tnc method."""
            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                # options=dict(maxiter=1000)
            )
            # TODO remove var optlpogs
            # self.model = self.model

    @dataclass
    class SparceHomoscedasticGPC(GPC):
        """A class for Sparce GPC's where the uncertainty on the labels is the same everywhere."""

        def __post_init__(self):
            self._generate_model()

        def _generate_model(self) -> None:
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
            """Train GPC using scipy optimizer, tnc method and `maxiter`=1000."""

            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                options={"maxiter": 1000},
            )

            # self.model = model

    @dataclass
    class HeteroscedasticGPC(GPC):
        """A class for GPC's where the training data has known uncertainty.
        Specifically, at every observation there is a probabilistic assignment of the labels.
        """

        # Probabilistic labeling
        probabilities: (
            np.ndarray
        )  # NxC matrix, where C is the number of clusters - rows must sum to 1.
        _from_trained: bool = False

        def __init__(self, from_trained: bool = False, **kwargs):
            self.probabilities = kwargs["probabilities"]
            kwargs.pop("probabilities")
            if from_trained:
                self.kernel = kwargs["kernel"]
                _params_dict = {
                    ".q_mu": kwargs["q_mu"],
                    ".q_sqrt": kwargs["q_sqrt"],
                }
                kwargs.pop("q_mu")
                kwargs.pop("q_sqrt")
                kwargs.pop("kernel")
                kwargs.update({"_from_trained": True})
                super().__init__(**kwargs)
                self._generate_model()
                gpflow.utilities.multiple_assign(self.model, _params_dict)  # type: ignore
            else:
                super().__init__(**kwargs)

        def __post_init__(self):
            if not self._from_trained:
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
            sigma_y: Union[tf.Tensor, np.ndarray],
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

            return m

        # Train the models
        def train(self):
            """Train GPC using scipy optimizer and tnc method."""
            #### Train the GPC ####
            opt = gpflow.optimizers.Scipy()

            opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="TNC",
                # options=dict(maxiter=1000)
            )

        def save(self, path: Union[str, Path]) -> None:
            """Save model and GPC object to a file.

            Save the trained model along with the GPC
            Python object to a HDF5 file.

            Parameters
            ----------
            path : str | Path
                Path to save the model and object.
            """
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
                # file.create_group("data")
                # file.create_dataset("data/x", data=self._data[0])
                # file.create_dataset("data/y", data=self._data[1])
                file.create_dataset("labels", data=self.labels)
                file.create_dataset("locations", data=self.locations)
                file.create_dataset("indexes", data=self.indexes)
                file.create_dataset("measured_indexes", data=self.measured_indexes)
                file.create_dataset("domain", data=self.domain)
                file.create_dataset("probabilities", data=self.probabilities)
                file.create_dataset("q_mu", data=self.model.q_mu.numpy())  # type: ignore
                file.create_dataset("q_sqrt", data=self.model.q_sqrt.numpy())  # type: ignore

        @classmethod
        def load(cls, path: Union[str, Path], t: Optional[int]) -> "HeteroscedasticGPC":
            """Load trained object and model from a file."""
            with h5py.File(str(path), "r") as file:
                kernel_name = file["kernel/name"][()].decode()  # type: ignore # pylint: disable=E1101
                kernel_variance = file["kernel/variance"][()]  # type: ignore
                kernel_lengthscales = file["kernel/lengthscales"][()]  # type: ignore
                probabilities = file["probabilities"][()]  # type: ignore
                locations = file["locations"][()]  # type: ignore
                labels = file["labels"][()]  # type: ignore
                indexes = file["indexes"][()]  # type: ignore
                measured_indexes = file["measured_indexes"][()]  # type: ignore
                domain = file["domain"][()]  # type: ignore
                q_mu = file["q_mu"][()]  # type: ignore
                q_sqrt = file["q_sqrt"][()]  # type: ignore

            kernel = getattr(gpflow.kernels, kernel_name)(
                lengthscales=kernel_lengthscales, variance=kernel_variance
            )
            obj = HeteroscedasticGPC(
                kernel=kernel,
                probabilities=probabilities,
                domain=domain,
                q_mu=q_mu,
                q_sqrt=q_sqrt,
                locations=locations,
                labels=labels,
                indexes=indexes,
                measured_indexes=measured_indexes,
                from_trained=True,
            )
            return obj

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
        probabilities: (
            np.ndarray
        )  # NxC matrix, where C is the number of clusters - rows must sum to 1.

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

            opt.minimize(
                self.model.training_loss_closure(),
                self.model.trainable_variables,
                method="tnc",
                options={"maxiter": 1000},
            )

            # self.model = m

            # TODO Kernel not RBF.
