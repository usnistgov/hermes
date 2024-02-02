"""A Heteroscedastic Gaussian Process for classification (GPC).
Each training data point has a location, label, an the vector of 
multi-class distribution of probabilities of all the possible labels.
These probabilites are propegated through the training, and predictions of the GPC."""

import logging
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow
logger = logging.getLogger("hermes")

from typing import Any, Optional

# Define the new classes for GPflow
# from gpflow import Module, Parameter

try:
    import tensorflow_probability as tfp
    from check_shapes import check_shapes, inherit_check_shapes
    from gpflow import Parameter
    from gpflow.base import TensorType
    from gpflow.config import default_float

    # from gpflow.utilities import to_default_float, to_default_int
    # from gpflow.quadrature import hermgauss
    from gpflow.likelihoods import Likelihood, RobustMax
    from gpflow.likelihoods.base import (
        Likelihood,
        MeanAndVariance,
        Module,
        MonteCarloLikelihood,
    )
    from gpflow.quadrature import hermgauss
    from gpflow.utilities import to_default_float, to_default_int

    # from gpflow.config import default_float
except ModuleNotFoundError:
    raise ModuleNotFoundError("GPFlow needs to be installed")
except BaseException as exc:
    raise exc


# """Build an inverse-link function for the Heteroscedastic Gaussian Process"""
# class HeteroscedasticRobustMax(Module):
#     """
#     This class represent a multi-class inverse-link function. Given a vector
#     f=[f_1, f_2, ... f_k], the result of the mapping is
#     y = [y_1 ... y_k]
#     with
#     y_i = (1-epsilon)  i == argmax(f)
#           epsilon/(k-1)  otherwise
#     where k is the number of classes.

#     Sigma_y is the label probabilities
#     """

#     def __init__(self, num_classes, Sigma_y, epsilon=1e-3, **kwargs):
#         """
#         `epsilon` represents the fraction of 'errors' in the labels of the
#         dataset. This may be a hard parameter to optimize, so by default
#         it is set un-trainable, at a small value.
#         """
#         super().__init__(**kwargs)
#         transform = tfp.bijectors.Sigmoid()
#         prior = tfp.distributions.Beta(to_default_float(0.2), to_default_float(5.0))
#         self.epsilon = Parameter(epsilon, transform=transform, prior=prior, trainable=False)
#         self.num_classes = num_classes
#         self._squash = 1e-6
#         self.Sigma_y = Sigma_y #Sigma_y is the label probabilities

#     def __call__(self, F):
#         i = tf.argmax(F, 1)
#         return tf.one_hot(
#             i, self.num_classes, tf.squeeze(1.0 - self.epsilon), tf.squeeze(self.eps_k1)
#         )

#     @property
#     def eps_k1(self):
#         return self.epsilon / (self.num_classes - 1.0)

#     def safe_sqrt(self, val):
#         return tf.sqrt(tf.clip_by_value(val, 1e-10, np.inf))

#     def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
#         Y = to_default_int(Y)
#         # work out what the mean and variance is of the indicated latent function.
#         '''If we have the class membership probabilties, train on those.
#         Otherwise use a one hot encoding of the labels'''
#         if mu.shape == self.Sigma_y.shape:
#           oh_on = self.Sigma_y
#         else:
#           oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1.0, 0.0), dtype=mu.dtype)

#         mu_selected = tf.reduce_sum(oh_on * mu, 1)
#         var_selected = tf.reduce_sum(oh_on * var, 1)

#         # generate Gauss Hermite grid
#         X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
#             self.safe_sqrt(2.0 * var_selected), (-1, 1)
#         )

#         # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
#         dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
#             self.safe_sqrt(var), 2
#         )
#         cdfs = 0.5 * (1.0 + tf.math.erf(dist / np.sqrt(2.0)))

#         cdfs = cdfs * (1 - 2 * self._squash) + self._squash

#         # blank out all the distances on the selected latent function
#         # oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0.0, 1.0), dtype=mu.dtype)
#         oh_off = tf.ones_like(oh_on) - oh_on
#         cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

#         # take the product over the latent functions, and the sum over the GH grid.
#         return tf.reduce_prod(cdfs, axis=[1]) @ tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))

# """Build the Heteroscedastic Gaussian Process for Classification"""
# class HeteroscedasticMultiClass(Likelihood):
#     def __init__(self, num_classes, invlink=None, **kwargs):
#         """
#         A likelihood for multi-way classification.  Currently the only valid
#         choice of inverse-link function (invlink) is an instance of RobustMax.
#         For most problems, the stochastic `Softmax` likelihood may be more
#         appropriate (note that you then cannot use Scipy optimizer).
#         """
#         super().__init__(input_dim = None, latent_dim=num_classes, observation_dim=None, **kwargs)
#         self.num_classes = num_classes
#         self.num_gauss_hermite_points = 20

#         if invlink is None:
#             invlink = RobustMax(self.num_classes)

#         self.invlink = invlink

#     def _log_prob(self, F, Y):
#         hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
#         yes = tf.ones(tf.shape(Y), dtype=default_float()) - self.invlink.epsilon
#         no = tf.zeros(tf.shape(Y), dtype=default_float()) + self.invlink.eps_k1
#         p = tf.where(hits, yes, no)
#         return tf.reduce_sum(tf.math.log(p), axis=-1)

#     def _variational_expectations(self, Fmu, Fvar, Y):
#         gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
#         p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
#         ve = p * tf.math.log(1.0 - self.invlink.epsilon) + (1.0 - p) * tf.math.log(
#             self.invlink.eps_k1
#         )
#         return tf.reduce_sum(ve, axis=-1)

#     def _predict_mean_and_var(self, Fmu, Fvar):
#         possible_outputs = [
#             tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64))
#             for i in range(self.num_classes)
#         ]
#         ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
#         ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
#         return ps, ps - tf.square(ps)

#     def _predict_log_density(self, Fmu, Fvar, Y):
#         return tf.reduce_sum(tf.math.log(self._predict_non_logged_density(Fmu, Fvar, Y)), axis=-1)

#     def _predict_non_logged_density(self, Fmu, Fvar, Y):
#         gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
#         p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
#         den = p * (1.0 - self.invlink.epsilon) + (1.0 - p) * (self.invlink.eps_k1)
#         return den

#     def _conditional_mean(self, F):
#         return self.invlink(F)

#     def _conditional_variance(self, F):
#         p = self.conditional_mean(F)
#         return p - tf.square(p)


###################################################################################
class HeteroscedasticRobustMax(Module):
    r"""
    This class represent a multi-class inverse-link function. Given a vector
    :math:`f=[f_1, f_2, ... f_k]`, the result of the mapping is

    .. math::

       y = [y_1 ... y_k]

    with

    .. math::

       y_i = \left\{
       \begin{array}{ll}
           (1-\varepsilon)   & \textrm{if} \ i = \textrm{argmax}(f) \\
           \varepsilon/(k-1) & \textrm{otherwise}
       \end{array}
       \right.

    where :math:`k` is the number of classes.
    """

    @check_shapes(
        "epsilon: []",
    )
    def __init__(
        self, num_classes: int, Sigma_y, epsilon: float = 1e-3, **kwargs: Any
    ) -> None:
        """
        Sigma_y is the membership probability of each point to each class. Rows must sum to unity.


        `epsilon` represents the fraction of 'errors' in the labels of the
        dataset. This may be a hard parameter to optimize, so by default
        it is set un-trainable, at a small value.
        """
        super().__init__(**kwargs)
        transform = tfp.bijectors.Sigmoid()
        prior = tfp.distributions.Beta(to_default_float(0.2), to_default_float(5.0))
        self.epsilon = Parameter(
            epsilon, transform=transform, prior=prior, trainable=False
        )
        self.num_classes = num_classes
        self._squash = 1e-6
        self.Sigma_y = Sigma_y  # Sigma_y is the label probabilities

    @check_shapes(
        "F: [broadcast batch..., latent_dim]",
        "return: [batch..., latent_dim]",
    )
    def __call__(self, F: TensorType) -> tf.Tensor:
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, 1.0 - self.epsilon, self.eps_k1)

    @property  # type: ignore[misc]  # Mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def eps_k1(self) -> tf.Tensor:
        return self.epsilon / (self.num_classes - 1.0)

    @check_shapes(
        "val: [batch...]",
        "return: [batch...]",
    )
    def safe_sqrt(self, val: TensorType) -> tf.Tensor:
        return tf.sqrt(tf.maximum(val, 1e-10))

    @check_shapes(
        "Y: [broadcast batch..., observation_dim]",
        "mu: [broadcast batch..., latent_dim]",
        "var: [broadcast batch..., latent_dim]",
        "gh_x: [n_quad_points]",
        "gh_w: [n_quad_points]",
        "return: [batch..., observation_dim]",
    )
    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = to_default_int(Y)
        # work out what the mean and variance is of the indicated latent function.
        """If we have the class membership probabilties, train on those. 
        Otherwise use a one hot encoding of the labels"""
        if mu.shape == self.Sigma_y.shape:
            oh_on = self.Sigma_y
        else:
            oh_on = tf.cast(
                tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1.0, 0.0),
                dtype=mu.dtype,
            )

        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            self.safe_sqrt(2.0 * var_selected), (-1, 1)
        )

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            self.safe_sqrt(var), 2
        )
        cdfs = 0.5 * (1.0 + tf.math.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2 * self._squash) + self._squash

        # blank out all the distances on the selected latent function
        # oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0.0, 1.0), dtype=mu.dtype)
        oh_off = tf.ones_like(oh_on) - oh_on
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.reduce_prod(cdfs, axis=[1]) @ tf.reshape(
            gh_w / np.sqrt(np.pi), (-1, 1)
        )


class HeteroscedasticMultiClass(Likelihood):
    def __init__(
        self, num_classes: int, invlink: Optional[RobustMax] = None, **kwargs: Any
    ) -> None:
        """
        A likelihood for multi-way classification.  Currently the only valid
        choice of inverse-link function (invlink) is an instance of RobustMax.

        For most problems, the stochastic `Softmax` likelihood may be more
        appropriate (note that you then cannot use Scipy optimizer).
        """
        super().__init__(
            input_dim=None, latent_dim=num_classes, observation_dim=None, **kwargs
        )
        self.num_classes = num_classes
        self.num_gauss_hermite_points = 20

        if invlink is None:
            invlink = RobustMax(self.num_classes)

        # if not isinstance(invlink, RobustMax):
        #     raise NotImplementedError

        self.invlink = invlink

    @inherit_check_shapes
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
        yes = tf.ones(tf.shape(Y), dtype=default_float()) - self.invlink.epsilon
        no = tf.zeros(tf.shape(Y), dtype=default_float()) + self.invlink.eps_k1
        p = tf.where(hits, yes, no)
        return tf.reduce_sum(tf.math.log(p), axis=-1)

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        ve = p * tf.math.log(1.0 - self.invlink.epsilon) + (1.0 - p) * tf.math.log(
            self.invlink.eps_k1
        )
        return tf.reduce_sum(ve, axis=-1)

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        possible_outputs = [
            tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64))
            for i in range(self.num_classes)
        ]
        ps = [
            self._predict_non_logged_density(X, Fmu, Fvar, po)
            for po in possible_outputs
        ]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
        return ps, ps - tf.square(ps)

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return tf.reduce_sum(
            tf.math.log(self._predict_non_logged_density(X, Fmu, Fvar, Y)), axis=-1
        )

    @check_shapes(
        "X: [broadcast batch..., input_dim]",
        "Fmu: [broadcast batch..., latent_dim]",
        "Fvar: [broadcast batch..., latent_dim]",
        "Y: [broadcast batch..., observation_dim]",
        "return: [batch..., observation_dim]",
    )
    def _predict_non_logged_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        den = p * (1.0 - self.invlink.epsilon) + (1.0 - p) * (self.invlink.eps_k1)
        return den

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - tf.square(p)
