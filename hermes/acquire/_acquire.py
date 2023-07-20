"""Classes of Acquisition Functions"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from pydantic import Field
from pydantic.dataclasses import dataclass as typesafedataclass
from scipy.special import erf


@dataclass
class Acquisition:
    """Base level class for acquisiton functions"""

    unmeasured_locations: np.ndarray
    mean: np.ndarray
    var: np.ndarray


@dataclass
class Random(Acquisition):
    def calculate(self):
        next = np.random.randint(0, self.mean.shape[0])

        next_loc = self.unmeasured_locations[next]
        return next_loc


@dataclass
class PureExploit(Acquisition):
    def calculate(self):
        next = np.argmax(self.mean)

        next_loc = self.unmeasured_locations[next]
        return next_loc


@dataclass
class PureExplore(Acquisition):
    def calculate(self):
        next = np.argmax(self.var)

        next_loc = self.unmeasured_locations[next].reshape(
            -1, self.unmeasured_locations.shape[1]
        )
        return next_loc


@dataclass
class UpperConfidenceBound(Acquisition):
    num_sigmas = 1.96

    def calculate(self):
        next = np.argmax(self.mean + self.num_sigmas * np.sqrt(self.var))

        next_loc = self.unmeasured_locations[next]
        return next_loc


@dataclass
class ScheduledUpperConfidenceBound(Acquisition):
    num_measurements: np.array

    num_sigmas = 1.96

    def calculate(self):
        beta = self.num_sigmas * self.num_measurements / 2
        next = np.argmax(self.mean + beta * np.sqrt(self.var))

        next_loc = self.unmeasured_locations[next]
        return next_loc


@dataclass
class ThompsonSampling(Acquisition):
    full_cov: tf.Tensor
    batch_size: int

    def calculate(self):
        # Sum full covariance to get NxN matrix
        cov = tf.reduce_sum(self.full_cov, axis=0)

        # Define a multivariate normal distribution and take draws from that
        Y = np.random.multivariate_normal(
            np.array(self.mean).flatten(), np.array(cov), size=self.batch_size
        )

        # Find the max point in each draw
        next_points = []
        for y in Y:
            next = np.argmax(y)
            next_points.append(next)

        next_args = np.array(next_points)

        next_loc = self.unmeasured_locations[next_args]
        return next_loc


@dataclass
class ProbabilityofImprovement(Acquisition):
    measurements: np.ndarray

    def calculate(self):
        y_best = np.max(self.measurements)
        # Calculate the probability of finding a measurment above the current best
        PoI = 1 - 0.5 * (1 + erf((y_best - self.mean) / (np.sqrt(self.var * 2))))

        # Choose the highest probability
        next = np.argmax(PoI)

        next_loc = self.unmeasured_locations[next]
        return next_loc


@dataclass
class ExpectedImprovement(Acquisition):
    measurements: np.ndarray

    def calculate(self):
        y_best = np.max(self.measurements)

        # alpha = (y_best - mean)/np.sqrt(var)
        alpha = (self.mean - y_best) / np.sqrt(self.var)

        phi_alpha = np.exp(-0.5 * (alpha - 0) ** 2 / 1) / (np.sqrt(1 * 2 * np.pi))
        Phi_alpha = 0.5 * (1 + erf((alpha - 0) / (np.sqrt(1 * 2))))

        # Calculate the expected improvement
        EI = (self.mean - y_best) * (Phi_alpha) + np.sqrt(self.var) * phi_alpha

        # Choose the highest expected improvement
        next = np.argmax(EI)

        next_loc = self.unmeasured_locations[next]
        return next_loc
