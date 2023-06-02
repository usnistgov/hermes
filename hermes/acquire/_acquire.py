"""Classes of Acquisition Functions"""
from dataclasses import dataclass
from pydantic.dataclasses import dataclass as typesafedataclass
from pydantic import Field
from pathlib import Path
from typing import Optional

import numpy as np

@dataclass
class Acquisition:
    """Base level class for acquisiton functions"""

    mean: np.array
    var: np.array

@dataclass
class Random(Acquisition):

    def calculate(self):
        next = np.random.randint(0, self.mean.shape[0])
        return next


@dataclass
class PureExploit(Acquisition):

    def calculate(self):
        next = np.argmax(self.mean)
        return next

@dataclass
class PureExplore(Acquisition):

    def calculate(self):
        next = np.argmax(self.var)
        return next

@dataclass
class UpperConfidenceBound(Acquisition):
    num_sigmas = 1.96

    def calculate(self):
        next = np.argmax(self.mean + num_sigmas*np.sqrt(self.var))
        return next
    
@dataclass
class ScheduledUpperConfidenceBound(Acquisition):
    num_measurements: np.array
    
    num_sigmas = 1.96
    
    def calculate(self):
        beta = num_sigmas*num_measurements/2
        next = np.argmax(self.mean + beta*np.sqrt(self.var))

@dataclass
class ThompsonSampling(Acquisition):

    unmeasured_locations: np.array
    full_cov: np.array

    def calculate(self):
        # cov = 
