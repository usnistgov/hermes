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


