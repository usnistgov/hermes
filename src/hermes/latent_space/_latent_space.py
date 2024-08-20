import numpy as np
from sklearn.decomposition import PCA

from dataclasses import field
from pathlib import Path
from typing import Any, Optional

from pydantic.dataclasses import dataclass as typesafedataclass

from hermes._base import Analysis

from hermes.utils import _check_attr, _default_ndarray

logger = logging.getLogger("hermes")



class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True



@typesafedataclass(config=_Config)
class LatentSpace(Analysis):
    """Class for creating latent spaces.
    Latent Space Algorithms take in an array of measurements and return an array of a smaller dimension.

    Attributes
    ----------
    measurements: np.ndarray
        Array of all the measurements in shape N x D for N measurements of dimension D.
        
    latent_space: np.ndarray
        Array of all the latent space descriptions in shape N x M, for M < D (typically) 
        
    """
    measurements: np.ndarray

    latent_space: np.ndarray = field(
                                    init=False,
                                    default_factory=_default_ndarray,
                                    repr=False,
                                    # init=False, repr=False
                                    )


@typesafedataclass(config=_Config)
class PCA(LatentSpace):
    """Class for Principle Component Analysis
    
     Attributes
    ----------
    loading_vectors: np.ndarray
        Array of vectors (aka principle componets) used for the representation.
        In PCA these are linear combinations of the original decriptors (aka measurements)

    explained_variance: np.ndarray
        Array showing the proportion of variance explained by each vector. 
        Sums to 1 over all the vectors.

    Methods
    -------
    calc_latent_space(self)
        Calculates the latent space representation of the measurements.
    """

    def calc_latent_space(self):
        #fit PCA to the measurements
        pca = PCA().fit(self.measurements)
        #transform the measurements to the latent space
        self.latent_space = pca.transform(self.measurements)
        
        #store the loading vectors
        self.loading_vectors = pca.components_
        #store the explained variance
        self.explained_variance = pca.explained_variance_
 

