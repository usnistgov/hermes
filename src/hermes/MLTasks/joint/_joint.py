# -*- coding: utf-8 -*-
# pylint: disable=C0103, E0401, E0611
"""
Created on Tue Sep 27 11:57:27 2022

@author: Aaron Kusne, aaron.kusne@nist.gov

These are tools for jointly infering the material structure and functional property maps.
The joint inference is based in the material science concept
that the structure is informative of the functional property, and vice versa.
Therefore changes in the structure should be informative of changes in the functional property.
And changes in the functional property should be informative of a change in the structure. 
"""

import logging
import time
from dataclasses import field
from typing import Any, Optional, Union

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as ndist
import pyro
import pyro.distributions as dist
import torch
from jax import config
from jax.lax import dynamic_slice
from numpyro import handlers
from numpyro.infer import MCMC as nMCMC
from numpyro.infer import NUTS as nNUTS
from numpyro.infer import SVI as nSVI
from numpyro.infer import Trace_ELBO as nTrace_ELBO
from numpyro.infer.initialization import init_to_value
from pydantic.dataclasses import dataclass as typesafedataclass
from pyro.infer import MCMC, NUTS, Predictive
from scipy.stats import entropy
from torch.nn.functional import one_hot
from tqdm import tqdm, trange

from hermes.base import Analysis
from hermes.utils import _default_ndarray

logger = logging.getLogger("hermes")
torch.set_default_dtype(torch.float64)


config.update("jax_enable_x64", True)


class UnspecifiedType(Exception):
    """Raised when no Distance or Similarity type is specified."""


class _Config:  # pylint: disable=too-few-public-methods
    arbitrary_types_allowed = True
    # validate_assignment = True


@typesafedataclass(config=_Config)
class Joint(Analysis):
    """Class for Joint algorithms.

    Automatically re-calculate all the distances and similarities when the atributes are set.
    This prevents miss-labeling the distances when the type is changed after the initial calcuation.
    """


@typesafedataclass(config=_Config)
class SAGE1D(Joint):
    """
    Joint segmentation and regression algorithms.
    Currently set up for only one change point.

    Attributes
    ----------
    num_phase_regions : int
        regions
    num_samples : int
        samples
    num_warmup : int
        warmup
    target_accept_prob : float
        accept
    max_tree_depth : int
        max
    jitter : float
        jitter
    locations_structure : np.ndarray, default=np.array([])
        description
    locations_functional_property : np.ndarray, default=np.array([])
        description
    locations_prediction : np.ndarray, default=np.array([])
        description
    target_structure_labels : np.ndarray, default=np.array([])
        description
    target_functional_properties : np.ndarray, default=np.array([])
        description
    gpr_bias_bounds : np.ndarray, default=np.array([])
        description
    gpr_variance_bounds : np.ndarray, default=np.array([])
        description
    gpr_lengthscale_bounds : np.ndarray, default=np.array([])
        description
    gpr_noise_bounds : np.ndarray, default=np.array([])
        description
    change_point_bounds : np.ndarray, default=np.array([])
        description
    predictions : np.ndarray, default=np.array([])
        description


    Methods
    -------
    run()
        Run model.
    model_SAGE_1D()
        Model for SAGE 1D.
    predict_SAGE_1D()
        Predict the phase region labels and functional properties for new data.

    """

    # needed inputs
    num_phase_regions: int
    num_samples: int
    num_warmup: int
    target_accept_prob: float
    max_tree_depth: int
    jitter: float

    locations_structure: np.ndarray = field(default_factory=_default_ndarray)
    locations_functional_property: np.ndarray = field(default_factory=_default_ndarray)
    locations_prediction: np.ndarray = field(default_factory=_default_ndarray)
    target_structure_labels: np.ndarray = field(default_factory=_default_ndarray)
    target_functional_properties: np.ndarray = field(default_factory=_default_ndarray)

    gpr_bias_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_variance_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_lengthscale_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_noise_bounds: np.ndarray = field(default_factory=_default_ndarray)
    change_point_bounds: np.ndarray = field(default_factory=_default_ndarray)

    # Outputs
    # TODO {} inmutable?
    predictions: dict[str, Any] = {}

    def run(self) -> None:
        """Run the model and update the internal state of the object.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value, it performs the predictions and
            saves the results in `self.predictions`.

        """
        xs = to_torch(self.locations_structure).double()
        ys = to_torch(self.target_structure_labels).long()
        xf = to_torch(self.locations_functional_property).double()
        yf = to_torch(self.target_functional_properties).double()
        gpr_var_bounds = to_torch(self.gpr_variance_bounds).double()
        gpr_ls_bounds = to_torch(self.gpr_lengthscale_bounds).double()
        gpr_noise_bounds = to_torch(self.gpr_noise_bounds).double()
        gpr_bias_bounds = to_torch(self.gpr_noise_bounds).double()
        cp_bounds = to_torch(self.change_point_bounds).double()
        Xpred = to_torch(self.locations_prediction).double()
        num_regions = self.num_phase_regions

        nuts = MCMC(
            NUTS(
                self.model_SAGE_1D,
                target_accept_prob=self.target_accept_prob,
                max_tree_depth=self.max_tree_depth,
            ),
            num_samples=self.num_samples,
            warmup_steps=self.num_warmup,
        )
        nuts.run(
            xs,
            ys,
            xf,
            yf,
            num_regions,
            gpr_var_bounds=gpr_var_bounds,
            gpr_ls_bounds=gpr_ls_bounds,
            gpr_noise_bounds=gpr_noise_bounds,
            cp_bounds=cp_bounds,
            gpr_bias_bounds=gpr_bias_bounds,
        )

        nuts_posterior_samples = nuts.get_samples()

        predictive = Predictive(self.model_SAGE_1D, nuts_posterior_samples)(
            xs,
            ys,
            xf,
            yf,
            num_regions,
            gpr_var_bounds=gpr_var_bounds,
            gpr_ls_bounds=gpr_ls_bounds,
            gpr_noise_bounds=gpr_noise_bounds,
            cp_bounds=cp_bounds,
            gpr_bias_bounds=gpr_bias_bounds,
        )

        idx = torch.argmax(predictive["llk"])
        max_llk_sample_cp = nuts_posterior_samples["changepoint"][idx]
        # TODO unused var
        (
            gpc_probs_mllk,
            gpr_mean_noiseless_mllk,
            gpr_samples,
            _,
            gpr_var_noiseless_mllk,
        ) = self.predict_SAGE_1D(
            nuts_posterior_samples, idx, xs, Xpred, num_regions, xf, yf
        )

        preds = [
            self.predict_SAGE_1D(
                nuts_posterior_samples, i, xs, Xpred, num_regions, xf, yf
            )
            for i in trange(nuts_posterior_samples["gpr_var"].shape[0])
        ]

        phase_region_labels_mean = np.stack([item[0] for item in preds]).mean(axis=0)
        phase_region_labels_std = np.stack([item[0] for item in preds]).std(axis=0)

        # squeeze assumes 1 functional property:
        # TODO unused var
        gpr_samples_noiseless = np.stack([item[1] for item in preds], axis=2).squeeze(
            axis=1
        )
        gpr_samples = np.stack([item[2] for item in preds], axis=2).squeeze(axis=1)

        # assume Mf == 1
        functional_properties_mean = gpr_samples.mean(axis=1)
        functional_properties_std = gpr_samples.std(axis=1)

        cp_samples = np.array(nuts_posterior_samples["changepoint"].flatten())
        cp_mean = cp_samples.mean()
        cp_std = cp_samples.std()

        # gpc_new_mean, gpc_new_std, gpr_new_mean, gpr_new_var, cp_samples, gpr_mean_noiseless_mllk, gpr_var_noiseless_mllk
        self.predictions = {
            "phase_region_labels_mean": phase_region_labels_mean,
            "phase_region_labels_std": phase_region_labels_std,
            "functional_property_mean": functional_properties_mean,
            "functional_property_std": functional_properties_std,
            "change_point_samples": cp_samples,
            "change_point_mean": cp_mean,
            "change_point_std": cp_std,
            "max_likelihood_functional_property_noiseless_mean": gpr_mean_noiseless_mllk.squeeze(),
            "max_likelihood_functional_property_noiseless_var": gpr_var_noiseless_mllk.squeeze(),
            "max_likelihood_change_points": max_llk_sample_cp,
        }

    def model_SAGE_1D(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xf: torch.Tensor,
        yf: torch.Tensor,
        num_regions: int,
        gpr_var_bounds: Union[tuple, list],
        gpr_ls_bounds: Union[tuple, list],
        gpr_noise_bounds: Union[tuple, list],
        cp_bounds: Union[tuple, list],
        gpr_bias_bounds: Union[tuple, list],
    ):
        """
        Model for SAGE 1D.

        Parameters
        ----------
            xs : torch.Tensor
                Input tensor for training samples.
            ys : torch.Tensor
                Target tensor for training samples.
            xf : torch.Tensor
                Input tensor for test samples.
            yf : torch.Tensor
                Target tensor for test samples.
            num_regions : int
                Number of regions.
            gpr_var_bounds : tuple, list
                Bounds for Gaussian Process Regression (GPR) variance.
            gpr_ls_bounds : tuple, list
                Bounds for GPR lengthscale.
            gpr_noise_bounds : tuple, list
                Bounds for GPR noise.
            cp_bounds : tuple, list
                Bounds for changepoint.
            gpr_bias_bounds : tuple, list
                Bounds for GPR bias.

        Returns
        -------
        None
            This function does not return any value. It performs the prediction and updates the internal state of the object.
        """
        Mf = yf.shape[1]
        Nf = xf.shape[0]
        Ns = xs.shape[0]
        # TODO: Nsf not used
        Nsf = Ns + Nf
        Xsf = torch.vstack((xs, xf))

        changepoint_min_bound = cp_bounds[0] * torch.ones((num_regions - 1))
        changepoint_max_bound = cp_bounds[1] * torch.ones((num_regions - 1))
        gpr_var_min_bound = gpr_var_bounds[0] * torch.ones((num_regions, Mf))
        gpr_var_max_bound = gpr_var_bounds[1] * torch.ones((num_regions, Mf))
        gpr_ls_min_bound = gpr_ls_bounds[0] * torch.ones((num_regions, Mf))
        gpr_ls_max_bound = gpr_ls_bounds[1] * torch.ones((num_regions, Mf))
        gpr_bias_min_bound = gpr_bias_bounds[0] * torch.ones((num_regions, Mf))
        gpr_bias_max_bound = gpr_bias_bounds[1] * torch.ones((num_regions, Mf))

        changepoint = pyro.sample(
            "changepoint", dist.Uniform(changepoint_min_bound, changepoint_max_bound)
        )
        gpr_noise = pyro.sample(
            "gpr_noise", dist.Uniform(gpr_noise_bounds[0], gpr_noise_bounds[1])
        )
        gpr_var = pyro.sample(
            "gpr_var", dist.Uniform(gpr_var_min_bound, gpr_var_max_bound)
        )
        gpr_lengthscale = pyro.sample(
            "gpr_lengthscale", dist.Uniform(gpr_ls_min_bound, gpr_ls_max_bound)
        )
        gpr_bias = pyro.sample(
            "gpr_bias", dist.Uniform(gpr_bias_min_bound, gpr_bias_max_bound)
        )

        region_labels = change_points_to_labels_torch(changepoint, Xsf)

        probs = one_hot(region_labels, num_regions)
        probs_fp = probs[Ns:, :]

        F = torch.zeros((Nf, num_regions, Mf))
        for j in range(Mf):
            for i in range(num_regions):
                with pyro.plate("latent_response" + str(i), Nf):
                    eta = pyro.sample(
                        "sample" + str(i) + "_Mf" + str(j), dist.Normal(0, 1)
                    )

                f = compute_f_torch(
                    gpr_var[i, j],
                    gpr_lengthscale[i, j],
                    gpr_bias[i, j],
                    eta,
                    xf,
                    self.jitter,
                )
                F[:, i, j] = f

        f_piecewise = torch.zeros((Nf, Mf))
        for j in range(Mf):
            for i in range(num_regions):
                f_piecewise[:, j] = f_piecewise[:, j] + probs_fp[:, i] * F[:, i, j]

        llk = dist.Categorical(probs=probs[:Ns, :]).log_prob(ys.flatten()).sum()

        for j in range(Mf):
            llk = (
                llk
                + dist.Normal(f_piecewise[:, j], torch.sqrt(gpr_noise))
                .log_prob(yf[:, j])
                .sum()
            )

        pyro.deterministic("llk", llk)
        pyro.factor("obs", llk)

    # TODO: do docstring (Camilo)
    # TODO: add type hints
    # TODO: xs not used
    def predict_SAGE_1D(
        self,
        samples: dict[str, Any],
        i: int,
        xs: np.ndarray,
        Xnew: np.ndarray,
        num_regions: int,
        xf: np.ndarray,
        yf: np.ndarray,
    ):
        """
        Predict the phase region labels and functional properties for new data.

        Parameters
        ----------
        samples : dict
            Dictionary containing the samples for the Gaussian Process Regression (GPR) model.
        i : int
            Index of the sample to use from the `samples` dictionary.
        xs : np.ndarray
            Array containing the input data for the phase region labels.
        Xnew : np.ndarray
            Array containing the new input data for prediction.
        num_regions : int
            Number of phase regions.
        xf : np.ndarray
            Array containing the input data for the functional properties.
        yf : np.ndarray
            Array containing the target values for the functional properties.

        Returns
        -------
        tuple
            A tuple containing the following arrays:
            - probs (np.ndarray): Array of shape (Nnew, num_regions) representing
            the predicted probabilities for each phase region label.
            - f_piecewise (np.ndarray): Array of shape (Nnew, Mf, 1) representing
            the piecewise function values.
            - f_sample (np.ndarray): Array of shape (Nnew, Mf, 1) representing
            the sampled function values.
            - F (np.ndarray): Array of shape (Nnew, num_regions, Mf) representing
            the mean function values for each phase region and functional property.
            - v_piecewise (np.ndarray): Array of shape (Nnew, Mf, 1) representing
            the piecewise function variances.
        """
        Nnew = Xnew.shape[0]
        Mf = yf.shape[1]
        gpr_new_mean_regions = torch.zeros((Nnew, Mf, num_regions))
        gpr_new_var_regions = torch.zeros((Nnew, Mf, num_regions))
        gpr_new_cov_regions = torch.zeros((Nnew, Nnew, Mf, num_regions))
        gpr_new_mean_mixture = torch.zeros((Nnew, Mf))
        gpr_new_var_mixture = torch.zeros((Nnew, Mf))

        region_labels = change_points_to_labels_torch(samples["changepoint"][i], Xnew)
        probs = one_hot(region_labels, num_regions)

        F = torch.zeros((Nnew, num_regions, Mf))
        V = torch.zeros((Nnew, num_regions, Mf))
        for k in range(Mf):
            for j in range(num_regions):
                eta = samples["sample" + str(j) + "_Mf" + str(k)][i]
                f = compute_f_torch(
                    samples["gpr_var"][i, j, k],
                    samples["gpr_lengthscale"][i, j, k],
                    samples["gpr_bias"][i, j, k],
                    eta,
                    xf,
                    self.jitter,
                )
                mean, _, var = gpr_forward_torch(
                    samples["gpr_var"][i, j, k],
                    samples["gpr_lengthscale"][i, j, k],
                    xf,
                    f,
                    Xnew,
                    samples["gpr_noise"][i],
                    include_noise=False,
                    jitter=self.jitter,
                )
                F[:, j, k] = mean
                V[:, j, k] = var

        f_piecewise = torch.zeros((Nnew, Mf, 1))
        v_piecewise = torch.zeros((Nnew, Mf, 1))
        f_sample = torch.zeros((Nnew, Mf, 1))
        for k in range(Mf):
            for j in range(num_regions):
                f_piecewise[:, k, 0] = f_piecewise[:, k, 0] + probs[:, j] * F[:, j, k]
                v_piecewise[:, k, 0] = v_piecewise[:, k, 0] + probs[:, j] * V[:, j, k]
            f_sample[:, k, 0] = dist.Normal(
                f_piecewise[:, k, 0], torch.sqrt(samples["gpr_noise"][i])
            ).sample()

        # return np.array(probs), np.array(gpr_new_mean_mixture), np.array(gpr_new_var_mixture)
        return (
            np.array(probs),
            np.array(f_piecewise),
            np.array(f_sample),
            np.array(F),
            np.array(v_piecewise),
        )

        # TODO: the same as line 1515
        # if type(X) is np.ndarray:
        #     X = torch.tensor(X)
        # cp, _ = torch.sort(cp)
        # cl = torch.zeros((X.shape[0])).long()
        # N = cp.shape[0]  # N = 3
        # for i in range(0, N):
        #     if i < N - 1:
        #         idx = torch.logical_and(X > cp[i], X < cp[i + 1])
        #     elif i == N - 1:
        #         idx = X > cp[i]
        #     cl[idx.flatten()] = i + 1
        # return cl


@typesafedataclass(config=_Config)
class SAGEND(Joint):
    """Joint segmentation and regression algorithms.

    This class assumes that there is 1 set structure measument discrite labels.
    There can be multiple sets of functional property measurements, if they are measured at the same location.

    Attributes
    ----------
    num_phase_regions : int
        regions
    num_samples : int
        samples
    num_warmup : int
        warmup
    num_chains : int
        chains
    target_accept_prob : float
        accept
    max_tree_depth : int
        max
    phase_map_SVI_num_steps : int
        steps
    jitter : float
        jitter
    Adam_step_size : float
        step
    posterior_sampling : int
        sampling
    locations_structure : np.ndarray, default=np.array([])
        description
    locations_functional_property : np.ndarray, default=np.array([])
        description
    locations_prediction : np.ndarray, default=np.array([])
        description
    target_structure_labels : np.ndarray, default=np.array([])
        description
    target_functional_properties : np.ndarray, default=np.array([])
        description
    gpc_variance_bounds : np.ndarray, default=np.array([])
        description
    gpc_lengthscale_bounds : np.ndarray, default=np.array([])
        description
    gpr_bias_bounds : np.ndarray, default=np.array([])
        description
    gpr_variance_bounds : np.ndarray, default=np.array([])
        description
    gpr_lengthscale_bounds : np.ndarray, default=np.array([])
        description
    gpr_noise_bounds : np.ndarray, default=np.array([])
        description
    key : np.ndarray, default=np.array([])
        description, not initialized by user

    Methods
    -------
    run()
        Run model.
    model_SAGE_ND()
        Model for SAGE ND.
    model_SAGE_ND_PM()
        Model for SAGE ND PM.
    predict_SAGE_ND()
        Predict the phase region labels and functional properties for new data.
    """

    # needed inputs
    num_phase_regions: int
    num_samples: int
    num_warmup: int
    num_chains: int
    target_accept_prob: float
    max_tree_depth: int
    phase_map_SVI_num_steps: int
    jitter: float
    Adam_step_size: float
    posterior_sampling: int

    locations_structure: np.ndarray = field(default_factory=_default_ndarray)
    locations_functional_property: np.ndarray = field(default_factory=_default_ndarray)
    locations_prediction: np.ndarray = field(default_factory=_default_ndarray)
    target_structure_labels: np.ndarray = field(default_factory=_default_ndarray)
    target_functional_properties: np.ndarray = field(default_factory=_default_ndarray)

    gpc_variance_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpc_lengthscale_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_bias_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_variance_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_lengthscale_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_noise_bounds: np.ndarray = field(default_factory=_default_ndarray)

    key = jax.random.PRNGKey(0)

    # Outputs
    predictions: dict[str, Any] = {}

    def run(self):
        """Run the model and update the internal state of the object.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value, it performs the predictions and
            saves the results in `self.predictions`.

        """
        xs = jnp.asarray(self.locations_structure, dtype=jnp.float64)
        ys = jnp.asarray(self.target_structure_labels, dtype=jnp.integer)
        xf = jnp.asarray(self.locations_functional_property, dtype=jnp.float64)
        yf = jnp.asarray(self.target_functional_properties, dtype=jnp.float64)

        gpc_var_bounds = jnp.asarray(self.gpc_variance_bounds).copy()
        gpc_ls_bounds = jnp.asarray(self.gpc_lengthscale_bounds).copy()
        gpr_var_bounds = jnp.asarray(self.gpr_variance_bounds).copy()
        gpr_ls_bounds = jnp.asarray(self.gpr_lengthscale_bounds).copy()
        gpr_noise_bounds = jnp.asarray(self.gpr_noise_bounds).copy()
        gpr_bias_bounds = jnp.asarray(self.gpr_noise_bounds).copy()
        Xpred = jnp.asarray(self.locations_prediction).copy()
        num_regions = self.num_phase_regions
        num_proc = self.num_chains

        # TODO make this private?
        def predict_sage(post_samples, model, *args, **kwargs):
            model = handlers.seed(handlers.condition(model, post_samples), self.key)
            model_trace = handlers.trace(model).get_trace(*args, **kwargs)
            return (
                model_trace["Fr_new"]["value"],
                model_trace["f_piecewise"]["value"],
                model_trace["f_sample"]["value"],
                model_trace["gpc_new_probs"]["value"],
                model_trace["v_piecewise"]["value"],
            )

        # TODO many vars unused, necessary lambda?
        def predict_fn_sage_1core(samples):
            return predict_sage(
                samples,
                self.predict_SAGE_ND,
                Xpred,
                xs,
                ys,
                xf,
                yf,
                num_regions=num_regions,
            )

        data = [xs, ys, xf, num_regions, gpc_var_bounds, gpc_ls_bounds]

        autoguide_mle = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(
            self.model_SAGE_ND_PM
        )
        optimizer = numpyro.optim.Adam(step_size=self.Adam_step_size)

        svi = nSVI(self.model_SAGE_ND_PM, autoguide_mle, optimizer, loss=nTrace_ELBO())
        svi_result = svi.run(self.key, self.phase_map_SVI_num_steps, *data)

        params = svi_result.params
        mle_st = autoguide_mle.median(params)

        gpc_latent_ = jnp.vstack((mle_st["gpc_latent_0"], mle_st["gpc_latent_1"]))

        preds_fp = None
        preds_st = None

        init_params = {
            "gpc_latent_0": mle_st["gpc_latent_0"],
            "gpc_latent_1": mle_st["gpc_latent_1"],
            "gpc_var": mle_st["gpc_var"],
            "gpc_lengthscale": mle_st["gpc_lengthscale"],
            "gpc_bias": mle_st["gpc_bias"],
        }
        init_strategy = init_to_value(values=init_params)

        tic = time.perf_counter()
        nuts = nMCMC(
            nNUTS(
                self.model_SAGE_ND,
                target_accept_prob=self.target_accept_prob,
                max_tree_depth=self.max_tree_depth,
                init_strategy=init_strategy,
            ),
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            num_chains=self.num_chains,
        )
        nuts.run(
            self.key,
            xs,
            ys,
            xf,
            yf,
            num_regions,
            gpc_var_bounds=gpc_var_bounds,
            gpc_ls_bounds=gpc_ls_bounds,
            gpr_var_bounds=gpr_var_bounds,
            gpr_ls_bounds=gpr_ls_bounds,
            gpr_bias_bounds=gpr_bias_bounds,
            gpr_noise_bounds=gpr_noise_bounds,
        )

        nuts_posterior_samples = nuts.get_samples()
        samples = subsample(nuts_posterior_samples, step=self.posterior_sampling)

        num_length = samples["gpr_noise"].shape[0]
        sl = split_samples(samples, num_proc, num_length)
        splits = np.array(num_length / num_proc).astype(int)

        predict_fn_sage = jax.pmap(
            lambda samples: predict_sage(
                samples,
                self.predict_SAGE_ND,
                Xnew=Xpred,
                xs=xs,
                ys=ys,
                xf=xf,
                yf=yf,
                num_regions=num_regions,
                gpc_var_bounds=gpc_var_bounds,
                gpc_ls_bounds=gpc_ls_bounds,
                gpr_var_bounds=gpr_var_bounds,
                gpr_ls_bounds=gpr_ls_bounds,
                gpr_bias_bounds=gpr_bias_bounds,
                gpr_noise_bounds=gpr_noise_bounds,
            ),
            axis_name=0,
        )

        print("starting pred analysis, for #", num_length)
        labels = ["Fr_new", "f_piecewise", "f_sample", "gpc_new_probs", "v_piecewise"]

        for i in trange(splits):
            if i == 0:
                preds = predict_fn_sage(sl[i])
                preds_stacked = {
                    labels[0]: preds[0],
                    labels[1]: preds[1],
                    labels[2]: preds[2],
                    labels[3]: preds[3],
                    labels[4]: preds[4],
                }
            else:
                preds = predict_fn_sage(sl[i])
                for j in range(len(labels)):
                    preds_stacked[labels[j]] = np.vstack(
                        (preds_stacked[labels[j]], preds[j])
                    )
        toc = time.perf_counter()
        print(f"Run in {toc - tic:0.4f} seconds")

        print("done pred analysis")

        output = {
            "preds": preds_stacked
        }  # , 'preds_st':preds_st, 'preds_fp':preds_fp, 'starting_data':starting_data}

        # ------------------

        preds_sage = output["preds"]
        # preds_st = output['preds_st']
        # preds_fp = output['preds_fp']
        # starting_data = output['starting_data']

        phase_region_labels_mean = np.nanmean(preds_sage["gpc_new_probs"], axis=0)
        phase_region_labels_std = np.nanstd(preds_sage["gpc_new_probs"], axis=0)
        phase_region_labels_mean_estimate = np.argmax(phase_region_labels_mean, axis=1)
        phase_region_labels_mean_entropy = entropy(phase_region_labels_mean, axis=1)
        functional_properties_mean = np.nanmean(preds_sage["f_piecewise"], axis=0)
        functional_properties_std = np.sqrt(
            np.nanmean(preds_sage["v_piecewise"], axis=0)
        )

        # gpc_new_mean, gpc_new_std, gpr_new_mean, gpr_new_var, cp_samples, gpr_mean_noiseless_mllk, gpr_var_noiseless_mllk
        self.predictions = {
            "phase_region_labels_mean_estimate": phase_region_labels_mean_estimate,
            "phase_region_labels_mean": phase_region_labels_mean,
            "phase_region_labels_mean_entropy": phase_region_labels_mean_entropy,
            "phase_region_labels_std": phase_region_labels_std,
            "functional_property_mean": functional_properties_mean,
            "functional_property_std": functional_properties_std,
        }

    def model_SAGE_ND(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xf: torch.Tensor,
        yf: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
        gpr_var_bounds: Union[tuple, list],
        gpr_ls_bounds: Union[tuple, list],
        gpr_bias_bounds: Union[tuple, list],
        gpr_noise_bounds: Union[tuple, list],
    ) -> None:
        """Model for SAGE ND.

        Parameters
        ----------
        xs : torch.Tensor
            Input tensor for training samples.
        ys : torch.Tensor
            Target tensor for training samples.
        xf : torch.Tensor
            Input tensor for test samples.
        yf : torch.Tensor
            Target tensor for test samples.
        num_regions : int
            Number of regions.
        gpc_var_bounds : tuple, list
            Bounds for Gaussian Process Classification (GPC) variance.
        gpc_ls_bounds : tuple, list
            Bounds for GPC lengthscale.
        gpr_var_bounds : tuple, list
            Bounds for Gaussian Process Regression (GPR) variance.
        gpr_ls_bounds : tuple, list
            Bounds for GPR lengthscale.
        gpr_bias_bounds : tuple, list
            Bounds for GPR bias.
        gpr_noise_bounds : tuple, list
            Bounds for GPR noise.

        Returns
        -------
        None
            This function does not return any value. It performs the prediction and updates the internal state of the object.

        """
        # TODO this returs None and
        # I cannot see what it modifies

        # assumes all function property measurements measured at same locations.
        Ns = ys.shape[0]  # number of observations of structure
        Nf = yf.shape[0]  # number of observations of functional properties
        Mf = yf.shape[1]  # number of functional properties that were measured.
        Nsf = xs.shape[0] + xf.shape[0]
        x_ = jnp.vstack((xs, xf))

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Priors: GPR
        gpr_var_bound_min = gpr_var_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_var_bound_max = gpr_var_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_min = gpr_ls_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_max = gpr_ls_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_min = gpr_bias_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_max = gpr_bias_bounds[1] * jnp.ones((num_regions, Mf))

        gpr_noise = numpyro.sample(
            "gpr_noise", ndist.Uniform(gpr_noise_bounds[0], gpr_noise_bounds[1])
        )
        gpr_var = numpyro.sample(
            "gpr_var", ndist.Uniform(gpr_var_bound_min, gpr_var_bound_max)
        )
        gpr_lengthscale_x = numpyro.sample(
            "gpr_lengthscale_x",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_lengthscale_y = numpyro.sample(
            "gpr_lengthscale_y",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_bias = numpyro.sample(
            "gpr_bias", ndist.Uniform(gpr_bias_bound_min, gpr_bias_bound_max)
        )

        # Get latent functions, one for each region (i.e., segment).
        Fc = jnp.zeros((Ns + Nf, num_regions))
        for i in range(num_regions):
            with numpyro.plate("gpc_latent_response" + str(i), Nsf):
                gpc_latent = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))

            f = compute_f_matern52_jax(
                gpc_var, gpc_lengthscale, gpc_bias, gpc_latent, x_, jitter=self.jitter
            )
            Fc = Fc.at[:, i].set(f)  # x = x.at[idx].set(y)

        probs = logits_to_probs_jax(Fc)
        probs_fp = probs[Ns:, :]

        # gpr for each region.
        Fr = jnp.zeros((Nf, num_regions, Mf))
        for j in range(Mf):
            for i in range(num_regions):
                with numpyro.plate("gpr_latent_response" + str(i), Nf):
                    gpr_latent = numpyro.sample(
                        "gpr_latent_" + str(i) + "_Mf_" + str(j), ndist.Normal(0, 1)
                    )

                gpr_lengthscale_array = jnp.array(
                    [gpr_lengthscale_x[i, j], gpr_lengthscale_y[i, j]]
                )
                f = compute_f_jax(
                    gpr_var[i, j],
                    gpr_lengthscale_array,
                    gpr_bias[i, j],
                    gpr_latent,
                    xf,
                    jitter=self.jitter,
                )
                Fr = Fr.at[:, i, j].set(f)

        f_piecewise = jnp.zeros((Nf, Mf))
        for j in range(Mf):
            for i in range(num_regions):
                f_piecewise = f_piecewise.at[:, j].set(
                    f_piecewise[:, j] + probs_fp[:, i] * Fr[:, i, j]
                )

        llk = ndist.Categorical(probs=probs[:Ns, :]).log_prob(ys.flatten()).sum()

        for j in range(Mf):
            llk = (
                llk
                + ndist.Normal(f_piecewise[:, j], jnp.sqrt(gpr_noise))
                .log_prob(yf[:, j])
                .sum()
            )

        numpyro.deterministic("llk", llk)
        numpyro.factor("obs", llk)  # likelihood of segmentation

    def model_SAGE_ND_PM(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xf: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
    ) -> None:
        """Model for SAGE ND PM.

        Parameters
        ----------
        xs : torch.Tensor
            Input tensor for training samples.
        ys : torch.Tensor
            Target tensor for training samples.
        xf : torch.Tensor
            Input tensor for test samples.
        num_regions : int
            Number of regions.
        gpc_var_bounds : tuple, list
            Bounds for Gaussian Process Classification (GPC) variance.
        gpc_ls_bounds : tuple, list
            Bounds for GPC lengthscale.

        Returns
        -------
        None
            This function does not return any value. It performs the prediction and updates the internal state of the object.
        """
        # assumes all function property measurements measured at same locations.
        Ns = xs.shape[0]
        Nf = xf.shape[0]
        Nsf = Ns + Nf

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Get latent functions, one for each region (i.e., segment).
        Fc = jnp.zeros((Ns, num_regions))
        for i in range(num_regions):
            with numpyro.plate("gpc_latent_response" + str(i), Nsf):
                gpc_latent = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))

            f = compute_f_matern52_jax(
                gpc_var,
                gpc_lengthscale,
                gpc_bias,
                gpc_latent[:Ns],
                xs,
                jitter=self.jitter,
            )
            Fc = Fc.at[:, i].set(f)  # x = x.at[idx].set(y)

        probs = logits_to_probs_jax(Fc)

        llk = ndist.Categorical(probs=probs[:Ns, :]).log_prob(ys.flatten()).sum()

        numpyro.deterministic("llk", llk)
        numpyro.factor("obs", llk)  # likelihood of segmentation

    def predict_SAGE_ND(
        self,
        Xnew: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xf: torch.Tensor,
        yf: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
        gpr_var_bounds: Union[tuple, list],
        gpr_ls_bounds: Union[tuple, list],
        gpr_bias_bounds: Union[tuple, list],
        gpr_noise_bounds: Union[tuple, list],
    ) -> tuple:
        """
        Predict the phase region labels and functional properties for new data.

        This function assumes that the phase region labels
        and functional properties are measured at the same locations.

        Parameters
        ----------
        Xnew : torch.Tensor
            The new data for which to predict the phase region labels and functional properties.
        xs : torch.Tensor
            The phase region labels measured at the same locations as the functional properties.
        ys : torch.Tensor
            The functional properties measured at the same locations as the phase region labels.
        xf : torch.Tensor
            The phase region labels for the training data.
        yf : torch.Tensor
            The functional properties for the training data.
        num_regions : int
            The number of phase regions.
        gpc_var_bounds : Union[tuple, list]
            The bounds for the variance of the Gaussian Process Classifier (GPC).
        gpc_ls_bounds : Union[tuple, list]
            The bounds for the lengthscale of the GPC.
        gpr_var_bounds : Union[tuple, list]
            The bounds for the variance of the Gaussian Process Regression (GPR).
        gpr_ls_bounds : Union[tuple, list]
            The bounds for the lengthscale of the GPR.
        gpr_bias_bounds : Union[tuple, list]
            The bounds for the bias of the GPR.
        gpr_noise_bounds : Union[tuple, list]
            The bounds for the noise of the GPR.

        Returns
        -------
        tuple
            A tuple containing the following arrays:
            - probs (np.ndarray): Array of shape (Nnew, num_regions) representing the predicted probabilities for each phase region label.
            - f_piecewise (np.ndarray): Array of shape (Nnew, Mf, 1) representing the piecewise function values.
            - f_sample (np.ndarray): Array of shape (Nnew, Mf, 1) representing the sampled function values.
            - F (np.ndarray): Array of shape (Nnew, num_regions, Mf) representing the mean function values for each phase region and functional property.
            - v_piecewise (np.ndarray): Array of shape (Nnew, Mf, 1) representing the piecewise function variances.

        """
        # assumes all function property measurements measured at same locations.
        key_in = self.key
        _, subkey = jax.random.split(key_in)
        jitter = self.jitter
        Ns = ys.shape[0]
        Nf = yf.shape[0]
        Mf = yf.shape[1]
        Nsf = xs.shape[0] + xf.shape[0]
        x_ = jnp.vstack((xs, xf))

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Priors: GPR
        gpr_var_bound_min = gpr_var_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_var_bound_max = gpr_var_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_min = gpr_ls_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_max = gpr_ls_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_min = gpr_bias_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_max = gpr_bias_bounds[1] * jnp.ones((num_regions, Mf))

        gpr_noise = numpyro.sample(
            "gpr_noise", ndist.Uniform(gpr_noise_bounds[0], gpr_noise_bounds[1])
        )
        gpr_var = numpyro.sample(
            "gpr_var", ndist.Uniform(gpr_var_bound_min, gpr_var_bound_max)
        )
        gpr_lengthscale_x = numpyro.sample(
            "gpr_lengthscale_x",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_lengthscale_y = numpyro.sample(
            "gpr_lengthscale_y",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_bias = numpyro.sample(
            "gpr_bias", ndist.Uniform(gpr_bias_bound_min, gpr_bias_bound_max)
        )

        # ------- added --------------
        Nnew = Xnew.shape[0]
        gpc_train_latent = jnp.zeros((x_.shape[0], num_regions))
        gpc_new_latent = jnp.zeros((Nnew, num_regions))
        gpc_new_probs = jnp.zeros((Nnew, num_regions))
        # get region labels

        gpc_latent = [0] * num_regions
        for i in range(num_regions):
            gpc_latent[i] = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))

        for j in range(num_regions):
            f = compute_f_matern52_jax(
                gpc_var,
                gpc_lengthscale,
                gpc_bias,
                gpc_latent[j],
                x_,
                jitter=self.jitter,
            )

            gpc_train_latent = gpc_train_latent.at[:, j].set(f)
            gpc_noise = self.jitter
            mean, cov, _ = gpr_forward_matern52_jax(
                gpc_var,
                gpc_lengthscale,
                x_,
                f,
                Xnew,
                gpc_noise,
                include_noise=False,
                jitter=self.jitter,
            )
            fhat = ndist.MultivariateNormal(
                mean, cov + jnp.eye(Nnew) * self.jitter
            ).sample(subkey)
            gpc_new_latent = gpc_new_latent.at[:, j].set(fhat)
        gpc_new_probs = logits_to_probs_jax(gpc_new_latent)
        # -----------------------------

        gpr_latent = [[0] * Mf for i in range(num_regions)]
        for j in range(Mf):
            for i in range(num_regions):
                gpr_latent[i][j] = numpyro.sample(
                    "gpr_latent_" + str(i) + "_Mf_" + str(j), ndist.Normal(0, 1)
                )

        # ---added -------------------------------------
        Fr_new = jnp.zeros((Nnew, num_regions, Mf))
        Vr_new = jnp.zeros((Nnew, num_regions, Mf))
        for k in range(Mf):
            for j in range(num_regions):
                eta = gpr_latent[j][k]
                gpr_lengthscale_array = jnp.array(
                    [gpr_lengthscale_x[j, k], gpr_lengthscale_y[j, k]]
                )
                f = compute_f_jax(
                    gpr_var[j, k],
                    gpr_lengthscale_array,
                    gpr_bias[j, k],
                    eta,
                    xf,
                    jitter=self.jitter,
                )
                mean, _, var = gpr_forward_jax(
                    gpr_var[j, k],
                    gpr_lengthscale_array,
                    xf,
                    f,
                    Xnew,
                    gpr_noise,
                    include_noise=False,
                    jitter=self.jitter,
                )
                Fr_new = Fr_new.at[:, j, k].set(mean)
                Vr_new = Vr_new.at[:, j, k].set(var)

        f_piecewise = jnp.zeros((Nnew, Mf, 1))
        v_piecewise = jnp.zeros((Nnew, Mf, 1))
        f_sample = jnp.zeros((Nnew, Mf, 1))
        for k in range(Mf):
            for j in range(num_regions):
                f_piecewise = f_piecewise.at[:, k, 0].set(
                    f_piecewise[:, k, 0] + gpc_new_probs[:, j] * Fr_new[:, j, k]
                )
                v_piecewise = v_piecewise.at[:, k, 0].set(
                    v_piecewise[:, k, 0] + gpc_new_probs[:, j] * Vr_new[:, j, k]
                )
            f_sample = f_sample.at[:, k, 0].set(
                ndist.Normal(f_piecewise[:, k, 0], jnp.sqrt(gpr_noise)).sample(subkey)
            )

        gpc_new_probs_ = numpyro.sample("gpc_new_probs", ndist.Delta(gpc_new_probs))
        f_piecewise_ = numpyro.sample("f_piecewise", ndist.Delta(f_piecewise))
        f_sample_ = numpyro.sample("f_sample", ndist.Delta(f_sample))
        Fr_new_ = numpyro.sample("Fr_new", ndist.Delta(Fr_new))
        v_piecewise_ = numpyro.sample("v_piecewise", ndist.Delta(v_piecewise))

        return gpc_new_probs_, f_piecewise_, f_sample_, Fr_new_, v_piecewise_


@typesafedataclass(config=_Config)
class SAGENDCoreg(Joint):
    """Class for ND SAGE co-regionalization: joint segmentation and regression algorithms.

    This class allows for multiple sets of structure labels,
    And multiple sets of functional property measurements.

    Attributes
    ----------
    num_phase_regions : int
        number of phase regions
    num_samples : int
        number of samples
    num_warmup : int
        number warmup
    num_chains : int
        number of chains
    target_accept_prob : float
        target accept probability
    max_tree_depth : int
        max tree depth
    phase_map_SVI_num_steps : int
        phase map SVI number of steps
    jitter : float
        jitter
    Adam_step_size : float
        Adam step size
    posterior_sampling : int
        posterior sampling
    locations_structure : list
        locations structure
    locations_functional_property : list
        locations functional properties
    locations_prediction : np.ndarray
        locations predictions
    target_structure_labels : list
        target structure labels
    target_functional_properties : list
        target functional properties
    gpc_variance_bounds : np.ndarray
        gpc variance bounds
    gpc_lengthscale_bounds : np.ndarray
        gpc lengthscale bounds
    gpr_bias_bounds : np.ndarray
        gpr bias bounds
    gpr_variance_bounds : np.ndarray
        gpr variance bounds
    gpr_lengthscale_bounds : np.ndarray
        gpr lengthscale bounds
    gpr_noise_bounds : np.ndarray
        gpr noise bounds
    key : np.ndarray
        pseudo-random number generator (PRNG) key not initialized by user

    Methods
    -------
    run()
        Run model.
    model_SAGE_ND_Coreg()
        Model for SAGE ND Coreg.
    model_SAGE_ND_Coreg_PM()
        Model for SAGE ND Coreg PM.
    predict_SAGE_ND_Coreg()
        Predict the phase region labels and functional properties for new data.
    """

    # needed inputs
    num_phase_regions: int
    num_samples: int
    num_warmup: int
    num_chains: int
    target_accept_prob: float
    max_tree_depth: int
    phase_map_SVI_num_steps: int
    jitter: float
    Adam_step_size: float
    posterior_sampling: int

    locations_structure: list
    locations_functional_property: list
    target_structure_labels: list
    target_functional_properties: list
    locations_prediction: np.ndarray

    gpc_variance_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpc_lengthscale_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_bias_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_variance_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_lengthscale_bounds: np.ndarray = field(default_factory=_default_ndarray)
    gpr_noise_bounds: np.ndarray = field(default_factory=_default_ndarray)

    key = jax.random.PRNGKey(0)

    # Outputs
    predictions: dict[str, Any] = {}

    def run(self) -> None:
        """Run the model and update the internal state of the object.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value, it performs the predictions and
            saves the results in `self.predictions`.

        """
        Xs_ = self.locations_structure
        ys_ = self.target_structure_labels
        Xf_ = self.locations_functional_property
        yf_ = self.target_functional_properties
        gpc_var_bounds = jnp.asarray(self.gpc_variance_bounds).copy()
        gpc_ls_bounds = jnp.asarray(self.gpc_lengthscale_bounds).copy()
        gpr_var_bounds = jnp.asarray(self.gpr_variance_bounds).copy()
        gpr_ls_bounds = jnp.asarray(self.gpr_lengthscale_bounds).copy()
        gpr_noise_bounds = jnp.asarray(self.gpr_noise_bounds).copy()
        gpr_bias_bounds = jnp.asarray(self.gpr_noise_bounds).copy()
        Xpred = jnp.asarray(self.locations_prediction).copy()
        num_regions = self.num_phase_regions
        num_proc = self.num_chains

        def predict_sage(post_samples, model, *args, **kwargs):
            model = handlers.seed(handlers.condition(model, post_samples), self.key)
            model_trace = handlers.trace(model).get_trace(*args, **kwargs)
            return (
                model_trace["Fr_new"]["value"],
                model_trace["f_piecewise"]["value"],
                model_trace["f_sample"]["value"],
                model_trace["gpc_new_probs"]["value"],
                model_trace["v_piecewise"]["value"],
            )

        predict_fn_sage = jax.pmap(
            lambda samples: predict_sage(
                samples,
                self.predict_SAGE_ND_Coreg,
                Xpred,
                Xs_,
                ys_,
                Xf_,
                yf_,
                num_regions,
                gpc_var_bounds,
                gpc_ls_bounds,
                gpr_var_bounds,
                gpr_ls_bounds,
                gpr_bias_bounds,
                gpr_noise_bounds,
            ),
            axis_name=0,
        )

        tic = time.perf_counter()

        data = [Xs_, ys_, Xf_, num_regions, gpc_var_bounds, gpc_ls_bounds]

        autoguide_mle = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(
            self.model_SAGE_ND_Coreg_PM
        )
        optimizer = numpyro.optim.Adam(step_size=self.Adam_step_size)

        svi = nSVI(
            self.model_SAGE_ND_Coreg_PM, autoguide_mle, optimizer, loss=nTrace_ELBO()
        )
        svi_result = svi.run(self.key, self.phase_map_SVI_num_steps, *data)

        params = svi_result.params
        mle_st = autoguide_mle.median(params)

        # gpc_new_probs_st = predict_fn_st(mle_2a_st)
        init_params = {
            "gpc_latent_0": mle_st["gpc_latent_0"],
            "gpc_latent_1": mle_st["gpc_latent_1"],
            "gpc_var": mle_st["gpc_var"],
            "gpc_lengthscale": mle_st["gpc_lengthscale"],
            "gpc_bias": mle_st["gpc_bias"],
        }
        init_strategy = init_to_value(values=init_params)

        nuts = nMCMC(
            nNUTS(
                self.model_SAGE_ND_Coreg,
                target_accept_prob=self.target_accept_prob,
                max_tree_depth=self.max_tree_depth,
                init_strategy=init_strategy,
            ),
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            num_chains=self.num_chains,
        )
        nuts.run(
            self.key,
            Xs_,
            ys_,
            Xf_,
            yf_,
            num_regions,
            gpc_var_bounds=gpc_var_bounds,
            gpc_ls_bounds=gpc_ls_bounds,
            gpr_var_bounds=gpr_var_bounds,
            gpr_ls_bounds=gpr_ls_bounds,
            gpr_bias_bounds=gpr_bias_bounds,
            gpr_noise_bounds=gpr_noise_bounds,
        )

        nuts_posterior_samples = nuts.get_samples()
        samples = subsample(nuts_posterior_samples, step=self.posterior_sampling)

        num_length = samples["gpr_noise"].shape[0]
        sl = split_samples(samples, num_proc, num_length)
        splits = np.array(num_length / num_proc).astype(int)

        labels = ["Fr_new", "f_piecewise", "f_sample", "gpc_new_probs", "v_piecewise"]

        for i in trange(splits):
            if i == 0:
                preds = predict_fn_sage(sl[i])
                preds_stacked = {
                    labels[0]: preds[0],
                    labels[1]: preds[1],
                    labels[2]: preds[2],
                    labels[3]: preds[3],
                    labels[4]: preds[4],
                }
            else:
                preds = predict_fn_sage(sl[i])
                for j in range(len(labels)):
                    preds_stacked[labels[j]] = np.vstack(
                        (preds_stacked[labels[j]], preds[j])
                    )
        toc = time.perf_counter()
        print(f"Run in {toc - tic:0.4f} seconds")
        output = {"preds": preds_stacked}

        # ------------------

        preds_sage = output["preds"]
        phase_region_labels_mean = np.nanmean(preds_sage["gpc_new_probs"], axis=0)
        phase_region_labels_std = np.nanstd(preds_sage["gpc_new_probs"], axis=0)
        phase_region_labels_mean_estimate = np.argmax(phase_region_labels_mean, axis=1)
        phase_region_labels_mean_entropy = entropy(phase_region_labels_mean, axis=1)
        functional_properties_mean = np.nanmean(preds_sage["f_piecewise"], axis=0)
        functional_properties_std = np.sqrt(
            np.nanmean(preds_sage["v_piecewise"], axis=0)
        )

        # gpc_new_mean, gpc_new_std, gpr_new_mean, gpr_new_var, cp_samples, gpr_mean_noiseless_mllk, gpr_var_noiseless_mllk
        self.predictions = {
            "phase_region_labels_mean_estimate": phase_region_labels_mean_estimate,
            "phase_region_labels_mean": phase_region_labels_mean,
            "phase_region_labels_mean_entropy": phase_region_labels_mean_entropy,
            "phase_region_labels_std": phase_region_labels_std,
            "functional_property_mean": functional_properties_mean,
            "functional_property_std": functional_properties_std,
        }

    def model_SAGE_ND_Coreg(
        self,
        xs_: torch.Tensor,
        ys_: torch.Tensor,
        xf_: torch.Tensor,
        yf_: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
        gpr_var_bounds: Union[tuple, list],
        gpr_ls_bounds: Union[tuple, list],
        gpr_bias_bounds: Union[tuple, list],
        gpr_noise_bounds: Union[tuple, list],
    ) -> None:
        """Model for SAGE ND Coreg.

        Parameters
        ----------
        xs_ : torch.Tensor
            Input tensor for training samples.
        ys_ : torch.Tensor
            Target tensor for training samples.
        xf_ : torch.Tensor
            Input tensor for test samples.
        yf_ : torch.Tensor
            Target tensor for test samples.
        num_regions : int
            Number of regions.
        gpc_var_bounds : tuple, list
            Bounds for Gaussian Process Classification (GPC) variance.
        gpc_ls_bounds : tuple, list
            Bounds for GPC lengthscale.
        gpr_var_bounds : tuple, list
            Bounds for Gaussian Process Regression (GPR) variance.
        gpr_ls_bounds : tuple, list
            Bounds for GPR lengthscale.
        gpr_bias_bounds : tuple, list
            Bounds for GPR bias.
        gpr_noise_bounds : tuple, list
            Bounds for GPR noise.

        Returns
        -------
        None
            This function does not return any value. It performs the prediction and updates the internal state of the object.
        """

        # assume all inputs are lists
        # assumes all function property measurements measured at same locations.
        # TODO unused jitter
        jitter = self.jitter

        Ns = np.array(
            [xs_[i].shape[0] for i in range(len(xs_))], dtype=np.int64
        )  # total number of structure observations
        Nf = np.array(
            [xf_[i].shape[0] for i in range(len(xf_))], dtype=np.int64
        )  # total number of functional property observations

        Ns_indices = np.concatenate((np.zeros((1), dtype=np.int64), Ns.cumsum()))
        Nf_indices = np.concatenate((np.zeros((1), dtype=np.int64), Nf.cumsum()))

        Mf = len(xf_)  # number of functional property data sets
        Ms = len(xs_)  # number of structure data sets.

        xs = jnp.vstack(xs_)
        xf = jnp.vstack(xf_)
        x_ = jnp.vstack([xs, xf])

        Nsf = x_.shape[0]  # number of all data points across all sets.

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Priors: GPR
        gpr_var_bound_min = gpr_var_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_var_bound_max = gpr_var_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_min = gpr_ls_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_max = gpr_ls_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_min = gpr_bias_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_max = gpr_bias_bounds[1] * jnp.ones((num_regions, Mf))

        gpr_noise = numpyro.sample(
            "gpr_noise", ndist.Uniform(gpr_noise_bounds[0], gpr_noise_bounds[1])
        )
        gpr_var = numpyro.sample(
            "gpr_var", ndist.Uniform(gpr_var_bound_min, gpr_var_bound_max)
        )
        gpr_lengthscale_x = numpyro.sample(
            "gpr_lengthscale_x",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_lengthscale_y = numpyro.sample(
            "gpr_lengthscale_y",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_bias = numpyro.sample(
            "gpr_bias", ndist.Uniform(gpr_bias_bound_min, gpr_bias_bound_max)
        )

        # Get latent functions, one for each region (i.e., segment).
        Fc = jnp.zeros((Nsf, num_regions))
        for i in range(num_regions):
            with numpyro.plate("gpc_latent_response" + str(i), Nsf):
                gpc_latent = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))
            # print('gpc_latent', gpc_latent.shape, 'x_', x_.shape)
            f = compute_f_matern52_jax(
                gpc_var, gpc_lengthscale, gpc_bias, gpc_latent, x_, jitter=self.jitter
            )
            Fc = Fc.at[:, i].set(f)  # x = x.at[idx].set(y)

        probs = logits_to_probs_jax(Fc)

        # predicted the region label for each functional property data point.
        Ns_sum = Ns.sum()
        probs_fp_ = []  # probs[Ns_sum:,:].double()
        probs_st_ = []

        for i in range(Ms):
            probs_st_.append(dynamic_slice(probs, (Ns_indices[i], 0), (Ns[i], 2)))

        for i in range(Mf):
            probs_fp_.append(
                dynamic_slice(probs, (Ns_sum + Nf_indices[i], 0), (Nf[i], 2))
            )

        # gpr for each region.
        Fr_ = []
        for j in range(Mf):
            fr = jnp.zeros((Nf[j], num_regions))
            for i in range(num_regions):
                with numpyro.plate("gpr_latent_response" + str(i), Nf[j]):
                    gpr_latent = numpyro.sample(
                        "gpr_latent_" + str(i) + "_Mf_" + str(j), ndist.Normal(0, 1)
                    )

                gpr_lengthscale_array = jnp.array(
                    [gpr_lengthscale_x[i, j], gpr_lengthscale_y[i, j]]
                )
                f = compute_f_jax(
                    gpr_var[i, j],
                    gpr_lengthscale_array,
                    gpr_bias[i, j],
                    gpr_latent,
                    xf_[j],
                    jitter=self.jitter,
                )
                fr = fr.at[:, i].set(f)
            Fr_.append(fr)

        f_piecewise_ = []
        for j in range(Mf):
            fpw = jnp.zeros((Nf[j]))
            for i in range(num_regions):
                fpw = fpw.at[:].set(fpw + probs_fp_[j][:, i] * Fr_[j][:, i])
            f_piecewise_.append(fpw)

        llk = ndist.Categorical(probs=probs_st_[0]).log_prob(ys_[0].flatten()).sum()
        for i in range(1, Ms):
            llk += (
                ndist.Categorical(probs=probs_st_[i]).log_prob(ys_[i].flatten()).sum()
            )

        for j in range(Mf):
            llk = (
                llk
                + ndist.Normal(f_piecewise_[j], jnp.sqrt(gpr_noise))
                .log_prob(yf_[j])
                .sum()
            )

        numpyro.deterministic("llk", llk)
        numpyro.factor("obs", llk)

    def model_SAGE_ND_Coreg_PM(
        self,
        xs_: torch.Tensor,
        ys_: torch.Tensor,
        xf_: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
    ) -> None:
        """Model for SAGE ND Coreg PM.

        Parameters
        ----------
        xs_ : torch.Tensor
            Input tensor for training samples.
        ys_ : torch.Tensor
            Target tensor for training samples.
        xf_ : torch.Tensor
            Input tensor for test samples.
        num_regions : int
            Number of regions.
        gpc_var_bounds : tuple, list
            Bounds for Gaussian Process Classification (GPC) variance.
        gpc_ls_bounds : tuple, list
            Bounds for GPC lengthscale.

        Returns
        -------
        None
            This function does not return any value. It performs the prediction and updates the internal state of the object.

        """

        # assume all inputs are lists
        # assumes all function property measurements measured at same locations.
        # TODO unused jitter
        jitter = self.jitter

        Ns = np.array([xs_[i].shape[0] for i in range(len(xs_))], dtype=np.int64)
        Nf = np.array([xf_[i].shape[0] for i in range(len(xf_))], dtype=np.int64)

        Ns_indices = np.concatenate((np.zeros((1), dtype=np.int64), Ns.cumsum()))
        Nf_indices = np.concatenate((np.zeros((1), dtype=np.int64), Nf.cumsum()))

        Mf = len(xf_)  # number of functional property data sets
        Ms = len(xs_)  # number of structure data sets.

        xs = jnp.vstack(xs_)
        xf = jnp.vstack(xf_)
        x_ = jnp.vstack([xs, xf])

        Nsf = x_.shape[0]  # number of all data points across all sets.

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Get latent functions, one for each region (i.e., segment).
        Fc = jnp.zeros((x_.shape[0], num_regions))
        for i in range(num_regions):
            with numpyro.plate("gpc_latent_response" + str(i), Nsf):
                gpc_latent = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))

            f = compute_f_matern52_jax(
                gpc_var, gpc_lengthscale, gpc_bias, gpc_latent, x_, jitter=self.jitter
            )
            Fc = Fc.at[:, i].set(f)  # x = x.at[idx].set(y)

        probs = logits_to_probs_jax(Fc)

        # predicted the region label for each functional property data point.
        Ns_sum = Ns.sum()
        probs_st_ = []

        for i in range(Ms):
            probs_st_.append(dynamic_slice(probs, (Ns_indices[i], 0), (Ns[i], 2)))

        llk = ndist.Categorical(probs=probs_st_[0]).log_prob(ys_[0].flatten()).sum()
        for i in range(1, Ms):
            llk += (
                ndist.Categorical(probs=probs_st_[i]).log_prob(ys_[i].flatten()).sum()
            )

        numpyro.deterministic("llk", llk)
        numpyro.factor("obs", llk)

    # TODO unused vars
    def predict_SAGE_ND_Coreg(
        self,
        Xnew: torch.Tensor,
        xs_: torch.Tensor,
        ys_: torch.Tensor,
        xf_: torch.Tensor,
        yf_: torch.Tensor,
        num_regions: int,
        gpc_var_bounds: Union[tuple, list],
        gpc_ls_bounds: Union[tuple, list],
        gpr_var_bounds: Union[tuple, list],
        gpr_ls_bounds: Union[tuple, list],
        gpr_bias_bounds: Union[tuple, list],
        gpr_noise_bounds: Union[tuple, list],
    ) -> tuple:
        """Predict for SAGE ND Coreg.

        Parameters
        ----------
        Xnew : torch.Tensor
            Input tensor for test samples.
        xs_ : torch.Tensor
            Input tensor for training samples.
        ys_ : torch.Tensor
            Target tensor for training samples.
        xf_ : torch.Tensor
            Input tensor for test samples.
        yf_ : torch.Tensor
            Target tensor for test samples.
        num_regions : int
            Number of regions.
        gpc_var_bounds : tuple, list
            Bounds for Gaussian Process Classification (GPC) variance.
        gpc_ls_bounds : tuple, list
            Bounds for GPC lengthscale.
        gpr_var_bounds : tuple, list
            Bounds for Gaussian Process Regression (GPR) variance.
        gpr_ls_bounds : tuple, list
            Bounds for GPR lengthscale.
        gpr_bias_bounds : tuple, list
            Bounds for GPR bias.
        gpr_noise_bounds : tuple, list
            Bounds for GPR noise.

        Returns
        -------
        tuple
            Tuple of predictions.

        """

        # assume all inputs are lists
        # assumes all function property measurements measured at same locations.

        key_in = self.key
        _, subkey = jax.random.split(key_in)

        jitter = self.jitter

        Ns = np.array([xs_[i].shape[0] for i in range(len(xs_))], dtype=np.int64)
        Nf = np.array([xf_[i].shape[0] for i in range(len(xf_))], dtype=np.int64)

        Ns_indices = np.concatenate((np.zeros((1), dtype=np.int64), Ns.cumsum()))
        Nf_indices = np.concatenate((np.zeros((1), dtype=np.int64), Nf.cumsum()))

        Mf = len(xf_)  # number of functional property data sets
        Ms = len(xs_)  # number of structure data sets.
        Nnew = Xnew.shape[0]

        xs = jnp.vstack(xs_)
        xf = jnp.vstack(xf_)
        x_ = jnp.vstack([xs, xf])

        Nsf = x_.shape[0]  # number of all data points across all sets.

        # Priors: Segmentation.
        gpc_var = numpyro.sample(
            "gpc_var", ndist.Uniform(gpc_var_bounds[0], gpc_var_bounds[1])
        )  # variance
        gpc_lengthscale = numpyro.sample(
            "gpc_lengthscale", ndist.Uniform(gpc_ls_bounds[0], gpc_ls_bounds[1])
        )  # ls
        gpc_bias = numpyro.sample("gpc_bias", ndist.Normal(0, 1))  # bias

        # Priors: GPR
        gpr_var_bound_min = gpr_var_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_var_bound_max = gpr_var_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_min = gpr_ls_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_lengthscale_bound_max = gpr_ls_bounds[1] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_min = gpr_bias_bounds[0] * jnp.ones((num_regions, Mf))
        gpr_bias_bound_max = gpr_bias_bounds[1] * jnp.ones((num_regions, Mf))

        gpr_noise = numpyro.sample(
            "gpr_noise", ndist.Uniform(gpr_noise_bounds[0], gpr_noise_bounds[1])
        )
        gpr_var = numpyro.sample(
            "gpr_var", ndist.Uniform(gpr_var_bound_min, gpr_var_bound_max)
        )
        gpr_lengthscale_x = numpyro.sample(
            "gpr_lengthscale_x",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_lengthscale_y = numpyro.sample(
            "gpr_lengthscale_y",
            ndist.Uniform(gpr_lengthscale_bound_min, gpr_lengthscale_bound_max),
        )
        gpr_bias = numpyro.sample(
            "gpr_bias", ndist.Uniform(gpr_bias_bound_min, gpr_bias_bound_max)
        )

        # --- added ----------------------------
        gpc_train_latent = jnp.zeros((x_.shape[0], num_regions))
        gpc_new_latent = jnp.zeros((Nnew, num_regions))
        gpc_new_probs = jnp.zeros((Nnew, num_regions))

        gpc_latent = [0] * num_regions
        for i in range(num_regions):
            gpc_latent[i] = numpyro.sample("gpc_latent_" + str(i), ndist.Normal(0, 1))

        # get region labels
        for j in range(num_regions):
            f = compute_f_matern52_jax(
                gpc_var,
                gpc_lengthscale,
                gpc_bias,
                gpc_latent[j],
                x_,
                jitter=self.jitter,
            )
            gpc_train_latent.at[:, j].set(f)
            gpc_noise = self.jitter * 10.0
            mean, cov, _ = gpr_forward_matern52_jax(
                gpc_var,
                gpc_lengthscale,
                x_,
                f,
                Xnew,
                gpc_noise,
                include_noise=False,
                jitter=self.jitter,
            )

            fhat = ndist.MultivariateNormal(
                mean, cov + jnp.eye(Nnew) * self.jitter
            ).sample(subkey)
            gpc_new_latent = gpc_new_latent.at[:, j].set(fhat)

        gpc_new_probs = logits_to_probs_jax(gpc_new_latent)

        # get gpr
        Fr_new = jnp.zeros((Nnew, num_regions, Mf))
        Vr_new = jnp.zeros((Nnew, num_regions, Mf))

        gpr_latent = [[0] * Mf for i in range(num_regions)]
        for j in range(Mf):
            for i in range(num_regions):
                gpr_latent[i][j] = numpyro.sample(
                    "gpr_latent_" + str(i) + "_Mf_" + str(j), ndist.Normal(0, 1)
                )

        for k in range(Mf):
            for j in range(num_regions):
                gpr_lengthscale_array = jnp.array(
                    [gpr_lengthscale_x[j, k], gpr_lengthscale_y[j, k]]
                )
                f = compute_f_jax(
                    gpr_var[j, k],
                    gpr_lengthscale_array,
                    gpr_bias[j, k],
                    gpr_latent[j][k],
                    xf_[k],
                    jitter=self.jitter,
                )
                mean, _, var = gpr_forward_jax(
                    gpr_var[j, k],
                    gpr_lengthscale_array,
                    xf_[k],
                    f,
                    Xnew,
                    gpr_noise,
                    include_noise=False,
                    jitter=self.jitter,
                )
                Fr_new = Fr_new.at[:, j, k].set(mean)
                Vr_new = Vr_new.at[:, j, k].set(var)

        f_piecewise = jnp.zeros(
            (Nnew, Mf, 1)
        )  # last dimension added for stacking purposes in plotting func.
        v_piecewise = jnp.zeros((Nnew, Mf, 1))
        f_sample = jnp.zeros((Nnew, Mf, 1))
        for k in range(Mf):
            for j in range(num_regions):
                f_piecewise = f_piecewise.at[:, k, 0].set(
                    f_piecewise[:, k, 0] + gpc_new_probs[:, j] * Fr_new[:, j, k]
                )
                v_piecewise = v_piecewise.at[:, k, 0].set(
                    v_piecewise[:, k, 0] + gpc_new_probs[:, j] * Vr_new[:, j, k]
                )
            f_sample = f_sample.at[:, k, 0].set(
                ndist.Normal(f_piecewise[:, k, 0], jnp.sqrt(gpr_noise)).sample(subkey)
            )

        gpc_new_probs_ = numpyro.sample("gpc_new_probs", ndist.Delta(gpc_new_probs))
        f_piecewise_ = numpyro.sample("f_piecewise", ndist.Delta(f_piecewise))
        f_sample_ = numpyro.sample("f_sample", ndist.Delta(f_sample))
        Fr_new_ = numpyro.sample("Fr_new", ndist.Delta(Fr_new))
        v_piecewise_ = numpyro.sample("v_piecewise", ndist.Delta(v_piecewise))

        return gpc_new_probs_, f_piecewise_, f_sample_, Fr_new_, v_piecewise_


# TODO typehint
def compute_f_torch(variance, lengthscales, bias, eta, X, jitter) -> torch.Tensor:
    """Compute f with torch."""
    N = X.shape[0]
    K = RBF_torch(variance, lengthscales, X) + torch.eye(N) * jitter
    L = torch.linalg.cholesky(K)
    return torch.matmul(L, eta) + bias


# TODO typehint
def gpr_forward_torch(
    variance,
    lengthscales,
    xtrain,
    ytrain,
    xnew,
    noise_var,
    include_noise=True,
    prob_weights=None,
    jitter=1e-6,
) -> tuple:
    """Gaussian Process Regression Forward.

    Parameters
    ----------
    variance : float
        Variance.
    lengthscales : np.ndarray
        Lengthscales.
    xtrain : np.ndarray
        Training input tensor.
    ytrain : np.ndarray
        Training target tensor.
    xnew : np.ndarray
        New input tensor.
    noise_var : float
        Noise variance.
    include_noise : bool, optional
        Include noise, by default True.
    prob_weights : np.ndarray, optional
        Probability weights, by default None.
    jitter : float, optional
        Jitter, by default 1e-6.

    Returns
    -------
    tuple
        Mean, covariance, variance.

    """
    # n is new, t is train
    ytrain = ytrain.flatten()

    K_nt = RBF_torch(variance, lengthscales, xnew, xtrain)
    K_tt = RBF_torch(variance, lengthscales, xtrain, xtrain)
    K_nn = RBF_torch(variance, lengthscales, xnew, xnew)

    I_noise = torch.eye(K_tt.shape[0]) * noise_var
    L = torch.linalg.inv(K_tt + I_noise)

    if prob_weights is None:
        mean = torch.matmul(K_nt, torch.matmul(L, ytrain[:, None]))
    else:
        fit_mean = torch.sum(ytrain * prob_weights.flatten()) / torch.sum(
            prob_weights.flatten()
        )
        ytrain = ytrain - fit_mean
        mean = torch.matmul(K_nt, torch.matmul(L, ytrain[:, None])) + fit_mean

    cov = K_nn - torch.matmul(K_nt, torch.matmul(L, K_nt.T))
    cov = cov + torch.eye(cov.shape[0]) * jitter
    if include_noise:
        cov = cov + torch.eye(cov.shape[0]) * noise_var
    var = torch.diagonal(cov)
    return mean.flatten(), cov, var.flatten()


# TODO typehints
def RBF_torch(
    variance: float,
    lengthscales: np.ndarray,
    X: Union[np.ndarray, torch.Tensor],
    Z: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> torch.Tensor:
    """Radial Basis Function Kernel.

    Parameters
    ----------
    variance : float
        Variance of the kernel.
    lengthscales : np.ndarray
        Lengthscales of the kernel.
    X : np.ndarray, torch.Tensor
        Input tensor.
    Z : np.ndarray, torch.Tensor, optional
        Input tensor.

    Returns
    -------
    torch.Tensor
        Radial Basis Function Kernel.

    """
    # built from: https://github.com/pyro-ppl/pyro/blob/727aff741e105715840bfdafee5bfeda7e8b65e8/pyro/contrib/gp/kernels/isotropic.py#L41
    if Z is None:
        Z = X
    #     if jnp.isscalar(lengthscales):
    #         lengthscales = lengthscales*jnp.ones((2))
    scaled_X = X / lengthscales
    scaled_Z = Z / lengthscales
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = torch.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return variance * torch.exp(-0.5 * r2)


def change_points_to_labels_torch(
    cp: torch.Tensor, X: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """Change Points to Labels.

    Parameters
    ----------
    cp : torch.Tensor
        Change points.
    X : np.ndarray, torch.Tensor
        Input tensor.

    Returns
    -------
    cl: torch.Tensor
        Labels.
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    cp, _ = torch.sort(cp)
    cl = torch.zeros((X.shape[0])).long()
    N = cp.shape[0]  # N = 3
    for i in range(0, N):
        if i < N - 1:
            idx = torch.logical_and(X > cp[i], X < cp[i + 1])
        elif i == N - 1:
            idx = X > cp[i]
        cl[idx.flatten()] = i + 1
    return cl


def to_torch(v: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert to torch tensor.

    Parameters
    ----------
    v : np.ndarray, torch.Tensor
        Input tensor.

    Returns
    -------
    v: torch.Tensor
        Torch tensor.
    """
    if not isinstance(v, (torch.Tensor, np.ndarray)):
        raise ValueError("Input must be a numpy array or torch tensor.")
    return torch.Tensor(v)


def subsample(samples: dict, step: int) -> dict:
    """Subsample dictionary.

    Parameters
    ----------
    samples : dict
        Samples.
    step : int
        Step size.

    Returns
    -------
    tamples: dict
        Subsampled samples.
    """
    tamples = {}
    for k in samples.keys():
        tamples[k] = samples[k][::step]
    return tamples


def split_samples(samples: dict, num_proc: int, length: int) -> list:
    """Split samples.

    Parameters
    ----------
    samples : dict
        Samples.
    num_proc : int
        Number of processes.
    length : int
        Length.

    Returns
    -------
    sample_list: list
        List of split samples.
    """
    sample_list = []
    splits = np.array(length / num_proc).astype(int)
    s = {}
    for i in trange(splits):
        for k in samples.keys():
            s[k] = samples[k][(i * num_proc) : ((i + 1) * num_proc)]
        sample_list.append(s)
    return sample_list


# TODO unused vars
def get_samples_split(samples: dict, num_proc: int, length: int, i: int) -> dict:
    """Get samples split.

    Parameters
    ----------
    samples : dict
        Samples.
    num_proc : int
        Number of processes.
    length : int
        Length.
    i : int
        Index.

    Returns
    -------
    s: dict
        Samples.

    """
    s = {}
    for k in samples.keys():
        s[k] = samples[k][(i * num_proc) : ((i + 1) * num_proc)]
    return s


def logits_to_probs_jax(logits: np.ndarray) -> jax.Array:
    """Logits to probabilities.

    Assumes obs x num_of_categories.

    Parameters
    ----------
    logits : np.ndarray
        Logits.

    Returns
    -------
    probs: jax.Array
        Probabilities.

    """
    # assumes obs x num_of_categories
    logits = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    probs = jax.nn.softmax(logits, axis=-1)
    return probs


def remap_array(v: np.ndarray) -> torch.Tensor:
    """Remap array.

    Parameters
    ----------
    v : np.ndarray
        Input array.

    Returns
    -------
    vnew: torch.Tensor
        Remapped array.
    """
    vnew = torch.zeros(v.shape)
    uv = torch.unique(v)
    for i in range(uv.shape[0]):
        vnew[v == uv[i]] = i
    return vnew


def flip_keys_and_indices(samples: dict, step: int = 1):
    """Flip keys and indices.

    Parameters
    ----------
    samples : dict
        Samples.
    step : int, optional
        Step size, default to 1

    Returns
    -------
    s: list
        Flipped samples.

    """
    s = []
    K = list(samples.keys())
    Nf = samples["gpr_noise"].shape[0]

    for n in tqdm(np.arange(0, Nf, step)):
        temp = {}
        for k in K:
            temp[k] = samples[k][n]
        temp["seed"] = n
        s.append(temp)
    return s


def gpr_forward_jax(
    variance: float,
    lengthscales: np.ndarray,
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xnew: np.ndarray,
    noise_var: float,
    include_noise: bool = True,
    jitter: float = 1e-6,
) -> tuple:
    """Gaussian Process Regression Forward using Jax.

    Parameters
    ----------
    variance : float
        Variance.
    lengthscales : np.ndarray
        Lengthscales.
    xtrain : np.ndarray
        Training input tensor.
    ytrain : np.ndarray
        Training target tensor.
    xnew : np.ndarray
        New input tensor.
    noise_var : float
        Noise variance.
    include_noise : bool, optional
        Include noise, by default True.
    jitter : float, optional
        Jitter, by default 1e-6.

    Returns
    -------
    tuple
        Mean, covariance, variance.
    """
    # n is new, t is train
    K_nt = RBF_jax(variance, lengthscales, xnew, xtrain)
    K_tt = RBF_jax(variance, lengthscales, xtrain, xtrain)
    K_nn = RBF_jax(variance, lengthscales, xnew, xnew)
    I_noise = jnp.eye(K_tt.shape[0]) * (noise_var + jitter)
    L = jnp.linalg.inv(K_tt + I_noise)
    mean = jnp.matmul(K_nt, jnp.matmul(L, ytrain.flatten()[:, None]))
    cov = K_nn - jnp.matmul(K_nt, jnp.matmul(L, K_nt.T))
    if include_noise:
        cov = cov + jnp.eye(cov.shape[0]) * noise_var
    var = jnp.diagonal(cov)
    return mean.flatten(), cov, var.flatten()


def gpr_forward_matern52_jax(
    variance: float,
    lengthscale: np.ndarray,
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xnew: np.ndarray,
    noise_var: float,
    include_noise: bool = True,
    jitter: float = 1e-6,
) -> tuple:
    """Gaussian Process Regression Forward Matern52.

    Parameters
    ----------
    variance : float
        Variance.
    lengthscales : np.ndarray
        Lengthscales.
    xtrain : np.ndarray
        Training input tensor.
    ytrain : np.ndarray
        Training target tensor.
    xnew : np.ndarray
        New input tensor.
    noise_var : float
        Noise variance.
    include_noise : bool, optional
        Include noise, by default True.
    jitter : float, optional
        Jitter, by default 1e-6.

    Returns
    -------
    tuple
        Mean, covariance, variance.
    """
    # n is new, t is train
    K_nt = Matern52_2D_jax(variance, lengthscale, xnew, xtrain)
    K_tt = Matern52_2D_jax(variance, lengthscale, xtrain, xtrain)
    K_nn = Matern52_2D_jax(variance, lengthscale, xnew, xnew)
    I_noise = jnp.eye(K_tt.shape[0]) * (noise_var + jitter)
    L = jnp.linalg.inv(K_tt + I_noise)
    mean = jnp.matmul(K_nt, jnp.matmul(L, ytrain.flatten()[:, None]))
    cov = K_nn - jnp.matmul(K_nt, jnp.matmul(L, K_nt.T))
    if include_noise:
        cov = cov + jnp.eye(cov.shape[0]) * noise_var
    var = jnp.diagonal(cov)
    return mean.flatten(), cov, var.flatten()


# TODO typehint
def RBF_jax(
    variance: float,
    lengthscales: Union[np.ndarray, torch.Tensor],
    X: Union[torch.Tensor, np.ndarray],
    Z: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> np.ndarray:
    """Radial Basis Function Kernel Jax.

    Parameters
    ----------
    variance : float
        Variance of the kernel.
    lengthscales : np.ndarray
        Lengthscales of the kernel.
    X : np.ndarray, torch.Tensor
        Input tensor.
    Z : np.ndarray, torch.Tensor, optional
        Input tensor.

    Returns
    -------
    np.ndarray
        Radial Basis Function Kernel.
    """
    if Z is None:
        Z = X.copy()
    #     if jnp.isscalar(lengthscales):
    #         lengthscales = lengthscales*jnp.ones((2))
    scaled_X = X / lengthscales
    scaled_Z = Z / lengthscales
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - XZ + Z2.T
    return variance * jnp.exp(-0.5 * r2)


def Matern52_2D_jax(
    variance: float,
    lengthscale: float,
    X: Union[torch.Tensor, np.ndarray],
    Z: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> np.ndarray:
    """Matern52 2D Kernel Jax.

    Parameters
    ----------
    variance : float
        Variance of the kernel.
    lengthscale : float
        Lengthscale of the kernel.
    X : np.ndarray, torch.Tensor
        Input tensor.
    Z : np.ndarray, torch.Tensor, optional
        Input tensor.

    Returns
    -------
    np.ndarray
        Matern52 2D Kernel.
    """
    if Z is None:
        Z = X.copy()

    kernel0 = gpx.kernels.Matern52(lengthscale=lengthscale, variance=variance)
    kernel1 = gpx.kernels.Matern52(lengthscale=lengthscale, variance=variance)
    prod_kernel = gpx.kernels.ProductKernel(kernels=[kernel0, kernel1])

    return prod_kernel.cross_covariance(X, Z)


# TODO typehint
def euclidean_jax(
    X1: Union[torch.Tensor, np.ndarray],
    X2: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> jax.Array:
    """Euclidean distance using Jax.

    Parameters
    ----------
    X1 : np.ndarray, torch.Tensor
        Input tensor.
    X2 : np.ndarray, torch.Tensor, optional
        Input tensor.

    Returns
    -------
    jax.Array
        Euclidean distance.
    """
    if X2 is None:
        X2 = X1.copy()
    c = X1[:, None] - X2[None, :]
    return jnp.sqrt(jnp.sum(c**2, axis=2))


# TODO typehints
def compute_f_jax(
    variance: float,
    lengthscales: np.ndarray,
    bias: np.ndarray,
    eta: np.ndarray,
    X: np.ndarray,
    jitter: Optional[float] = 1e-6,
) -> jax.Array:
    """Compute f using Jax.

    Parameters
    ----------
    variance : float
        Variance.
    lengthscales : np.ndarray
        Lengthscales.
    bias : np.ndarray
        Bias.
    eta : np.ndarray
        Eta.
    X : np.ndarray
        Input tensor.
    jitter : float, optional
        Jitter, by default 1e-6.

    Returns
    -------
    jax.Array
        matrix mult.

    """
    N = X.shape[0]
    K = RBF_jax(variance, lengthscales, X) + jnp.eye(N) * jitter
    L = jnp.linalg.cholesky(K)
    return jnp.matmul(L, eta) + bias


def compute_f_matern52_jax(
    variance: float,
    lengthscale: float,
    bias: np.ndarray,
    eta: np.ndarray,
    X: np.ndarray,
    jitter: Optional[float] = 1e-6,
) -> jax.Array:
    """Compute f matern52 using Jax.

    Parameters
    ----------
    variance : float
        Variance.
    lengthscale : float
        Lengthscale.
    bias : np.ndarray
        Bias.
    eta : np.ndarray
        Eta.
    X : np.ndarray
        Input tensor.
    jitter : float, optional
        Jitter, by default 1e-6.

    Returns
    -------
    jax.Array
        matrix mult.
    """
    N = X.shape[0]
    K = Matern52_2D_jax(variance, lengthscale, X) + jnp.eye(N) * jitter
    L = jnp.linalg.cholesky(K)
    return jnp.matmul(L, eta) + bias
