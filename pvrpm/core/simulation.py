import os

import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
from scipy.special import gamma, gammaln
from tqdm import tqdm

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase


def sample(
    distribution: str,
    parameters: dict,
    num_samples: int,
    method: str = "rou",
) -> np.array:
    """
    Sample data from a distribution. If distribution is a supported distribution, parameters should be a dictionary with keys "mean" and "std". Otherwise, distribution should be a scipy stats function and parameters be the kwargs for the distribution.

    Supported Distributions (only requires mean and std):
        - lognormal
        - normal
        - uniform (one std around mean)
        - weibull
        - exponential

    Supported sampling methods:
        - 'rou': Ratio of uniforms, the default. Pretty much random sampling

    Args:
        distribution (str): Name of the distribution function
        parameters (:obj:`dict`): Kwargs for the distribution (for a supported distribution should only be the mean and std)
        num_samples (int): Number of samples to return from distribution
        method (str, Optional): Sampling method to use, defaults to ratio of uniforms

    Returns:
        :obj:(list): List of floats containing samples from the distribution
    """
    distribution = distribution.lower().strip()
    method = method.lower().strip()

    if distribution == "lognormal":
        # lognormal uses the mean and std of the underlying normal distribution of log(X)
        # so they must be normalized first
        mu, sigma = parameters[ck.MEAN], parameters[ck.STD]
        normalized_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        normalized_mean = np.log(mu) - normalized_std ** 2 / 2
        dist = stats.lognorm(s=normalized_std, scale=np.exp(normalized_mean))
    elif distribution == "normal":
        dist = stats.norm(loc=parameters[ck.MEAN], scale=parameters[ck.STD])
    elif distribution == "uniform":
        a = parameters[ck.MEAN] - parameters[ck.STD]
        b = parameters[ck.STD] * 2
        dist = stats.uniform(loc=a, scale=b)
    elif distribution == "weibull":
        # for weibull, we have to solve for c and the scale parameter
        # this files for certain parameter ranges, raising a runtime error
        # see https://github.com/scipy/scipy/issues/12134 for reference
        def _h(c):
            r = np.exp(gammaln(2 / c) - 2 * gammaln(1 / c))
            return np.sqrt(1 / (2 * c * r - 1))

        mean, std = parameters[ck.MEAN], parameters[ck.STD]
        c0 = 1.27 * np.sqrt(mean / std)
        c, info, ier, msg = scipy.optimize.fsolve(
            lambda t: _h(t) - (mean / std),
            c0,
            xtol=1e-10,
            full_output=True,
        )

        # Test residual rather than error code.
        if np.abs(info["fvec"][0]) > 1e-8:
            raise RuntimeError(f"with mean={mean} and std={std}, solve failed: {msg}")

        c = c[0]
        scale = mean / gamma(1 + 1 / c)
        dist = stats.weibull_min(c=c, scale=scale)
    elif distribution == "exponential":
        dist = stats.expon(scale=parameters[ck.MEAN])
    else:
        # else, we don't know this distribution, pass the distribution directly to scipy
        dist = getattr(stats, distribution)
        if not dist:
            raise AttributeError(f"Scipy stats doesn't have a distribution '{distribution}'")
        dist = dist(**parameters)

    if method == "lhs":
        pass
    else:
        # scipy rvs uses rou sampling method
        return dist.rvs(size=num_samples)


def cf_interval(alpha: float, std: float, num_samples: int) -> float:
    """
    Calculates the two tails margin of error given the desired input. The margin of error is the value added and subtracted by the sample mean to obtain the confidence interval

    Sample sizes less then equal to 30 use t score, greater then 30 use z score

    Args:
        alpha (float): The significance level for the interval
        std (float): The standard deviation of the data
        num_samples (int): The number of samples in the data

    Returns:
        float: The margin of error
    """
    # two tails
    alpha = alpha / 2

    if num_samples > 30:
        score = stats.norm.ppf(alpha)
    else:
        score = stats.t.ppf(1 - alpha, num_samples - 1)

    return score * std / np.sqrt(num_samples)


def pvrpm_sim(case: SamCase):
    """
    Run the PVRPM simulation on a specific case. Results will be saved to the folder specified in the configuration.

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
    """
    pass
