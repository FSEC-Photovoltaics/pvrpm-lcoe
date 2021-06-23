import os
import time

import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
from scipy.special import gamma, gammaln
from tqdm import tqdm

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import summarize_dc_energy
from pvrpm.core.logger import logger


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


def component_degradation(percent_per_day: float, t: int):
    """
    Calculate the degradation of a component given the time since last replacement

    Args:
        percent_per_day (float): The percent degradation per day of the module
        t (int): Time since the module was last replaced, or if its a new module installed

    Returns:
        float: The performance of the module, between 0 and 1

    Note:
        This gives the overall module performance based on degradation, so if the module has degraded 2 percent so far, this function returns 0.98
    """

    return 1 / np.power((1 + percent_per_day / 100), t)


def pvrpm_sim(case: SamCase):
    """
    Run the PVRPM simulation on a specific case. Results will be saved to the folder specified in the configuration.

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
    """
    # run the dummy base case
    case.value("en_dc_lifetime_losses", 0)
    case.value("en_ac_lifetime_losses", 0)

    case.value("om_fixed", [0])

    module_degradation_rate = case.config[ck.MODULE].get(ck.DEGRADE, 0) / 365

    degrade = [
        (1 - component_degradation(module_degradation_rate, i)) * 100
        for i in range(int(case.config[ck.LIFETIME_YRS] * 365))
    ]

    case.value("en_dc_lifetime_losses", 1)
    case.value("dc_lifetime_losses", degrade)

    logger.info("Running base case simulation...")
    start = time.time()
    case.simulate()
    logger.info("Base case simulation took: {:.2f} seconds".format(time.time() - start))

    index = ["Base Case"]
    data = {"lcoe": [case.output("lcoe_real")]}
    lifetime = case.value("analysis_period")

    # ac energy
    # remove the first element from cf_energy_net because it is always 0, representing year 0
    base_annual_energy = np.array(case.output("cf_energy_net")[1:])
    cumulative_ac_energy = np.cumsum(base_annual_energy)

    for i in range(int(lifetime)):
        data[f"annual_ac_energy_{i+1}"] = [base_annual_energy[i]]
        data[f"cumulative_ac_energy_{i+1}"] = [cumulative_ac_energy[i]]

    # dc energy
    timeseries_dc_power = case.output("dc_net")
    dc_energy = summarize_dc_energy(timeseries_dc_power, case.config[ck.LIFETIME_YRS])
    for i in range(len(dc_energy)):
        data[f"dc_energy_{i+1}"] = [dc_energy[i]]

    summary_results = pd.DataFrame(index=index, data=data)
    summary_results.index.name = "Realization"
