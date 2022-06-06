from typing import Any, Tuple
import pkgutil
import importlib

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
from scipy.special import gamma, gammaln

from pvrpm.core.enums import ConfigKeys as ck


# override to getattr to get modules case insensitve
def getattr_override(obj: Any, attr: str) -> Any:
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


# TODO: there has to be a better way to do this...
def load_pysam_modules():
    """
    Loads ALL of PySAM's modules manually and globalizes them

    This is needed because PySAM is a wrapper for the ssc and sdk of SAM, which includes dynamic modules that are not properly defined for pybind, so using pkgutil's walk_packages function does not work (import error). Since the modules need to be loaded in order for getattr to find it, this must be done once when the program starts
    """
    global pysam
    import PySAM as pysam

    for loader, module_name, is_pkg in pkgutil.walk_packages(pysam.__path__):
        try:
            importlib.import_module(f"{pysam.__name__}.{module_name}")
        except:
            pass


def filename_to_module(filename: str) -> object:
    """
    Takes the filename of an exported json file from SAM, extracts the module name, and returns a callback to that module that can be used to create an object

    Args:
        filename (str): Filename of the exported case

    Returns:
        :obj:`PySAM`: PySAM object the file represents
    """
    # for certain modules the name from SAM doesnt match up with the module name (extra spaces in module name)
    broken_modules = ["host_developer"]
    for mod in broken_modules:
        if mod in filename:
            module_str = filename.strip().split("_")[-2:]
            module_str = "".join(module_str).split(".")[0].strip()
            return getattr_override(pysam, module_str)

    # if not a broken module:
    # SAM case file exporting should be underscores, with the last word being the module type
    module_str = filename.strip().split("_")[-1].split(".")[0].strip()
    return getattr_override(pysam, module_str)


def summarize_dc_energy(dc_power_output: tuple, split: int) -> np.array:
    """
    Calculates the DC energy (kWh) based on an input array of timeseries DC power (kW) for the system lifetime (likely the 'dc_net' output from SAM)

    Can be used to summarize similar hourly, daily data to yearly

    Args:
        dc_power_output (:obj:`tuple`): Tuple output from SAM simulation
        split (int): The frequency to split the data too, typically this is the number of years the system was simulated for (system_lifetime_yrs)

    Returns:
        :obj:`np.array`: Numpy array of length system_lifetime_yrs containing the yearly energy in kWh
    """
    data = np.array(dc_power_output)
    data = np.reshape(data, (int(split), int(len(dc_power_output) / split)))
    return np.sum(data, axis=1)


def component_degradation(percent_per_day: float, t: int) -> float:
    """
    Calculate the degradation of a component given the time since last replacement

    Args:
        percent_per_day (float): The percent degradation per day of the module
        t (int): Time since the module was last replaced, or if its a new module, installed

    Returns:
        float: The performance of the module, between 0 and 1

    Note:
        This gives the overall module performance based on degradation, so if the module has degraded 2 percent so far, this function returns 0.98
    """

    return 1 / np.power((1 + percent_per_day / 100), t)


def sample(distribution: str, parameters: dict, num_samples: int) -> np.array:
    """
    Sample data from a distribution. If distribution is a supported distribution, parameters should be a dictionary with keys "mean" and "std". Otherwise, distribution should be a scipy stats function and parameters be the kwargs for the distribution.

    Supported Distributions (only requires mean and std):
        - lognormal
        - normal
        - uniform (one std around mean)
        - weibull
        - exponential

    Args:
        distribution (str): Name of the distribution function
        parameters (:obj:`dict`): Kwargs for the distribution (for a supported distribution should only be the mean and std)
        num_samples (int): Number of samples to return from distribution

    Returns:
        :obj:(list): List of floats containing samples from the distribution
    """
    distribution = distribution.lower().strip()

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
        # this fails for certain parameter ranges, raising a runtime error
        # see https://github.com/scipy/scipy/issues/12134 for reference
        def _h(c):
            r = np.exp(gammaln(2 / c) - 2 * gammaln(1 / c))
            return np.sqrt(1 / (2 * c * r - 1))

        if ck.STD in parameters:
            mean, std = parameters[ck.MEAN], parameters[ck.STD]
            c0 = 1.27 * np.sqrt(mean / std)
            c, info, ier, msg = scipy.optimize.fsolve(lambda t: _h(t) - (mean / std), c0, xtol=1e-10, full_output=True,)

            # Test residual rather than error code.
            if np.abs(info["fvec"][0]) > 1e-8:
                raise RuntimeError(f"with mean={mean} and std={std}, solve failed: {msg}")

            c = c[0]
        else:
            mean, c = parameters[ck.MEAN], parameters[ck.SHAPE]

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

    # scipy rvs uses rou sampling method
    return dist.rvs(size=num_samples)


def get_higher_components(
    top_level: str, start_level: str, case, start_level_df: pd.DataFrame = None,
) -> Tuple[np.array, np.array, int]:
    """
    Calculates the indicies of the top level that correspond to the given level df indicies and returns the given level indicies count per top level component and the total number of start_level components per top_level component

    Args:
        top_level (str): The string name of the component level to calculate indicies for
        start_level (str): The string name of the component level to start at
        case (SamCase): The case object for this simulation
        start_level_df (:obj:`pd.DataFrame`, Optional): The dataframe of the component level for which to find the corresponding top level indicies for

    Returns:
        tuple(:obj:`np.array`, :obj:`np.array`, int): If start_level_df is given, returns the top level indicies, the number of start_level components in start_level_df per top level index, and the total number of start_level components per top_level component. If start_level_df is None this only returns the total number of start_level components per top_level component.
    """
    # the number of disconnects equals the number of inverters, if this every changes this would need to be changed
    # otherwise, the inverter per trans is the same for disconnects
    # dictionaries to make is easier transitioning between levels
    component_per = {
        ck.TRANSFORMER: case.config[ck.INVERTER_PER_TRANS],
        ck.DISCONNECT: 1,
        ck.INVERTER: case.config[ck.COMBINER_PER_INVERTER],
        ck.COMBINER: case.config[ck.STR_PER_COMBINER],
        ck.STRING: case.config[ck.MODULES_PER_STR],
        ck.MODULE: 1,
    }

    # hierarchy of component levels:
    component_hier = {
        ck.MODULE: 0,
        ck.STRING: 1,
        ck.COMBINER: 2,
        ck.INVERTER: 3,
        ck.DISCONNECT: 4,
        ck.TRANSFORMER: 5,
    }

    above_levels = [
        c
        for c in component_hier.keys()
        if component_hier[c] > component_hier[start_level] and component_hier[c] <= component_hier[top_level]
    ]

    total_comp = 1
    # above levels is ordered ascending
    for level in above_levels:
        total_comp *= component_per[level]

    if start_level_df is not None:
        indicies = start_level_df.index.copy()
        indicies = np.floor(indicies / total_comp)
        # sum up the number of occurences for each index and return with total number of components at start level per top level
        indicies, counts = np.unique(indicies, return_counts=True)
        return indicies.astype(np.int64), counts.astype(np.int64), int(total_comp)
    else:
        return int(total_comp)
