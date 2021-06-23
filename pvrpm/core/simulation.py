import os
import time

import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.components import Components
from pvrpm.core.utils import summarize_dc_energy
from pvrpm.core.logger import logger


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


def run_system_realization(case: SamCase):
    """
    Run a full realization for calculating costs

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
    """
    # data storage
    comp = Components(case)
    lifetime = case.config[ck.LIFETIME_YRS]
    module_degradation_factor = np.zeros(int(lifetime * 365))
    dc_power_availability = np.zeros(int(lifetime * 365))
    ac_power_availability = np.zeros(int(lifetime * 365))
    labor_rate = 0

    if case.config[ck.TRACKING]:
        tracker_power_loss_factor = np.zeros(int(lifetime * 365))
    else:
        tracker_power_loss_factor = None


def pvrpm_sim(case: SamCase, save_graphs: bool = False):
    """
    Run the PVRPM simulation on a specific case. Results will be saved to the folder specified in the configuration.

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
        save_graphs (bool): Whether to save output graphs with results
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

    # calculate availability using sun hours
    # contains every hour in the year and whether is sun up, down, sunrise, sunset
    sunup = case.value("sunup")

    # 0 sun is down, 1 sun is up, 2 surnise, 3 sunset, we only considered sun up (1)
    sunup = np.reshape(np.array(sunup), (365, 24))
    # zero out every value except where the value is 1 (for sunup)
    sunup = np.where(sunup == 1, sunup, 0)
    # sum up daylight hours for each day
    daylight_hours = np.sum(sunup, axis=1)
    annual_daylight_hours = np.sum(daylight_hours)

    # realize what we are doing in life
    for i in tqdm(
        range(case.config[ck.NUM_REALIZATION]), ascii=True, desc="Running system realizations", unit="realization"
    ):
        run_system_realization(case)
