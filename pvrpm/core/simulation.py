import os
import time

import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
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


def simulate_day(case: SamCase, comp: Components, day: int):
    """
    Updates and increments the simulation by a day, performing all neccesary component updates.

    Args:
        case (:obj:`SamCase`): The current Sam Case of the simulation
        comp (:obj:`Components`): The components class containing all the outputs for this simulation
        day (int): Current day in the simulation
    """
    for c in ck.component_keys:
        if not case.config.get(c, None):
            continue

        df = comp.comps[c]
        # if component can't fail, just continue
        if case.config[c][ck.CAN_FAIL]:
            comp.uptime[c] += case.config[c][ck.NUM_COMPONENT]

            # decrement time to failures for operational modules
            df.loc[df["state"] == 1, "time_to_failure"] -= 1

            # fail components when their time has come
            comp.fail_component(c)

            if case.config[c][ck.CAN_REPAIR]:
                # decrement time to repair for failed modules
                df.loc[df["state"] == 0, "time_to_repair"] -= 1

                # repair components when they are done and can be repaired
                comp.repair_component(c, day)

            if case.config[c].get(ck.WARRANTY, None):
                df["time_left_on_warranty"] -= 1

            # availability
            if c == ck.GRID:
                # for the grid only, the availability is based on the full 24-hour day.
                df.loc[df["state"] == 0, "avail_downtime"] += 24
            else:
                # else, use the sun hours for this day
                df.loc[df["state"] == 0, "avail_downtime"] += case.daylight_hours[day % 365]

        # module can still degrade even if it cant fail
        if case.config[c].get(ck.DEGRADE, None):
            df["days_of_degradation"] += 1
            df["degradation_factor"] = [
                component_degradation(case.config[c][ck.DEGRADE] / 365, d) for d in df["days_of_degradation"]
            ]


def run_system_realization(case: SamCase) -> Components:
    """
    Run a full realization for calculating costs

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation

    Returns:
        :obj:`Components`: The components object which contains all the data for this realization
    """
    # data storage
    comp = Components(case)
    lifetime = int(case.config[ck.LIFETIME_YRS])

    if case.config[ck.TRACKING]:
        comp.tracker_power_loss_factor[0] = 1
        comp.tracker_availability[0] = 1

    # initial timestep
    comp.module_degradation_factor[0] = comp.current_degradation()
    comp.dc_power_availability[0] = comp.dc_availability()
    comp.ac_power_availability[0] = comp.ac_availability()

    for i in tqdm(range(1, lifetime * 365), ascii=True, desc="Running realization", unit="day"):
        # calculate new labor rate each year
        if i == 1 or i % 365 == 0:
            comp.labor_rate = case.config[ck.LABOR_RATE] * np.power((1 + case.config[ck.INFLATION]) / 100, i)
            if case.config[ck.TRACKING]:
                for fail in case.config[ck.TRACKER][ck.FAILURE].keys():
                    case.config[ck.TRACKER][ck.FAILURE][fail][ck.COST] *= np.power(
                        (1 + case.config[ck.INFLATION]) / 100, i
                    )

        # timestep is applied each day
        simulate_day(case, comp, i)

        if case.config[ck.TRACKING]:
            comp.tracker_availability[i], comp.tracker_power_loss_factor[i] = comp.tracker_power_loss(i)

        comp.module_degradation_factor[i] = comp.current_degradation()
        comp.dc_power_availability[i] = comp.dc_availability()
        comp.ac_power_availability[i] = comp.ac_availability()

    # create same performance adjustment tables for avail, degradation, tracker losses
    logger.info("Running SAM simulation for this realization...")
    if case.config[ck.TRACKING]:
        daily_dc_loss = 100 * (
            1 - (comp.dc_power_availability * comp.module_degradation_factor * comp.tracker_power_loss_factor)
        )
    else:
        daily_dc_loss = 100 * (1 - (comp.dc_power_availability * comp.module_degradation_factor))

    daily_ac_loss = 100 * (1 - comp.ac_power_availability)

    case.value("en_dc_lifetime_losses", 1)
    case.value("dc_lifetime_losses", list(daily_dc_loss))

    case.value("en_ac_lifetime_losses", 1)
    case.value("ac_lifetime_losses", list(daily_ac_loss))

    o_m_yearly_costs = np.zeros(lifetime)
    for c in ck.component_keys:
        if not case.config.get(c, None):
            continue

        comp_yearly_cost = np.sum(np.reshape(comp.costs[c], (lifetime, 365)), axis=1)
        o_m_yearly_costs += comp_yearly_cost

    case.value("om_fixed", list(o_m_yearly_costs))

    s_time = time.time()
    case.simulate()
    logger.info("Realization simulation took {:.2f} seconds".format(time.time() - s_time))

    # reset tracker failure cost
    if case.config[ck.TRACKING]:
        for fail in case.config[ck.TRACKER][ck.FAILURE].keys():
            case.config[ck.TRACKER][ck.FAILURE][fail][ck.COST] = comp.original_tracker_cost

    # add the results of the simulation to the components class and return
    comp.timeseries_dc_power = case.value("dc_net")
    comp.timeseries_ac_power = case.value("gen")
    comp.lcoe = case.value("lcoe_real")
    # remove the first element from cf_energy_net because it is always 0, representing year 0
    comp.annual_energy = np.array(case.output("cf_energy_net")[1:])
    return comp


def pvrpm_sim(case: SamCase, save_graphs: bool = False):
    """
    Run the PVRPM simulation on a specific case. Results will be saved to the folder specified in the configuration.

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
        save_graphs (bool): Whether to save output graphs with results
    """
    save_path = case.config[ck.RESULTS_FOLDER]

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

    summary_index = ["Base Case"]
    summary_data = {"lcoe": [case.output("lcoe_real")]}
    lifetime = int(case.config[ck.LIFETIME_YRS])

    # ac energy
    # remove the first element from cf_energy_net because it is always 0, representing year 0
    base_annual_energy = np.array(case.output("cf_energy_net")[1:])
    cumulative_ac_energy = np.cumsum(base_annual_energy)

    for i in range(int(lifetime)):
        summary_data[f"annual_ac_energy_{i+1}"] = [base_annual_energy[i]]
        summary_data[f"cumulative_ac_energy_{i+1}"] = [cumulative_ac_energy[i]]

    # dc energy
    timeseries_dc_power = case.output("dc_net")
    dc_energy = summarize_dc_energy(timeseries_dc_power, lifetime)
    for i in range(len(dc_energy)):
        summary_data[f"dc_energy_{i+1}"] = [dc_energy[i]]

    # calculate availability using sun hours
    # contains every hour in the year and whether is sun up, down, sunrise, sunset
    sunup = case.value("sunup")

    # 0 sun is down, 1 sun is up, 2 surnise, 3 sunset, we only considered sun up (1)
    sunup = np.reshape(np.array(sunup), (365, 24))
    # zero out every value except where the value is 1 (for sunup)
    sunup = np.where(sunup == 1, sunup, 0)
    # sum up daylight hours for each day
    case.daylight_hours = np.sum(sunup, axis=1)
    case.annual_daylight_hours = np.sum(case.daylight_hours)

    # realize what we are doing in life
    results = []
    for i in range(case.config[ck.NUM_REALIZATION]):
        logger.info(f"Running system realization {i + 1}...")
        results.append(run_system_realization(case))

    # write all those results
    # TODO: maybe move this to another function?
    # per realization results
    day_index = np.arange(lifetime * 365) + 1
    hour_index = np.arange(lifetime * 365 * 24)
    year_index = np.arange(lifetime) + 1
    yearly_cost_index = []
    degradation_data = {}
    timeseries_dc_data = {}
    timeseries_ac_data = {}
    yearly_cost_data = {}
    for i, comp in enumerate(results):
        # daily degradation
        degradation_data[f"Realization {i+1}"] = comp.module_degradation_factor

        # power
        timeseries_dc_data[f"Realization {i+1}"] = comp.timeseries_dc_power
        timeseries_ac_data[f"Realization {i+1}"] = comp.timeseries_ac_power

        # yearly cost
        yearly_cost_index.append(f"Realization {i+1}")
        for c in ck.component_keys:
            if not case.config.get(c, None):
                continue
            if c not in yearly_cost_data:
                yearly_cost_data[c] = []
            yearly_cost_data[c] += list(np.sum(np.reshape(comp.costs[c], (lifetime, 365)), axis=1))

        # summary
        summary_index.append(f"Realization {i+1}")
        summary_data["lcoe"] += [comp.lcoe]

        # ac energy
        # remove the first element from cf_energy_net because it is always 0, representing year 0
        cumulative_ac_energy = np.cumsum(comp.annual_energy)

        for i in range(int(lifetime)):
            summary_data[f"annual_ac_energy_{i+1}"] += [comp.annual_energy[i]]
            summary_data[f"cumulative_ac_energy_{i+1}"] += [cumulative_ac_energy[i]]

        # dc energy
        dc_energy = summarize_dc_energy(comp.timeseries_dc_power, lifetime)
        for i in range(len(dc_energy)):
            summary_data[f"dc_energy_{i+1}"] += [dc_energy[i]]

        # calculate total failures, availability, mttf, mtbf, etc
        for c in ck.component_keys:
            if not case.config.get(c, None):
                continue
            if case.config[c][ck.CAN_FAIL]:
                if f"{c}_total_failures" not in summary_data:
                    summary_data[f"{c}_total_failures"] = [None]  # no failures for base case
                sum_fails = comp.comps[c]["cumulative_failures"].sum()
                summary_data[f"{c}_total_failures"] += [sum_fails]
                for fail in case.config[c].get(ck.FAILURE, {}).keys():
                    if f"{c}_failures_by_type_{fail}" not in summary_data:
                        summary_data[f"{c}_failures_by_type_{fail}"] = [None]
                    summary_data[f"{c}_failures_by_type_{fail}"] += [comp.comps[c][f"failure_by_type_{fail}"].sum()]

                # mean time between failure
                if f"{c}_mtbf" not in summary_data:
                    summary_data[f"{c}_mtbf"] = [None]
                summary_data[f"{c}_mtbf"] += [comp.uptime[c] / sum_fails]
            else:
                # mean time between failure
                if f"{c}_mtbf" not in summary_data:
                    summary_data[f"{c}_mtbf"] = [None]
                summary_data[f"{c}_mtbf"] += [comp.uptime[c]]

            # availability
            if f"{c}_availability" not in summary_data:
                summary_data[f"{c}_availability"] = [None]
            summary_data[f"{c}_availability"] += [
                (
                    1
                    - (comp.comps[c]["avail_downtime"].sum() / (lifetime * case.annual_daylight_hours))
                    / case.config[c][ck.NUM_COMPONENT]
                )
            ]

    # generate dataframes
    summary_results = pd.DataFrame(index=summary_index, data=summary_data)
    summary_results.index.name = "Realization"
    # reorder columns for summary results
    reorder = [summary_results.columns[0]]  # lcoe
    reorder += list(summary_results.columns[lifetime * 3 + 1 :])  # failures and avail
    reorder += list(summary_results.columns[1 : lifetime * 3 + 1])  # energy
    summary_results = summary_results[reorder]

    degradation_results = pd.DataFrame(index=day_index, data=degradation_data)
    dc_power_results = pd.DataFrame(index=hour_index, data=timeseries_dc_data)
    ac_power_results = pd.DataFrame(index=hour_index, data=timeseries_ac_data)
    dc_power_results.index.name = "Hour"
    ac_power_results.index.name = "Hour"
    degradation_results.index.name = "Day"

    cost_index = pd.MultiIndex.from_product([yearly_cost_index, year_index], names=["Realization", "Year"])
    yearly_cost_results = pd.DataFrame(index=cost_index, data=yearly_cost_data)
    yearly_cost_results["total"] = yearly_cost_results.sum(axis=1)

    stats_append = []
    min = summary_results.min()
    min.name = "min"
    stats_append.append(min)

    max = summary_results.max()
    max.name = "max"
    stats_append.append(max)

    mean = summary_results.mean()
    mean.name = "mean"
    stats_append.append(mean)

    median = summary_results.median()
    median.name = "median"
    stats_append.append(median)

    std = summary_results.std()
    std.name = "stddev"
    stats_append.append(std)

    conf_interval = case.config[ck.CONF_INTERVAL]
    conf_int = cf_interval(1 - (conf_interval / 100), std, case.config[ck.NUM_REALIZATION])

    lower_conf = mean - conf_int
    lower_conf.name = f"{conf_interval}% lower confidence interval of mean"
    stats_append.append(lower_conf)

    upper_conf = mean + conf_int
    upper_conf.name = f"{conf_interval}% upper confidence interval of mean"
    stats_append.append(upper_conf)

    # TODO: p test, need to figure out what they are doing, should be a t test

    summary_results = summary_results.append(stats_append)

    # save results
    summary_results.to_csv(os.path.join(save_path, "PVRPM_Summary_Results.csv"), index=True)
    degradation_results.to_csv(os.path.join(save_path, "Daily_Degradation.csv"), index=True)
    dc_power_results.to_csv(os.path.join(save_path, "Timeseries_DC_Power.csv"), index=True)
    ac_power_results.to_csv(os.path.join(save_path, "Timeseries_AC_Power.csv"), index=True)
    yearly_cost_results.to_csv(os.path.join(save_path, "Yearly_Costs_By_Component.csv"), index=True)

    logger.info(f"Results saved to {save_path}")
