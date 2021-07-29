from typing import Tuple

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
from scipy.special import gamma, gammaln

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import summarize_dc_energy
from pvrpm.core.logger import logger
import time


class Components:
    """
    Data container for each component in the simulation, as well as component and simulation level data
    """

    def __init__(self, case: SamCase):
        self.case = case

        self.comps = {}
        self.costs = {}

        # keep track of total days spent on monitoring and repairs
        self.total_repair_time = {}
        self.total_monitor_time = {}

        # monitoring level bins to keep track of number of failures per top level monitoring
        self.monitor_bins = {}

        # the number of disconnects equals the number of inverters, if this every changes this would need to be changed
        # otherwise, the inverter per trans is the same for disconnects
        # dictionaries to make is easier transitioning between levels
        self.component_per = {
            ck.TRANSFORMER: self.case.config[ck.INVERTER_PER_TRANS],
            ck.DISCONNECT: 1,
            ck.INVERTER: self.case.config[ck.COMBINER_PER_INVERTER],
            ck.COMBINER: self.case.config[ck.STR_PER_COMBINER],
            ck.STRING: self.case.config[ck.MODULES_PER_STR],
            ck.MODULE: 1,
        }

        # hierarchy of component levels:
        self.component_hier = {
            ck.MODULE: 0,
            ck.STRING: 1,
            ck.COMBINER: 2,
            ck.INVERTER: 3,
            ck.DISCONNECT: 4,
            ck.TRANSFORMER: 5,
        }

        # keep track of static monitoring for every level
        index = []
        data = []
        for name, monitor_config in self.case.config.get(ck.STATIC_MONITOR, {}).items():
            index.append(name)
            data.append(monitor_config[ck.INTERVAL])
        if index:
            self.static_monitoring = pd.Series(index=index, data=data)
        else:
            self.static_monitoring = None

        lifetime = self.case.config[ck.LIFETIME_YRS]
        # every component level will contain a dataframe containing the data for all the components in that level
        for c in ck.component_keys:
            if self.case.config.get(c, None):
                self.comps[c] = self.initialize_components(c)
                self.total_repair_time[c] = 0
                self.total_monitor_time[c] = 0
                self.costs[c] = np.zeros(lifetime * 365)

                if (
                    self.case.config[c].get(ck.COMP_MONITOR, None)
                    and self.case.config[c][ck.COMP_MONITOR].get(ck.FAIL_PER_THRESH, None) is not None
                ):
                    top_level = self.case.config[c][ck.COMP_MONITOR][ck.LEVELS]
                    bins = np.zeros(len(self.comps[c]))
                    num_per_top = self.get_higher_components(top_level, c)
                    indicies = np.floor(self.comps[c].index / num_per_top)
                    self.monitor_bins[c] = {
                        "top_level": top_level,
                        "indicies": indicies,
                        "bins": bins,
                        "num_per_top": num_per_top,
                    }

        # additional aggreate data to track during simulation
        self.module_degradation_factor = np.zeros(lifetime * 365)
        self.dc_power_availability = np.zeros(lifetime * 365)
        self.ac_power_availability = np.zeros(lifetime * 365)
        self.labor_rate = 0

        # Data from simulation at end of realization
        self.timeseries_dc_power = None
        self.timeseries_ac_power = None
        self.lcoe = None
        self.annual_energy = None

        self.tax_cash_flow = None
        self.losses = {}

        if case.config[ck.TRACKING]:
            self.tracker_power_loss_factor = np.zeros(lifetime * 365)
            self.tracker_availability = np.zeros(lifetime * 365)
            fail = list(case.config[ck.TRACKER][ck.FAILURE].keys())[0]
            # TODO: why is this based only on the first failure?
            self.original_tracker_cost = self.case.config[ck.TRACKER][ck.FAILURE][fail].get(ck.COST, 0)

    @staticmethod
    def sample(distribution: str, parameters: dict, num_samples: int, method: str = "rou") -> np.array:
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
            # this fails for certain parameter ranges, raising a runtime error
            # see https://github.com/scipy/scipy/issues/12134 for reference
            def _h(c):
                r = np.exp(gammaln(2 / c) - 2 * gammaln(1 / c))
                return np.sqrt(1 / (2 * c * r - 1))

            if ck.STD in parameters:
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

        if method == "lhs":
            pass
        else:
            # scipy rvs uses rou sampling method
            return dist.rvs(size=num_samples)

    @staticmethod
    def compound_failures(function: str, parameters: dict, num_fails: int):
        """
        Compounds the failures using the provided function and it's parameters to calculate the number of days to reduce the time to detection by

        Possible functions and their parameters are:
            - step: Failures follow a step function, each step reducing the detection time by a static amount
                - threshold (float): fraction 0 <= threshold <= 1 that signifies the amount of modules that must fail before next step is reached. So if this is 0.2, every 0.2 * total_components components that fail will reduce detection time by step
                - step (int): The amount of days to reduce detection time for every step. So 2 steps reduces detection time by 2 * step
            - exponential: Failures compound on an exponential function
                - base (float): The base for the exponential function > 0
            - log: Failures compound on a logorithmic function
                - base (float): The base for the log function > 0
            - linear: Failures compound linearly
                - slope (float): Slope of the linear function > 0
            - constant: Each failure reduces the time to detection by a static fraction constant
                - constant (float): fraction 0 <= frac <= 1 that specifies how much of the overall time each failure reduces. So if fraction is 0.1, a failure will reduce time to detection by "time_to_detection * 0.1"
        """
        pass

    def get_higher_components(
        self,
        top_level: str,
        start_level: str,
        start_level_df: pd.DataFrame = None,
    ) -> Tuple[np.array, np.array, int]:
        """
        Calculates the indicies of the top level that correspond to the given level df indicies and returns the given level indicies count per top level component and the total number of start_level components per top_level component

        Args:
            top_level (str): The string name of the component level to calculate indicies for
            start_level (str): The string name of the component level to start at
            start_level_df (:obj:`pd.DataFrame`, Optional): The dataframe of the component level for which to find the corresponding top level indicies for

        Returns:
            tuple(:obj:`np.array`, :obj:`np.array`, int): If start_level_df is given, returns the top level indicies, the number of start_level components in start_level_df per top level index, and the total number of start_level components per top_level component. If start_level_df is None this only returns the total number of start_level components per top_level component.
        """

        above_levels = [
            c
            for c in self.component_hier.keys()
            if self.component_hier[c] > self.component_hier[start_level]
            and self.component_hier[c] <= self.component_hier[top_level]
        ]

        total_comp = 1
        # above levels is ordered ascending
        for level in above_levels:
            total_comp *= self.component_per[level]

        if start_level_df is not None:
            indicies = start_level_df.index
            indicies = np.floor(indicies / total_comp)
            # sum up the number of occurences for each index and return with total number of components at start level per top level
            indicies, counts = np.unique(indicies, return_counts=True)
            return indicies.astype(np.int64), counts.astype(np.int64), int(total_comp)
        else:
            return int(total_comp)

    def initialize_components(self, component_level: str) -> pd.DataFrame:
        """
        Initalizes all components for the first time

        Args:
            component_level (str): The configuration key for this component level

        Returns:
            :obj:`pd.DataFrame`: A dataframe containing the initalized values for this component level

        Note: Individual components have these columns:
            - state (bool): Operational (True) or failed (False)
            - defective (bool): Whehter component has a defect. True means the component is also eligible for the defective failure mode
            - time_to_failure (float): Number of days until component fails
            - failure_type (str): The name of the failure type time_to_failure represents
            - time_to_repair (float): Number of days from failure until compoent is repaired
            - time_to_detection (float): Number of days until the component failure is detected and repairs start
            - repair_times (float): the total repair time for this repair
            - monitor_times (float): the total monitoring time before repairs start
            - time_left_on_warranty (int): Number of days left on warranty (if applicable)
            - cumulative_failures (int): Total number of failures for that component
            - cumulative_oow_failures (int): Total number of out of warranty failures (if applicable)
            - failure_by_type_n (int): Each failure will have its own column with the number of failures of this type
            - defective (bool): Whether this component is defective or not (for defective failures)
            - defective_failures (int): Total number of defective failures
            - avail_downtime (int): How many hours this component was available
            - degradation_factor (float): 1 - percentage component has degraded at this time (module only)
            - days_of_degradation (int): Time that the module has been degrading for
        """
        component_info = self.case.config[component_level]

        component_ind = [i for i in range(component_info[ck.NUM_COMPONENT])]
        df = pd.DataFrame(index=component_ind)

        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())
        monitor_modes = list(component_info.get(ck.MONITORING, {}).keys())
        repair_modes = list(component_info.get(ck.REPAIR, {}).keys())

        # operational
        df["state"] = 1
        # degradation gets reset to zero
        if component_info.get(ck.DEGRADE, None):
            df["days_of_degradation"] = 0
            df["degradation_factor"] = 1

        if component_info.get(ck.WARRANTY, None):
            df["time_left_on_warranty"] = component_info[ck.WARRANTY][ck.DAYS]
        else:
            df["time_left_on_warranty"] = 0

        df["cumulative_failures"] = 0
        df["cumulative_oow_failures"] = 0
        df["avail_downtime"] = 0

        # if component can't fail, nothing else needs to be initalized
        if not component_info[ck.CAN_FAIL]:
            return df

        possible_failure_times = np.zeros((component_info[ck.NUM_COMPONENT], len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            # initalize failure mode by type
            df[f"failure_by_type_{mode}"] = 0
            fail = component_info[ck.FAILURE][mode]
            if fail.get(ck.FRAC, None):
                # choose a percentage of modules to be defective
                sample = np.random.random_sample(size=component_info[ck.NUM_COMPONENT])
                df["defective"] = sample < fail[ck.FRAC]

                sample = self.sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(list(df["defective"]), sample, np.finfo(np.float32).max)

            elif fail.get(ck.FRAC, None) is None:
                # setup failure times for each component
                # TODO: pvrpm instead takes a large sample once and just pulls from values from that vector, figure out if that makes a difference
                possible_failure_times[:, i] = self.sample(
                    fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT]
                )

        failure_ind = np.argmin(possible_failure_times, axis=1)
        df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        df["failure_type"] = [failure_modes[i] for i in failure_ind]

        # time to replacement/repair in case of failure
        if not component_info[ck.CAN_REPAIR]:
            df["time_to_repair"] = 1  # just initalize to 1 if no repair modes, means components cannot be repaired
        elif len(repair_modes) == 1:
            # same repair mode for every repair
            repair = component_info[ck.REPAIR][repair_modes[0]]
            df["time_to_repair"] = self.sample(repair[ck.DIST], repair[ck.PARAM], component_info[ck.NUM_COMPONENT])
            df["repair_times"] = df["time_to_repair"].copy()
        else:
            modes = [repair_modes[i] for i in failure_ind]
            df["time_to_repair"] = np.array(
                [
                    self.sample(component_info[ck.REPAIR][m][ck.DIST], component_info[ck.REPAIR][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            df["repair_times"] = df["time_to_repair"].copy()

        # monitoring times, these will be added to the repair time for each component
        # basically, the time until each failure is detected
        if component_info[ck.CAN_REPAIR]:
            if component_info[ck.CAN_MONITOR]:
                if len(monitor_modes) == 1:
                    # same monitor mode for every failure
                    monitor = component_info[ck.MONITORING][monitor_modes[0]]
                    monitor = self.sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
                    df["monitor_times"] = monitor
                    df["time_to_detection"] = df["monitor_times"].copy()
                elif len(monitor_modes) > 1:
                    modes = [monitor_modes[i] for i in failure_ind]
                    monitor = np.array(
                        [
                            self.sample(
                                component_info[ck.MONITORING][m][ck.DIST], component_info[ck.MONITORING][m][ck.PARAM], 1
                            )[0]
                            for m in modes
                        ]
                    )
                    df["monitor_times"] = monitor
                    df["time_to_detection"] = df["monitor_times"].copy()
            # monitoring across levels, only applies if properly defined and monitoring at the component level can_monitor is false
            elif component_info.get(ck.COMP_MONITOR, None):
                monitor = component_info[ck.COMP_MONITOR]
                df["monitor_times"] = self.sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
                df["time_to_detection"] = df["monitor_times"].copy()
            # only static detection available
            elif component_info.get(ck.STATIC_MONITOR, None):
                # the proper detection time with only static monitoring is the difference between the static monitoring that occurs after the failure
                # this will be set when a component fails for simplicity sake, since multiple static monitoring schemes can be defined,
                # and the time to detection would be the the time from the component fails to the next static monitoring occurance
                # so, set these to None and assume they will be properly updated in the simulation
                df["monitor_times"] = None
                df["time_to_detection"] = None

        return df

    def tracker_power_loss(self, day: int) -> Tuple[float, float]:
        """
        Calculates the current loss factor due to failed trackers

        Args:
            day (int): Current day in the simulation

        Returns:
            Tuple[float, float]: The fraction of trackers operational and the loss factor for failed trackers
        """
        df = self.comps[ck.TRACKER]
        day = day % 365
        operational_trackers = len(df[df["state"] == 1])

        fraction = operational_trackers / len(df)
        adjusted_factor = 1
        if self.case.config[ck.TRACKER][ck.CAN_FAIL]:
            adjusted_factor = self.case.daily_tracker_coeffs[day] + fraction * (1 - self.case.daily_tracker_coeffs[day])

        return fraction, adjusted_factor

    def current_degradation(self) -> float:
        """
        Calculates the current module degradation, which is averaged for only operational modules, since the power production hit from degradation of non-operational modules would be double counted

        Returns:
            float: Average degradation of operational modules
        """

        if not self.case.config[ck.MODULE].get(ck.DEGRADE, None):
            return 1

        df = self.comps[ck.MODULE]
        operational_modules = df["state"] == 1
        fleet_degradation_sum = df[operational_modules]["degradation_factor"].sum()

        return fleet_degradation_sum / len(df)

    def dc_availability(self) -> float:
        """
        Calculates the availability of the DC power due to DC component outages, including modules, strings, and combiners

        Returns:
            float: Decimal percentage of available DC modules
        """
        combiner_df = self.comps[ck.COMBINER]
        string_df = self.comps[ck.STRING]
        module_df = self.comps[ck.MODULE]

        mods_per_str = self.case.config[ck.MODULES_PER_STR]
        str_per_comb = self.case.config[ck.STR_PER_COMBINER]
        operational_combiners = combiner_df.index[combiner_df["state"] == 1]
        operational_strings = string_df.index[string_df["state"] == 1]

        str_by_comb = np.reshape(np.array(string_df.index), (len(combiner_df), str_per_comb))
        modules_by_string = np.reshape(np.array(module_df.index), (len(string_df), mods_per_str))

        # remove combiners and strings that are not operational
        str_by_comb = str_by_comb[operational_combiners].flatten()
        # make sure strings under operational combiners are also operational
        operational_strings = np.intersect1d(str_by_comb, operational_strings)
        # get all modules under operational strings
        modules_by_string = modules_by_string[operational_strings].flatten()

        # note that here, "operational modules" means modules whose power is REACHING the inverter, regardless of whether the module itself is failed or not
        operational_modules = module_df.iloc[modules_by_string]["state"].sum()

        return operational_modules / len(module_df)

    def ac_availability(self) -> float:
        """
        Calculates the availability of AC power due to DC component outages, including inverters, disconnects, transformers, and the grid

        Returns:
            float: Decimal percentage of AC modules available
        """
        grid_df = self.comps[ck.GRID]
        transformer_df = self.comps[ck.TRANSFORMER]
        inverter_df = self.comps[ck.INVERTER]
        disconnect_df = self.comps[ck.DISCONNECT]

        invert_per_trans = self.case.config[ck.INVERTER_PER_TRANS]

        # theres always only 1 grid
        if grid_df.iloc[[0]]["state"][0] == 0:
            return 0

        operational_transformers = transformer_df.index[transformer_df["state"] == 1]

        inverter_by_trans = np.reshape(np.array(inverter_df.index), (len(transformer_df), invert_per_trans))
        # remove inoperal transformers and their inverters
        inverter_by_trans = inverter_by_trans[operational_transformers].flatten()
        inverter_df = inverter_df.iloc[inverter_by_trans]
        inverter_df = inverter_df[inverter_df["state"] == 1].index
        disconnect_df = disconnect_df.iloc[inverter_by_trans]
        disconnect_df = disconnect_df[disconnect_df["state"] == 1].index

        operational_inverters = len(np.intersect1d(inverter_df, disconnect_df))

        return operational_inverters / self.case.config[ck.INVERTER][ck.NUM_COMPONENT]

    def static_monitor(self, day: int):
        """
        Updates static monitoring which is independent of component levels by one day

        Args:
            day (int): Current day in the simulation
        """
        if self.static_monitoring is not None:
            self.static_monitoring -= 1
            mask = self.static_monitoring < 1
            if len(self.static_monitoring.loc[mask]) > 0:
                # static monitoring method commences, instantly detecting all failed components in their specified levels
                # which have not been detected by other monitoring methods yet.
                for monitor in self.static_monitoring.loc[mask].index:
                    config = self.case.config[ck.STATIC_MONITOR][monitor]
                    for level in config[ck.LEVELS]:
                        df = self.comps[level]
                        mask = (df["state"] == 0) & (df["time_to_detection"] >= 1)
                        monitor_comps = df.loc[mask].copy()
                        monitor_comps["monitor_times"] -= monitor_comps["time_to_detection"]

                        monitor_comps["time_to_detection"] = 0
                        df.loc[mask] = monitor_comps
                        # add cost to total costs (split evenly among all the levels for this day)
                        self.costs[level][day] += config[ck.COST] / len(config[ck.LEVELS])

                    # reset static monitoring time
                    self.static_monitoring[monitor] = self.case.config[ck.STATIC_MONITOR][monitor][ck.INTERVAL]

    def monitor_component(self, component_level: str):
        """
        Updates time to detection from component level, static, and cross component level monitoring based on number of failures

        Monitoring defined for each level is unaffected, only cross level monitoring on levels with no monitoring at that level have times updated based on failures, since monitoring defined at the component level uses the defined monitoring distribution for those components instead of the cross level monitoring

        Args:
            component_level (str): The component level to check for monitoring

        Note:
            Updates the underlying dataframes in place
        """
        df = self.comps[component_level]

        # only decrement monitoring on failed components where repairs have not started
        if self.case.config[component_level][ck.CAN_MONITOR]:
            mask = (df["state"] == 0) & (df["time_to_detection"] > 1)
            df.loc[mask, "time_to_detection"] -= 1
        elif self.case.config[component_level].get(ck.COMP_MONITOR, None):
            # fraction failed is number of failed components, no matter if its being repaired, etc
            mask = df["state"] == 0
            num_failed = len(mask)
            mask = mask & (df["time_to_detection"] > 1)

            conf = self.case.config[component_level][ck.COMP_MONITOR]
            global_threshold = False
            if conf.get(ck.FAIL_THRESH, None) is not None:
                frac_failed = num_failed / len(df)
                # only decrement time to detection once failure threshold is met
                if frac_failed > conf[ck.FAIL_THRESH]:
                    # TODO: compound the failures... later
                    df.loc[mask, "time_to_detection"] -= 1
                    global_threshold = True

            if conf.get(ck.FAIL_PER_THRESH, None) is not None and not global_threshold:
                # TODO: idk how to make this more efficent, if i can reduce this down to just a single iloc or loc, and try to elimate the call to get_higher_components i can bump up run time, cause right now it doubles runtime per realization
                # first, need to get the proper components of this level under the top monitor level
                data = self.monitor_bins[component_level]
                undetected_failed = df.loc[mask].copy()
                indicies, counts, total = self.get_higher_components(
                    data["top_level"],
                    component_level,
                    start_level_df=undetected_failed,
                )
                # calculate failure threshold for each top level monitoring component for the monitoried components
                subtract = []
                add = []
                prev_cnt = 0
                # here, cnt is the number of failed components per top level component, no matter if its being repaired, etc
                for ind, cnt in zip(indicies, counts):
                    # total currently failed regardless of detected, being repaired, etc
                    frac_failed = data["bins"][ind] / total
                    if frac_failed > conf[ck.FAIL_PER_THRESH]:
                        subtract += np.arange(prev_cnt, prev_cnt + cnt).tolist()
                    else:
                        add += np.arange(prev_cnt, prev_cnt + cnt).tolist()
                    prev_cnt += cnt

                if subtract:
                    undetected_failed.iloc[subtract, undetected_failed.columns.get_loc("time_to_detection")] -= 1
                if add:
                    undetected_failed.iloc[add, undetected_failed.columns.get_loc("monitor_times")] += 1
                df.loc[mask] = undetected_failed
            elif conf.get(ck.FAIL_THRESH, None) is not None and not global_threshold:
                # for calculating mttd and other data, increment the monitor times for every component by 1 to account for days where time to detection is not decremented because of the failure threshold
                mask = (df["state"] == 0) & (df["time_to_detection"] > 1)
                df.loc[mask, "monitor_times"] += 1

        elif self.case.config[component_level].get(ck.STATIC_MONITOR, None):
            mask = (df["state"] == 0) & (df["time_to_detection"] > 1)
            df.loc[mask, "time_to_detection"] -= 1

    def fail_component(self, component_level: str):
        """
        Changes state of a component to failed, incrementing failures and checking warranty only for failed components of each failure type

        Args:
            component_level (str): The component level to check for failures

        Note:
            Updates the underlying dataframes in place
        """
        failure_modes = list(self.case.config[component_level][ck.FAILURE].keys())
        df = self.comps[component_level]
        mask = (df["state"] == 1) & (df["time_to_failure"] < 1)
        failed_comps = df.loc[mask].copy()

        if len(failed_comps) > 0:
            failed_comps["time_to_failure"] = 0
            failed_comps["cumulative_failures"] += 1
            for fail in failure_modes:
                fail_mask = failed_comps["failure_type"] == fail
                failed_comps.loc[fail_mask, f"failure_by_type_{fail}"] += 1

            warranty_mask = failed_comps["time_left_on_warranty"] <= 0
            failed_comps.loc[warranty_mask, "cumulative_oow_failures"] += 1

            failed_comps["state"] = 0

            # update bins for monitoring per component for cross component monitoring
            if component_level in self.monitor_bins:
                data = self.monitor_bins[component_level]
                indicies, counts, _ = self.get_higher_components(
                    data["top_level"],
                    component_level,
                    start_level_df=failed_comps,
                )
                data["bins"][indicies] += counts

            # update time to detection times for component levels with only static monitoring
            # which will have None for monitor times
            try:
                if failed_comps["monitor_times"].isnull().any():
                    # monitor and time to detection will be the time to next static monitoring
                    static_monitors = list(self.case.config[component_level][ck.STATIC_MONITOR].keys())
                    # next static monitoring is the min of the possible static monitors for this component level
                    failed_comps["monitor_times"] = np.amin(self.static_monitoring[static_monitors])
                    failed_comps["time_to_detection"] = failed_comps["monitor_times"].copy()
            # fails if no monitoring defined, faster then just doing a check if the column exists or whatever
            except KeyError:
                pass

            df.loc[mask] = failed_comps

    def repair_component(self, component_level: str, day: int):
        """
        Changes the state of a component to operational once repairs are complete, only for components where the time to repair is zero

        Args:
            component_level (str): The component level of this repair
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        df = self.comps[component_level]
        component_info = self.case.config[component_level]

        mask = (df["state"] == 0) & (df["time_to_repair"] < 1)
        failure_modes = list(component_info[ck.FAILURE].keys())
        monitor_modes = list(component_info.get(ck.MONITORING, {}).keys())
        repair_modes = list(component_info[ck.REPAIR].keys())

        if len(df.loc[mask]) <= 0:
            return

        # add costs for each failure mode
        for mode in failure_modes:
            fail = component_info[ck.FAILURE][mode]
            fail_mask = mask & (df["failure_type"] == mode)
            repair_cost = fail[ck.COST] + self.labor_rate * fail[ck.LABOR]

            if component_info.get(ck.WARRANTY, None):
                warranty_mask = fail_mask & (df["time_left_on_warranty"] <= 0)
                self.costs[component_level][day] += len(df.loc[warranty_mask]) * repair_cost
            else:
                self.costs[component_level][day] += len(df.loc[fail_mask]) * repair_cost

        repaired_comps = df.loc[mask].copy()

        # update bins for monitoring per component for cross component monitoring
        if component_level in self.monitor_bins:
            data = self.monitor_bins[component_level]
            indicies, counts, _ = self.get_higher_components(
                data["top_level"],
                component_level,
                start_level_df=repaired_comps,
            )
            data["bins"][indicies] -= counts

        # add up the repair and monitoring times
        self.total_repair_time[component_level] += repaired_comps["repair_times"].sum()
        if (
            component_info[ck.CAN_MONITOR]
            or component_info.get(ck.COMP_MONITOR, None)
            or component_info.get(ck.STATIC_MONITOR, None)
        ):
            self.total_monitor_time[component_level] += repaired_comps["monitor_times"].sum()

        # reinitalize all repaired modules
        # degradation gets reset to 0 (module only)
        if component_level == ck.MODULE:
            repaired_comps["days_of_degradation"] = 0

        # components replaced that are out of warranty have their warranty renewed (if applicable)
        if component_info.get(ck.WARRANTY, None):
            warranty_mask = repaired_comps["time_left_on_warranty"] <= 0
            repaired_comps.loc[warranty_mask, "time_left_on_warranty"] = component_info[ck.WARRANTY][ck.DAYS]

        # TODO: maybe combine this with code in init component that does the same thing just on a different slice
        num_repaired = len(repaired_comps)
        possible_failure_times = np.zeros((num_repaired, len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            fail = component_info[ck.FAILURE][mode]
            if fail.get(ck.FRAC, None):
                # choose a percentage of modules to be defective
                sample = np.random.random_sample(size=num_repaired)
                repaired_comps["defective"] = sample < fail[ck.FRAC]

                sample = self.sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(
                    list(repaired_comps["defective"]),
                    sample,
                    np.finfo(np.float32).max,
                )

            elif fail.get(ck.FRAC, None) is None:
                # setup failure times for each component
                # TODO: pvrpm instead takes a large sample once and just pulls from values from that vector, figure out if that makes a difference
                possible_failure_times[:, i] = self.sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

        failure_ind = np.argmin(possible_failure_times, axis=1)
        repaired_comps["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        repaired_comps["failure_type"] = [failure_modes[i] for i in failure_ind]

        # time to replacement/repair in case of failure
        if len(repair_modes) == 1:
            # same repair mode for every repair
            repair = component_info[ck.REPAIR][repair_modes[0]]
            repaired_comps["time_to_repair"] = self.sample(repair[ck.DIST], repair[ck.PARAM], num_repaired)
            repaired_comps["repair_times"] = repaired_comps["time_to_repair"].copy()
        else:
            modes = [repair_modes[i] for i in failure_ind]
            repaired_comps["time_to_repair"] = np.array(
                [
                    self.sample(component_info[ck.REPAIR][m][ck.DIST], component_info[ck.REPAIR][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            repaired_comps["repair_times"] = repaired_comps["time_to_repair"].copy()

        # monitoring times, these will be added to the repair time for each component
        # basically, the time until each failure is detected
        if component_info[ck.CAN_MONITOR]:
            if len(monitor_modes) == 1:
                # same monitor mode for every failure
                monitor = component_info[ck.MONITORING][monitor_modes[0]]
                repaired_comps["monitor_times"] = self.sample(monitor[ck.DIST], monitor[ck.PARAM], num_repaired)
                repaired_comps["time_to_detection"] = repaired_comps["monitor_times"].copy()
            elif len(monitor_modes) > 0:
                modes = [monitor_modes[i] for i in failure_ind]
                monitor = np.array(
                    [
                        self.sample(
                            component_info[ck.MONITORING][m][ck.DIST], component_info[ck.MONITORING][m][ck.PARAM], 1
                        )[0]
                        for m in modes
                    ]
                )
                repaired_comps["monitor_times"] = monitor
                repaired_comps["time_to_detection"] = repaired_comps["monitor_times"].copy()
        # monitoring across levels, only applies if properly defined and monitoring at the component level can_monitor is false
        elif component_info.get(ck.COMP_MONITOR, None):
            monitor = component_info[ck.COMP_MONITOR]
            repaired_comps["monitor_times"] = self.sample(monitor[ck.DIST], monitor[ck.PARAM], num_repaired)
            repaired_comps["time_to_detection"] = repaired_comps["monitor_times"].copy()
        # only static detection available
        elif component_info.get(ck.STATIC_MONITOR, None):
            # the proper detection time with only static monitoring is the difference between the static monitoring that occurs after the failure
            # this will be set when a component fails for simplicity sake, since multiple static monitoring schemes can be defined,
            # and the time to detection would be the the time from the component fails to the next static monitoring occurance
            # so, set these to None and assume they will be properly updated in the simulation
            repaired_comps["monitor_times"] = None
            repaired_comps["time_to_detection"] = None

        repaired_comps["state"] = 1
        df.loc[mask] = repaired_comps
