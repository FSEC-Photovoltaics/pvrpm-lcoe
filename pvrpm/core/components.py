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

        lifetime = self.case.config[ck.LIFETIME_YRS]
        # every component level will contain a dataframe containing the data for all the components in that level
        for c in ck.component_keys:
            if self.case.config.get(c, None):
                self.comps[c] = self.initialize_components(c)
                self.total_repair_time[c] = 0
                self.total_monitor_time[c] = 0
                self.costs[c] = np.zeros(lifetime * 365)

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
        if component_info[ck.CAN_REPAIR] and component_info[ck.CAN_MONITOR]:
            if len(monitor_modes) == 1:
                # same monitor mode for every failure
                monitor = component_info[ck.MONITORING][monitor_modes[0]]
                monitor = self.sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
                df["monitor_times"] = pd.Series(monitor)
                df["time_to_repair"] += monitor
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
                df["monitor_times"] = pd.Series(monitor)
                df["time_to_repair"] += monitor

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

            df.loc[mask] = failed_comps

    def repair_component(self, component_level: str, day: int) -> pd.Series:
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
            fail = self.case.config[component_level][ck.FAILURE][mode]
            fail_mask = mask & (df["failure_type"] == mode)
            repair_cost = fail[ck.COST] + self.labor_rate * fail[ck.LABOR]

            if self.case.config[component_level].get(ck.WARRANTY, None):
                warranty_mask = fail_mask & (df["time_left_on_warranty"] <= 0)
                self.costs[component_level][day] += len(df.loc[warranty_mask]) * repair_cost
            else:
                self.costs[component_level][day] += len(df.loc[fail_mask]) * repair_cost

        repaired_comps = df.loc[mask].copy()

        # add up the repair and monitoring times
        self.total_repair_time[component_level] += repaired_comps["repair_times"].sum()
        if component_info[ck.CAN_MONITOR]:
            self.total_monitor_time[component_level] += repaired_comps["monitor_times"].sum()

        # reinitalize all repaired modules
        # degradation gets reset to 0 (module only)
        if component_level == ck.MODULE:
            repaired_comps["days_of_degradation"] = 0

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
                monitor = self.sample(monitor[ck.DIST], monitor[ck.PARAM], num_repaired)
                repaired_comps["monitor_times"] = pd.Series(monitor)
                repaired_comps["time_to_repair"] += monitor
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
                repaired_comps["monitor_times"] = pd.Series(monitor)
                repaired_comps["time_to_repair"] += monitor

        repaired_comps["state"] = 1
        df.loc[mask] = repaired_comps
