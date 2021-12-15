from typing import Tuple

import numpy as np
import pandas as pd


from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import summarize_dc_energy
from pvrpm.core.logger import logger
from pvrpm.core.modules import failure, monitor, repair


class Components:
    """
    Data container for each component in the simulation, as well as component and simulation level data
    """

    def __init__(self, case: SamCase):
        self.case = case

        self.comps = {}
        self.fails = {c: [] for c in ck.component_keys if self.case.config.get(c, None)}
        self.monitors = {c: [] for c in ck.component_keys if self.case.config.get(c, None)}
        self.repairs = {c: [] for c in ck.component_keys if self.case.config.get(c, None)}
        self.costs = {}

        # keep track of total days spent on monitoring and repairs
        self.total_repair_time = {}
        self.total_monitor_time = {}

        lifetime = self.case.config[ck.LIFETIME_YRS]

        # additional aggreate data to track during simulation
        self.module_degradation_factor = np.zeros(lifetime * 365)
        self.dc_power_availability = np.zeros(lifetime * 365)
        self.ac_power_availability = np.zeros(lifetime * 365)

        # static monitoring setup
        # if theres no static monitoring defined, an exception is raised
        try:
            self.indep_monitor = monitor.IndepMonitor(self.case, self.comps, self.costs, self.dc_power_availability)
        except AttributeError:
            self.indep_monitor = None

        # every component level will contain a dataframe containing the data for all the components in that level
        for c in ck.component_keys:
            if self.case.config.get(c, None):
                self.total_repair_time[c] = 0
                self.total_monitor_time[c] = 0
                self.costs[c] = np.zeros(lifetime * 365)
                df, fails, monitors, repairs = self.initialize_components(c)
                self.comps[c] = df
                self.fails[c] += fails
                self.monitors[c] += monitors
                self.repairs[c] += repairs

        # Data from simulation at end of realization
        self.timeseries_dc_power = None
        self.timeseries_ac_power = None
        self.lcoe = None
        self.npv = None
        self.annual_energy = None

        self.tax_cash_flow = None
        self.losses = {}

        if case.config[ck.TRACKING]:
            self.tracker_power_loss_factor = np.zeros(lifetime * 365)
            self.tracker_availability = np.zeros(lifetime * 365)

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

    def summarize_failures(self, component_level: str):
        """
        Returns the number of failures per day for every failure defined

        Args:
            component_level (str): The configuration key for this component level

        Returns:
            :obj:`dict`: Dictionary containing the failure mode mapped to an np array of fails per each day
        """
        fails = {}
        for f in self.fails[component_level]:
            fails.update(f.fails_per_day)

        return fails

    def update_labor_rates(self, new_labor: float):
        """
        Update labor rates for a all levels for all types of repairs

        Args:
            new_labor (float): The new labor rate
        """
        if self.indep_monitor:
            self.indep_monitor.update_labor_rate(new_labor)
        for c in ck.component_keys:
            if self.case.config.get(c, None):
                for r in self.repairs[c]:
                    r.update_labor_rate(new_labor)

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

        component_ind = np.arange(component_info[ck.NUM_COMPONENT])
        df = pd.DataFrame(index=component_ind)

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
            return (df, [], [], [])

        if component_info.get(ck.FAILURE, None):
            fails = [failure.TotalFailure(component_level, df, self.case, self.indep_monitor)]
        else:
            fails = []

        partial_failures = component_info.get(ck.PARTIAL_FAIL, {})
        partial_fails = []
        for mode in partial_failures.keys():
            partial_fails.append(failure.PartialFailure(component_level, df, self.case, mode, self.indep_monitor))

        # monitoring times, these will be added to the repair time for each component
        # basically, the time until each failure is detected
        monitors = []
        # for independent monitoring, may not be used if none is defined
        df["indep_monitor"] = False
        if component_info[ck.CAN_REPAIR]:
            if component_info[ck.CAN_MONITOR]:
                monitors.append(monitor.LevelMonitor(component_level, df, self.case))
            # monitoring across levels, only applies if properly defined and monitoring at the component level can_monitor is false
            elif component_info.get(ck.COMP_MONITOR, None):
                monitors.append(monitor.CrossLevelMonitor(component_level, df, self.case))
            # only static detection available
            elif component_info.get(ck.INDEP_MONITOR, None):
                # the proper detection time with only static monitoring is the difference between the static monitoring that occurs after the failure
                # this will be set when a component fails for simplicity sake, since multiple static monitoring schemes can be defined,
                # and the time to detection would be the the time from the component fails to the next static monitoring occurance
                # so, set these to None and assume they will be properly updated in the simulation
                df["monitor_times"] = None
                df["time_to_detection"] = None

        # time to replacement/repair in case of failure
        if not component_info[ck.CAN_REPAIR]:
            repairs = []
            df["time_to_repair"] = 1  # just initalize to 1 if no repair modes, means components cannot be repaired
        elif component_info.get(ck.REPAIR, None):
            repairs = []
            repairs.append(
                repair.TotalRepair(
                    component_level,
                    df,
                    self.case,
                    self.costs[component_level],
                    fails,
                    repairs,
                    monitors,
                    self.indep_monitor,
                )
            )
        else:
            repairs = []
            df["time_to_repair"] = 1

        partial_repairs = component_info.get(ck.PARTIAL_REPAIR, {})
        if len(partial_repairs) == 1:
            repair_mode = list(component_info[ck.PARTIAL_REPAIR].keys())[0]
            for i, fail_mode in enumerate(partial_failures.keys()):
                repairs.append(
                    repair.PartialRepair(
                        component_level,
                        df,
                        self.case,
                        self.costs[component_level],
                        partial_fails[i],
                        fail_mode,
                        repair_mode,
                        self.indep_monitor,
                    )
                )
        else:
            for i, (fail_mode, repair_mode) in enumerate(zip(partial_failures.keys(), partial_repairs.keys())):
                repairs.append(
                    repair.PartialRepair(
                        component_level,
                        df,
                        self.case,
                        self.costs[component_level],
                        partial_fails[i],
                        fail_mode,
                        repair_mode,
                        self.indep_monitor,
                    )
                )

        fails += partial_fails

        return (df, fails, monitors, repairs)

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
            adjusted_factor = min(
                1, self.case.daily_tracker_coeffs[day] + fraction * (1 - self.case.daily_tracker_coeffs[day])
            )

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

    def update_fails(self, component_level: str, day: int):
        """
        Changes state of a component to failed, incrementing failures and checking warranty only for failed components of each failure type

        Args:
            component_level (str): The component level to check for failures
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        for f in self.fails[component_level]:
            f.update(day)

    def update_repairs(self, component_level: str, day: int):
        """
        Changes the state of a component to operational once repairs are complete, only for components where the time to repair is zero

        Args:
            component_level (str): The component level of this repair
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        component_info = self.case.config[component_level]

        for r in self.repairs[component_level]:
            monitor_time, repair_time = r.update(day)
            self.total_monitor_time[component_level] += monitor_time
            self.total_repair_time[component_level] += repair_time

        # only reinitalize monitoring if the components repaired are fully availabile (state == 1)
        # this is so that if parital failures occur while the component is already detected as failed for those partial failures, then the new partial failures occuring are instantly detected until all failures are repaired
        df = self.comps[component_level]
        if self.case.config[component_level].get(ck.PARTIAL_FAIL, None) and "time_to_detection" in df:
            mask = (df["state"] == 1) & (df["time_to_detection"] < 1)
            for mode, fail_config in self.case.config[component_level][ck.PARTIAL_FAIL].items():
                mask &= df[f"time_to_failure_{mode}"] >= 1

            repaired_comps = df.loc[mask].copy()
            if len(repaired_comps) < 1:
                return

            if self.indep_monitor:
                repaired_comps = self.indep_monitor.reinitialize_components(repaired_comps)

            for m in self.monitors[component_level]:
                repaired_comps = m.reinitialize_components(repaired_comps)
            if (
                component_info[ck.CAN_MONITOR]
                or component_info.get(ck.COMP_MONITOR, None)
                or component_info.get(ck.INDEP_MONITOR, None)
            ):
                self.total_monitor_time[component_level] += repaired_comps["monitor_times"].sum()

            df.loc[mask] = repaired_comps

    def update_monitor(self, component_level: str, day: int):
        """
        Updates time to detection from component level, static, and cross component level monitoring based on number of failures

        Monitoring defined for each level is unaffected, only cross level monitoring on levels with no monitoring at that level have times updated based on failures, since monitoring defined at the component level uses the defined monitoring distribution for those components instead of the cross level monitoring

        Args:
            component_level (str): The component level to check for monitoring
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """

        for m in self.monitors[component_level]:
            m.update(day)

    def update_indep_monitor(self, day: int):
        """
        If independent monitoring is defined, check it for the current day in simulation

        Args:
            day (int): Current day in the simulation
        """
        if self.indep_monitor:
            self.indep_monitor.update(day)

    def snapshot(self):
        """
        Returns the current state of the simulation including all internal dataframes, arrays, and variables for this object

        Returns:
            :obj:`dict`: A dictionary containing the simulation snapshot data

        Note:
            The returned objects are copies of this components objects to avoid changing data in the simulation unintentionally

            The returned dictionary has these keys and values:
                * module: DataFrame of simulation data for module level
                * string: DataFrame of simulation data for string level
                * combiner: DataFrame of simulation data for combiner level
                * inverter: DataFrame of simulation data for inverter level
                * disconnect: DataFrame of simulation data for disconnect level
                * transformer: DataFrame of simulation data for transformer level
                * grid: DataFrame of simulation data for grid level
                * tracker: DataFrame of simulation data for tracker level
                * misc_data: DataFrame containing costs (for each level), module degradation, dc and ac availability, and tracker loss/availability if tracking is used
        """
        costs = [
            "Module Costs",
            "String Costs",
            "Combiner Costs",
            "Inverter Costs",
            "Disconnect Costs",
            "Transformer Costs",
            "Grid Costs",
        ]

        others = [
            "Module Degradation",
            "Dc Availability",
            "Ac Availability",
        ]

        if self.case.config[ck.TRACKING]:
            costs += ["Tracker Costs"]
            others += ["Tracker Loss", "Tracker Availabilty"]

        misc = pd.DataFrame(columns=costs.extend(others))
        misc.index = misc.index.rename("Day")

        for column, cost in zip(costs, self.costs.values()):
            misc[column] = cost.copy()

        misc[others[0]] = self.module_degradation_factor.copy()
        misc[others[1]] = self.dc_power_availability.copy()
        misc[others[2]] = self.ac_power_availability.copy()

        if self.case.config[ck.TRACKING]:
            misc[others[3]] = self.tracker_power_loss_factor.copy()
            misc[others[4]] = self.tracker_availability.copy()

        d = {}
        for comp, data in self.comps.items():
            d[comp] = data.copy()

        d["misc_data"] = misc

        return d
