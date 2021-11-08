from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import sample, get_higher_components


class Monitor(ABC):
    """
    Abstract class for monitoring types
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
    ):
        """
        Initalizes a monitoring instance

        Args:
            level (str): The component level this monitoring is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
        """
        super().__init__()
        self.level = level
        self.df = comp_level_df
        self.case = case
        self.initialize_components()

    @abstractmethod
    def initialize_components(self):
        """
        Initalizes monitoring data for all components to be tracked during simulation for this monitor type

        Note:
            Updates the underlying dataframes in place
        """
        pass

    @abstractmethod
    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reinitalize components in a dataframe similiar to the inital initalization. Used for when repairs or other things may occur

        Args:
            df (:obj:`pd.DataFrame`): The dataframe containing the components to reinitalize

        Returns:
            :obj:`pd.DataFrame`: The reinitalized components
        """
        pass

    @abstractmethod
    def update(self, day: int):
        """
        Perform a monitoring update for one day in the simulation:

        Modifies time to detection for components

        Args:
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        pass


class LevelMonitor(Monitor):
    """
    Defines monitoring at a component level
    """

    def initialize_components(self):
        component_info = self.case.config[self.level]
        df = self.df
        monitor_modes = list(component_info.get(ck.MONITORING, {}).keys())
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())

        # monitoring times, these will be added to the repair time for each component
        # basically, the time until each failure is detected

        if len(monitor_modes) == 1:
            # same monitor mode for every failure
            monitor = component_info[ck.MONITORING][monitor_modes[0]]
            monitor = sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
            df["monitor_times"] = monitor
            df["time_to_detection"] = df["monitor_times"].copy()
        elif len(monitor_modes) > 1:
            failure_ind = [failure_modes.index(m) for m in df["failure_type"]]
            modes = [monitor_modes[i] for i in failure_ind]
            monitor = np.array(
                [
                    sample(component_info[ck.MONITORING][m][ck.DIST], component_info[ck.MONITORING][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            df["monitor_times"] = monitor
            df["time_to_detection"] = df["monitor_times"].copy()

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        component_info = self.case.config[self.level]
        monitor_modes = list(component_info.get(ck.MONITORING, {}).keys())
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())

        if len(monitor_modes) == 1:
            # same monitor mode for every failure
            monitor = component_info[ck.MONITORING][monitor_modes[0]]
            df["monitor_times"] = sample(monitor[ck.DIST], monitor[ck.PARAM], len(df))
            df["time_to_detection"] = df["monitor_times"].copy()
        elif len(monitor_modes) > 1:
            failure_ind = [failure_modes.index(m) for m in df["failure_type"]]
            modes = [monitor_modes[i] for i in failure_ind]
            monitor = np.array(
                [
                    sample(component_info[ck.MONITORING][m][ck.DIST], component_info[ck.MONITORING][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            df["monitor_times"] = monitor
            df["time_to_detection"] = df["monitor_times"].copy()

        return df

    def update(self, day: int):
        df = self.df

        mask = (df["state"] == 0) & (df["time_to_detection"] > 1) & (df["indep_monitor"] == False)
        df.loc[mask, "time_to_detection"] -= 1


class CrossLevelMonitor(Monitor):
    """
    Defines monitoring across different levels
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
    ):
        """
        Initalizes a cross level monitoring instance

        Args:
            level (str): The component level this monitoring is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
        """
        super().__init__(level, comp_level_df, case)
        if self.case.config[level][ck.COMP_MONITOR].get(ck.FAIL_PER_THRESH, None) is not None:
            top_level = self.case.config[level][ck.COMP_MONITOR][ck.LEVELS]
            bins = np.zeros(len(self.df))
            num_per_top = get_higher_components(top_level, level, self.case)
            indicies = np.floor(self.df.index / num_per_top)
            self.monitor_bins = {
                "top_level": top_level,
                "indicies": indicies,
                "bins": bins,
                "num_per_top": num_per_top,
            }

    def update_bins(self, df: pd.DataFrame):
        indicies, counts, _ = get_higher_components(
            self.monitor_bins["top_level"],
            self.level,
            self.case,
            start_level_df=df,
        )
        self.monitor_bins["bins"][indicies] = counts

    def initialize_components(self):
        component_info = self.case.config[self.level]
        df = self.df

        monitor = component_info[ck.COMP_MONITOR]
        df["monitor_times"] = sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
        df["time_to_detection"] = df["monitor_times"].copy()

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        component_info = self.case.config[self.level]
        monitor = component_info[ck.COMP_MONITOR]

        df["monitor_times"] = sample(monitor[ck.DIST], monitor[ck.PARAM], len(df))
        df["time_to_detection"] = df["monitor_times"].copy()

        return df

    def update(self, day: int):
        df = self.df

        # fraction failed is number of failed components, no matter if its being repaired, etc
        mask = df["state"] == 0
        failed_df = df.loc[mask]
        num_failed = len(failed_df)
        mask = mask & (df["time_to_detection"] > 1) & (df["indep_monitor"] == False)

        conf = self.case.config[self.level][ck.COMP_MONITOR]
        global_threshold = False
        if conf.get(ck.FAIL_THRESH, None) is not None:
            frac_failed = num_failed / len(df)
            # only decrement time to detection once failure threshold is met
            if frac_failed > conf[ck.FAIL_THRESH]:
                # TODO: compound the failures... later
                df.loc[mask, "time_to_detection"] -= 1
                global_threshold = True
            elif conf.get(ck.FAIL_PER_THRESH, None) is None:
                df.loc[mask, "monitor_times"] += 1

        if conf.get(ck.FAIL_PER_THRESH, None) is not None and not global_threshold:
            # TODO: idk how to make this more efficent, if i can reduce this down to just a single iloc or loc, and try to elimate the call to get_higher_components i can bump up run time, cause right now it doubles runtime per realization
            # first, need to get the proper components of this level under the top monitor level
            self.update_bins(failed_df)

            data = self.monitor_bins
            undetected_failed = df.loc[mask].copy()
            indicies, counts, total = get_higher_components(
                data["top_level"],
                self.level,
                self.case,
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


class IndepMonitor(Monitor):
    """
    Defines independent monitoring that is component level independent
    """

    def __init__(
        self,
        case: SamCase,
        comps: dict,
        costs: dict,
        dc_availability: np.array,
    ):
        """
        Initalizes an independent monitoring instance

        Args:
            case (:obj:`SamCase`): The SAM case for this simulation
            comps (dict): The dataframes for each component level that tracks information during the simulation
            costs (dict): The dictionary for costs accrued for each component level
            dc_availability (:obj:`np.array`): The dc availability for every day in the simulation
        """
        super().__init__(None, None, case)
        self.comps = comps
        self.costs = costs
        self.labor_rate = self.case.config[ck.LABOR_RATE]
        self.dc_availability = dc_availability
        # keep track of static monitoring for every level
        index = []
        data = []
        for name, monitor_config in self.case.config.get(ck.INDEP_MONITOR, {}).items():
            index.append(name)
            if ck.INTERVAL in monitor_config:
                data.append(monitor_config[ck.INTERVAL])
            else:
                data.append(np.nan)

        if index:
            self.indep_monitoring = pd.Series(index=index, data=data)
        else:
            raise AttributeError("No static monitoring defined for any component level!")

    def initialize_components(self):
        pass

    def update_labor_rate(self, new: float):
        """
        Updates the labor rate for this repair

        Args:
            new (float): The new labor rate
        """
        self.labor_rate = new

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        df["monitor_times"] = None
        df["time_to_detection"] = None
        df["indep_monitor"] = False

        return df

    def update(self, day: int):
        # need to update time to detections for components with a distribution and need to decrement time to detection
        # this overrides other monitoring methods, if they are defined, using the indep_monitor column
        for name, monitor_config in self.case.config.get(ck.INDEP_MONITOR, {}).items():
            if ck.DIST in monitor_config:  # if no dist, time to detection is None or 0
                for level in monitor_config[ck.LEVELS]:
                    if not self.case.config[level][ck.CAN_FAIL]:
                        continue

                    df = self.comps[level]
                    mask = (df["state"] == 0) & (df["time_to_detection"] >= 1) & df["indep_monitor"]
                    df.loc[mask, "time_to_detection"] -= 1

        self.indep_monitoring -= 1
        mask = self.indep_monitoring < 1
        current_monitors = self.indep_monitoring.loc[mask].index
        for monitor in current_monitors:
            # independent monitoring method commences, instantly detecting all failed components in their specified levels
            # which have not been detected by other monitoring methods yet.
            config = self.case.config[ck.INDEP_MONITOR][monitor]
            for level in config[ck.LEVELS]:
                if not self.case.config[level][ck.CAN_FAIL]:
                    continue
                df = self.comps[level]
                mask = (
                    (df["state"] == 0)
                    & ((df["time_to_detection"] >= 1) | (df["time_to_detection"].isna()))
                    & (df["indep_monitor"] == False)
                )
                monitor_comps = df.loc[mask].copy()
                if monitor_comps["time_to_detection"].isna().sum() == 0:
                    monitor_comps["monitor_times"] -= monitor_comps["time_to_detection"]

                if ck.DIST in config:
                    monitor_comps["time_to_detection"] = sample(config[ck.DIST], config[ck.PARAM], len(monitor_comps))
                else:
                    monitor_comps["time_to_detection"] = 0

                monitor_comps["indep_monitor"] = True

                df.loc[mask] = monitor_comps
                # add cost to total costs (split evenly among all the levels for this day)
                self.costs[level][day] += (config[ck.COST] + self.labor_rate * config[ck.LABOR]) / len(
                    config[ck.LEVELS]
                )

            # reset static monitoring time
            self.indep_monitoring[monitor] = self.case.config[ck.INDEP_MONITOR][monitor][ck.INTERVAL]

        # threshold monitoring calculations
        for name, monitor_config in self.case.config.get(ck.INDEP_MONITOR, {}).items():
            # ignore threshold if the interval was reached for this monitor
            if name in current_monitors or (
                ck.FAIL_THRESH not in monitor_config and ck.FAIL_PER_THRESH not in monitor_config
            ):
                continue

            threshold_met = False
            if ck.FAIL_THRESH in monitor_config:
                # calculate dc availability and determine if indep monitoring is needed
                # this does use the previous day since this is calculated at the end of each day, but i figured
                # that wouldn't matter in the end and saves computing it again here
                # subtracting 1 also works since the inital dc availability is calculated for day 0
                # and day starts at 1
                if (1 - (self.dc_availability[day - 1])) > monitor_config[ck.FAIL_THRESH]:
                    threshold_met = True

            if not threshold_met and ck.FAIL_PER_THRESH in monitor_config:
                for level in monitor_config[ck.LEVELS]:
                    if not self.case.config[level][ck.CAN_FAIL]:
                        continue

                    df = self.comps[level]
                    mask = df["state"] == 0

                    if (mask.sum() / len(df)) > monitor_config[ck.FAIL_PER_THRESH]:
                        threshold_met = True
                        break

            if not threshold_met:
                continue

            # threshold met, run independent monitor
            # for levels with interval and threshold, you can reverse engineer the day by taking current day, adding the time to indep monitoring to it, then subtracting the monitor time for the components. then you can take the difference from current day and fail day to get the monitor time
            for level in monitor_config[ck.LEVELS]:
                if not self.case.config[level][ck.CAN_FAIL]:
                    continue

                df = self.comps[level]
                mask = (
                    (df["state"] == 0)
                    & ((df["time_to_detection"] >= 1) | (df["time_to_detection"].isna()))
                    & (df["indep_monitor"] == False)
                )
                monitor_comps = df.loc[mask].copy()

                if len(monitor_comps) == 0:
                    continue

                if monitor_comps["time_to_detection"].isna().sum() != 0:
                    if ck.INTERVAL in monitor_config:
                        monitor_comps["monitor_times"] = day - (
                            (day + self.indep_monitoring[name]) - (monitor_comps["monitor_times"])
                        )
                    else:
                        monitor_comps["monitor_times"] = day - monitor_comps["monitor_times"]
                else:
                    monitor_comps["monitor_times"] -= monitor_comps["time_to_detection"]

                if ck.DIST in monitor_config:
                    monitor_comps["time_to_detection"] = sample(
                        monitor_config[ck.DIST], monitor_config[ck.PARAM], len(monitor_comps)
                    )
                else:
                    monitor_comps["time_to_detection"] = 0

                monitor_comps["indep_monitor"] = True

                df.loc[mask] = monitor_comps
                # add cost to total costs (split evenly among all the levels for this day)
                self.costs[level][day] += (monitor_config[ck.COST] + self.labor_rate * monitor_config[ck.LABOR]) / len(
                    monitor_config[ck.LEVELS]
                )
