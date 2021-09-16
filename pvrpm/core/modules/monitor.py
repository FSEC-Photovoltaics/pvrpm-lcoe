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
        """
        pass

    @abstractmethod
    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reinitalize components in a dataframe similiar to the inital initalization. Used for when repairs or other things may occur
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

        # monitoring times, these will be added to the repair time for each component
        # basically, the time until each failure is detected

        if len(monitor_modes) == 1:
            # same monitor mode for every failure
            monitor = component_info[ck.MONITORING][monitor_modes[0]]
            monitor = sample(monitor[ck.DIST], monitor[ck.PARAM], component_info[ck.NUM_COMPONENT])
            df["monitor_times"] = monitor
            df["time_to_detection"] = df["monitor_times"].copy()
        elif len(monitor_modes) > 1:
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

        if len(monitor_modes) == 1:
            # same monitor mode for every failure
            monitor = component_info[ck.MONITORING][monitor_modes[0]]
            df["monitor_times"] = sample(monitor[ck.DIST], monitor[ck.PARAM], len(df))
            df["time_to_detection"] = df["monitor_times"].copy()
        elif len(monitor_modes) > 0:
            failure_ind = [repair_modes.index(m) for m in df["failure_type"]]
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

        mask = (df["state"] == 0) & (df["time_to_detection"] > 1)
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
        mask = mask & (df["time_to_detection"] > 1)

        conf = self.case.config[self.level][ck.COMP_MONITOR]
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


class StaticMonitor(Monitor):
    """
    Defines static monitoring that is component level independent
    """

    def __init__(
        self,
        case: SamCase,
        comps: dict,
        costs: dict,
    ):
        super().__init__(None, None, case)
        self.comps = comps
        self.costs = costs

        # keep track of static monitoring for every level
        index = []
        data = []
        for name, monitor_config in self.case.config.get(ck.STATIC_MONITOR, {}).items():
            index.append(name)
            data.append(monitor_config[ck.INTERVAL])

        if index:
            self.static_monitoring = pd.Series(index=index, data=data)
        else:
            raise AttributeError("No static monitoring defined for any component level!")

    def initialize_components(self):
        # nothing in the dataframe needs to be initialized
        pass

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        df["monitor_times"] = None
        df["time_to_detection"] = None

        return df

    def update(self, day: int):
        self.static_monitoring -= 1
        mask = self.static_monitoring < 1

        for monitor in self.static_monitoring.loc[mask].index:
            # static monitoring method commences, instantly detecting all failed components in their specified levels
            # which have not been detected by other monitoring methods yet.
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
