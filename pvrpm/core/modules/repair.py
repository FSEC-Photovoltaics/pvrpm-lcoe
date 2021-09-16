from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import sample, get_higher_components
from pvrpm.core.modules.monitor import StaticMonitor


class Repair(ABC):
    """
    Abstract class defining how repairs work
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        costs: np.array,
        fails: List,
        monitors: List,
        static_monitoring: StaticMonitor = None,
    ):
        """
        Initalizes a repair instance

        Args:
            level (str): The component level this repair is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
        """
        super().__init__()
        self.level = level
        self.df = comp_level_df
        self.case = case
        self.costs = costs
        self.fails = fails
        self.monitors = monitors
        self.static_monitoring = static_monitoring
        self.labor_rate = self.case.config[ck.LABOR_RATE]
        self.initialize_components()

    @abstractmethod
    def initialize_components(self):
        """
        Initalizes repair data for all components to be tracked during simulation for this repair type
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
        Perform a repair update for one day in the simulation

        Args:
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        pass

    def update_labor_rate(self, new: float):
        """
        Updates the labor rate for this repair

        Args:
            new (float): The new labor rate
        """
        self.labor_rate = new


class TotalRepair(Repair):
    """
    Defines a normal and complete repair of failed components
    """

    def initialize_components(self):
        component_info = self.case.config[self.level]
        repair_modes = list(component_info.get(ck.REPAIR, {}).keys())
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())
        df = self.df

        if len(repair_modes) == 1:
            # same repair mode for every repair
            repair = component_info[ck.REPAIR][repair_modes[0]]
            df["time_to_repair"] = sample(repair[ck.DIST], repair[ck.PARAM], component_info[ck.NUM_COMPONENT])
            df["repair_times"] = df["time_to_repair"].copy()
        else:
            failure_ind = [failure_modes.index(m) for m in df["failure_type"]]
            modes = [repair_modes[i] for i in failure_ind]
            df["time_to_repair"] = np.array(
                [
                    sample(component_info[ck.REPAIR][m][ck.DIST], component_info[ck.REPAIR][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            df["repair_times"] = df["time_to_repair"].copy()

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        component_info = self.case.config[self.level]
        repair_modes = list(component_info.get(ck.REPAIR, {}).keys())
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())

        # time to replacement/repair in case of failure
        if len(repair_modes) == 1:
            # same repair mode for every repair
            repair = component_info[ck.REPAIR][repair_modes[0]]
            df["time_to_repair"] = sample(repair[ck.DIST], repair[ck.PARAM], len(df))
            df["repair_times"] = df["time_to_repair"].copy()
        else:
            failure_ind = [failure_modes.index(m) for m in df["failure_type"]]
            modes = [repair_modes[i] for i in failure_ind]
            df["time_to_repair"] = np.array(
                [
                    sample(component_info[ck.REPAIR][m][ck.DIST], component_info[ck.REPAIR][m][ck.PARAM], 1)[0]
                    for m in modes
                ]
            )
            df["repair_times"] = df["time_to_repair"].copy()

        return df

    def update(self, day: int):
        """
        Changes the state of a component to operational once repairs are complete, only for components where the time to repair is zero
        """
        df = self.df

        # decrement time to repair for failed and detected modules
        if "time_to_detection" in df:
            mask = (df["state"] == 0) & (df["time_to_detection"] < 1)
        else:
            mask = df["state"] == 0

        df.loc[mask, "time_to_repair"] -= 1

        component_info = self.case.config[self.level]
        total_repair_time = 0
        total_monitor_time = 0

        mask = (df["state"] == 0) & (df["time_to_repair"] < 1)
        failure_modes = list(component_info[ck.FAILURE].keys())

        if len(df.loc[mask]) <= 0:
            return (total_monitor_time, total_repair_time)

        # add costs for each failure mode
        for mode in failure_modes:
            fail = component_info[ck.FAILURE][mode]
            fail_mask = mask & (df["failure_type"] == mode)
            repair_cost = fail[ck.COST] + self.labor_rate * fail[ck.LABOR]

            if component_info.get(ck.WARRANTY, None):
                warranty_mask = fail_mask & (df["time_left_on_warranty"] <= 0)
                self.costs[day] += len(df.loc[warranty_mask]) * repair_cost
            else:
                self.costs[day] += len(df.loc[fail_mask]) * repair_cost

        repaired_comps = df.loc[mask].copy()

        # add up the repair and monitoring times
        total_repair_time += repaired_comps["repair_times"].sum()
        if (
            component_info[ck.CAN_MONITOR]
            or component_info.get(ck.COMP_MONITOR, None)
            or component_info.get(ck.STATIC_MONITOR, None)
        ):
            total_monitor_time += repaired_comps["monitor_times"].sum()

        # reinitalize all repaired modules
        # degradation gets reset to 0 (module only)
        if self.level == ck.MODULE:
            repaired_comps["days_of_degradation"] = 0

        # components replaced that are out of warranty have their warranty renewed (if applicable)
        if component_info.get(ck.WARRANTY, None):
            warranty_mask = repaired_comps["time_left_on_warranty"] <= 0
            repaired_comps.loc[warranty_mask, "time_left_on_warranty"] = component_info[ck.WARRANTY][ck.DAYS]

        # reinitalize the components
        for f in self.fails:
            repaired_comps = f.reinitialize_components(repaired_comps)

        repaired_comps = self.reinitialize_components(repaired_comps)

        if self.static_monitoring:
            repaired_comps = self.static_monitoring.reinitialize_components(repaired_comps)

        for m in self.monitors:
            repaired_comps = m.reinitialize_components(repaired_comps)

        repaired_comps["state"] = 1
        df.loc[mask] = repaired_comps

        return (total_monitor_time, total_repair_time)
