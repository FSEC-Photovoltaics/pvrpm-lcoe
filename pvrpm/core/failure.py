from abc import ABC

import numpy as np
import pandas as pd

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.components import Components


class Failure(ABC):
    """
    This abstract class defines how a failure should be set up
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        static_monitoring: pd.Series = None,
        monitor_bins: dict = None,
    ):
        """
        Initalizes a failure instance

        Args:
            level (str): The component level this failure is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
            static_monitoring (:obj:`pd.Series`, Optional): For updating static monitoring during simulation
            monitor_bins (dict, Optional): For tracking the failures per each component for cross level monitoring
        """
        super().__init__()
        self.level = level
        self.df = comp_level_df
        self.case = case
        self.fails_per_day = {}
        self.static_monitoring = static_monitoring
        self.monitor_bins = monitor_bins
        self.initialize_components()

    @abstractmethod
    def initialize_components(self):
        """
        Initalizes failure data for all components to be tracked during simulation for this failure type
        """
        pass

    @abstractmethod
    def update(self, day: int):
        """
        Perform a failure update for one day in the simulation:

        Changes state of a component to failed, incrementing failures and checking warranty only for failed components of each failure type

        Args:
            day (int): Current day in the simulation

        Note:
            Updates the underlying dataframes in place
        """
        pass


class TotalFailure(Failure):
    """
    Describes how a total failure of a component should operate
    """

    def initialize_components(self):
        component_info = self.case.config[self.level]
        df = self.df
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())

        possible_failure_times = np.zeros((component_info[ck.NUM_COMPONENT], len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            # initalize failure mode by type
            df[f"failure_by_type_{mode}"] = 0
            fail = component_info[ck.FAILURE][mode]
            if fail.get(ck.FRAC, None):
                # choose a percentage of components to be defective
                sample = np.random.random_sample(size=component_info[ck.NUM_COMPONENT])
                df["defective"] = sample < fail[ck.FRAC]

                sample = Components.sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(list(df["defective"]), sample, np.finfo(np.float32).max)

            elif fail.get(ck.FRAC, None) is None:
                # setup failure times for each component
                possible_failure_times[:, i] = Components.sample(
                    fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT]
                )

            # initalize failures per day for this failure mode
            self.fails_per_day[mode] = np.zeros(self.case.config[ck.LIFETIME_YRS] * 365)

        failure_ind = np.argmin(possible_failure_times, axis=1)
        df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        df["failure_type"] = [failure_modes[i] for i in failure_ind]

    def update(self, day: int):
        failure_modes = list(self.case.config[self.level][ck.FAILURE].keys())
        df = self.df

        mask = (df["state"] == 1) & (df["time_to_failure"] < 1)
        failed_comps = df.loc[mask].copy()

        if len(failed_comps) > 0:
            failed_comps["time_to_failure"] = 0
            failed_comps["cumulative_failures"] += 1
            for fail in failure_modes:
                fail_mask = failed_comps["failure_type"] == fail
                failed_comps.loc[fail_mask, f"failure_by_type_{fail}"] += 1
                self.fails_per_day[fail][day] += len(failed_comps.loc[fail_mask])

            warranty_mask = failed_comps["time_left_on_warranty"] <= 0
            failed_comps.loc[warranty_mask, "cumulative_oow_failures"] += 1

            failed_comps["state"] = 0

            # update bins for monitoring per component for cross component monitoring
            if self.monitor_bins:
                indicies, counts, _ = Components.get_higher_components(
                    self.monitor_bins["top_level"],
                    self.level,
                    self.case,
                    start_level_df=failed_comps,
                )
                self.monitor_bins["bins"][indicies] += counts

            # update time to detection times for component levels with only static monitoring
            # which will have None for monitor times
            try:
                if failed_comps["monitor_times"].isnull().any():
                    # monitor and time to detection will be the time to next static monitoring
                    static_monitors = list(self.case.config[self.level][ck.STATIC_MONITOR].keys())
                    # next static monitoring is the min of the possible static monitors for this component level
                    failed_comps["monitor_times"] = np.amin(self.static_monitoring[static_monitors])
                    failed_comps["time_to_detection"] = failed_comps["monitor_times"].copy()
            # fails if no monitoring defined, faster then just doing a check if the column exists or whatever
            except KeyError:
                pass

            df.loc[mask] = failed_comps
