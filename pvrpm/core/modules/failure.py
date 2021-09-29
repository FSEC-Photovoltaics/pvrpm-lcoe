from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import sample, get_higher_components
from pvrpm.core.modules.monitor import StaticMonitor


class Failure(ABC):
    """
    This abstract class defines how a failure should be set up
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        static_monitoring: StaticMonitor = None,
    ):
        """
        Initalizes a failure instance

        Args:
            level (str): The component level this failure is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
            static_monitoring (:obj:`StaticMonitoring`, Optional): For updating static monitoring during simulation
        """
        super().__init__()
        self.level = level
        self.df = comp_level_df
        self.case = case
        self.fails_per_day = {}
        self.static_monitoring = static_monitoring
        self.initialize_components()

    @abstractmethod
    def initialize_components(self):
        """
        Initalizes failure data for all components to be tracked during simulation for this failure type
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
                sample_ = np.random.random_sample(size=component_info[ck.NUM_COMPONENT])
                defective = sample_ < fail[ck.FRAC]

                sample_ = sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(list(defective), sample_, np.finfo(np.float32).max)

            elif fail.get(ck.FRAC, None) is None:
                # setup failure times for each component
                possible_failure_times[:, i] = sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

            # initalize failures per day for this failure mode
            self.fails_per_day[mode] = np.zeros(self.case.config[ck.LIFETIME_YRS] * 365)

        failure_ind = np.argmin(possible_failure_times, axis=1)
        df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        df["failure_type"] = [failure_modes[i] for i in failure_ind]

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        component_info = self.case.config[self.level]
        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())
        fraction_failures = []
        largest_frac = 0

        # print(f"reinit: {self.level}")

        num_repaired = len(df)
        possible_failure_times = np.zeros((num_repaired, len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            fail = component_info[ck.FAILURE][mode]

            if fail.get(ck.FRAC, None):
                fraction_failures.append(mode)
                if fail[ck.FRAC] > largest_frac:
                    largest_frac = fail[ck.FRAC]

                # choose a percentage of modules to be defective
                sample_ = np.random.random_sample(size=num_repaired)
                defective = sample_ < fail[ck.FRAC]

                sample_ = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(
                    list(defective),
                    sample_,
                    np.finfo(np.float32).max,
                )

            elif fail.get(ck.FRAC, None) is None:
                # setup failure times for each component
                possible_failure_times[:, i] = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

        failure_ind = np.argmin(possible_failure_times, axis=1)
        df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        df["failure_type"] = [failure_modes[i] for i in failure_ind]

        # now, need to make sure that our fractional failures percentages are met for all components in this level
        # TODO: need to speed this up somehow
        # TODO: also need to figure out why its not working for case 1
        if fraction_failures:
            # removes the diminishing effect where at the beginning of the simulation frac modules are a defective failure, then frac of frac is defective, etc.
            # possible failure times will also include whatever the current failure time is for the component, if its less then a defective one it doesn't change
            possible_failure_times = np.zeros((len(self.df), len(fraction_failures) + 1))
            possible_failure_times.fill(np.finfo(np.float32).max)
            # NOTE: i think i should just instead of doing the whole df, find the fraction, then sample that fraction from the components and just update those using the same method below
            for i, mode in enumerate(fraction_failures):
                counts = (self.df["failure_type"].astype(str) == mode).sum()
                frac = counts / len(self.df)
                fail = component_info[ck.FAILURE][mode]
                # print(f"fail frac: {frac}")

                if frac >= fail[ck.FRAC]:
                    continue
                sample_ = np.random.random_sample(size=len(self.df))
                # we just want the difference in fractions to bump it up to the failure fraction
                defective = sample_ < (fail[ck.FRAC] - frac)
                sample_ = sample(fail[ck.DIST], fail[ck.PARAM], len(self.df))

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(
                    list(defective),
                    sample_,
                    np.finfo(np.float32).max,
                )

            possible_failure_times[:, -1] = self.df["time_to_failure"]
            failure_ind = np.argmin(possible_failure_times, axis=1)
            types = []
            for comp, i in enumerate(failure_ind):
                if i != len(fraction_failures):
                    types.append(fraction_failures[i])
                else:
                    types.append(self.df["failure_type"].iloc[comp])

            self.df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
            self.df["failure_type"] = np.array(types).astype(str)

        return df

    def update(self, day: int):
        df = self.df

        # decrement time to failures for operational modules
        df.loc[df["state"] == 1, "time_to_failure"] -= 1

        failure_modes = list(self.case.config[self.level][ck.FAILURE].keys())

        mask = (df["state"] == 1) & (df["time_to_failure"] < 1)
        failed_comps = df.loc[mask].copy()

        if len(failed_comps) > 0:
            # print(f"failing {self.level}: {len(failed_comps)}")
            # print(failed_comps)
            failed_comps["time_to_failure"] = 0
            failed_comps["cumulative_failures"] += 1
            for fail in failure_modes:
                fail_mask = failed_comps["failure_type"].astype(str) == fail
                failed_comps.loc[fail_mask, f"failure_by_type_{fail}"] += 1
                self.fails_per_day[fail][day] += len(failed_comps.loc[fail_mask])

            warranty_mask = failed_comps["time_left_on_warranty"] <= 0
            failed_comps.loc[warranty_mask, "cumulative_oow_failures"] += 1

            failed_comps["state"] = 0

            # update time to detection times for component levels with only static monitoring
            # which will have None for monitor times
            try:
                if failed_comps["monitor_times"].isnull().any():
                    # monitor and time to detection will be the time to next static monitoring
                    static_monitors = list(self.case.config[self.level][ck.STATIC_MONITOR].keys())
                    # next static monitoring is the min of the possible static monitors for this component level
                    failed_comps["monitor_times"] = np.amin(self.static_monitoring.static_monitoring[static_monitors])
                    failed_comps["time_to_detection"] = failed_comps["monitor_times"].copy()
            # fails if no monitoring defined, faster then just doing a check if the column exists or whatever
            except KeyError:
                pass

            df.loc[mask] = failed_comps


class PartialFailure(Failure):
    """
    Specifies a decrease in the state of a component via a failure
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        static_monitoring: StaticMonitor = None,
    ):
        super().__init__(level, comp_level_df, case, static_monitoring=static_monitoring)
