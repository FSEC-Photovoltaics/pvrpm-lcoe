from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase
from pvrpm.core.utils import sample, get_higher_components
from pvrpm.core.modules.monitor import IndepMonitor


class Failure(ABC):
    """
    This abstract class defines how a failure should be set up
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        indep_monitoring: IndepMonitor = None,
    ):
        """
        Initalizes a failure instance

        Args:
            level (str): The component level this failure is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
            indep_monitoring (:obj:`IndepMonitoring`, Optional): For updating static monitoring during simulation
        """
        super().__init__()
        self.level = level
        self.df = comp_level_df
        self.case = case
        self.fails_per_day = {}
        self.indep_monitoring = indep_monitoring
        self.last_failure_day = 0
        self.mean = None
        self.initialize_components()

    @abstractmethod
    def initialize_components(self):
        """
        Initalizes failure data for all components to be tracked during simulation for this failure type

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
        self.mean = {}  # init mean for each failure mode

        possible_failure_times = np.zeros((component_info[ck.NUM_COMPONENT], len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            self.mean[mode] = 0
            # initalize failure mode by type
            df[f"failure_by_type_{mode}"] = 0
            fail = component_info[ck.FAILURE][mode]
            if fail.get(ck.FRAC, None) or fail.get(ck.DECAY_FRAC, None):
                frac = fail[ck.FRAC] if ck.FRAC in fail else fail[ck.DECAY_FRAC]
                # choose a percentage of components to be defective
                sample_ = np.random.random_sample(size=component_info[ck.NUM_COMPONENT])
                defective = sample_ < frac

                sample_ = sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(list(defective), sample_, np.finfo(np.float32).max)

            else:
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

        num_repaired = len(df)
        possible_failure_times = np.zeros((num_repaired, len(failure_modes)))
        for i, mode in enumerate(failure_modes):
            fail = component_info[ck.FAILURE][mode]

            if fail.get(ck.FRAC, None) or fail.get(ck.DECAY_FRAC, None):
                frac = 0
                if fail.get(ck.FRAC, None):
                    fraction_failures.append(mode)
                    frac = fail[ck.FRAC]
                else:
                    frac = fail[ck.DECAY_FRAC]

                # choose a percentage of modules to be defective
                sample_ = np.random.random_sample(size=num_repaired)
                defective = sample_ < frac

                sample_ = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

                # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                possible_failure_times[:, i] = np.where(
                    list(defective),
                    sample_,
                    np.finfo(np.float32).max,
                )

            else:
                # setup failure times for each component
                possible_failure_times[:, i] = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

        failure_ind = np.argmin(possible_failure_times, axis=1)
        df["time_to_failure"] = np.amin(possible_failure_times, axis=1)
        df["failure_type"] = [failure_modes[i] for i in failure_ind]

        # now, need to make sure that our fractional failures percentages are met for all components in this level
        # TODO: need to speed this up somehow
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
        # TODO: change this to state > 0 once partial failures implemented
        df["time_to_failure"] -= 1

        failure_modes = list(self.case.config[self.level][ck.FAILURE].keys())

        # TODO: change this to state > 0 once partial failures implemented
        mask = (df["state"] == 1) & (df["time_to_failure"] < 1)
        failed_comps = df.loc[mask].copy()

        if len(failed_comps) > 0:
            self.last_failure_day = day
            failed_comps["time_to_failure"] = 0
            failed_comps["cumulative_failures"] += 1
            for fail in failure_modes:
                fail_mask = failed_comps["failure_type"].astype(str) == fail
                failed_comps.loc[fail_mask, f"failure_by_type_{fail}"] += 1
                self.fails_per_day[fail][day] += len(failed_comps.loc[fail_mask])

            warranty_mask = failed_comps["time_left_on_warranty"] <= 0
            failed_comps.loc[warranty_mask, "cumulative_oow_failures"] += 1

            failed_comps["state"] = 0

            # update time to detection times for component levels with only independent monitoring
            # which will have None for monitor times
            try:
                if failed_comps["monitor_times"].isnull().any():
                    # monitor and time to detection will be the time to next indep monitoring
                    indep_monitors = list(self.case.config[self.level][ck.INDEP_MONITOR].keys())
                    # next indep monitoring is the min of the possible indep monitors for this component level
                    failed_comps["monitor_times"] = np.amin(self.indep_monitoring.indep_monitoring[indep_monitors])
                    # in order to calculate the time to detection for component levels only monitoring by an
                    # independment monitoring with a threshold (no interval), need to instead
                    # set the nans that will be there to the day in the simulation when these components failed
                    # so it can be calculated later
                    failed_comps["monitor_times"] = failed_comps["monitor_times"].fillna(day)
                    failed_comps["time_to_detection"] = None  # failed_comps["monitor_times"].copy()
            # fails if no monitoring defined, faster then just doing a check if the column exists or whatever
            except KeyError:
                pass

            df.loc[mask] = failed_comps
        else:
            # check to see when last failure was for fraction failure, and update components with new failures
            # if its been longer then the mean time of the distribution
            # this is so if repairs arent occuring due to poor monitoring, failures are still occuring
            failure_modes = list(self.case.config[self.level].get(ck.FAILURE, {}).keys())
            fraction_failures = []
            for mode in failure_modes:
                fail = self.case.config[self.level][ck.FAILURE][mode]
                if fail.get(ck.FRAC, None):
                    # extract mean, since some distributions might not have mean defined in params
                    if self.mean[mode] == 0:
                        self.mean[mode] = sample(fail[ck.DIST], fail[ck.PARAM], 10000).mean()

                    if day > (self.mean[mode] + self.last_failure_day):
                        fraction_failures.append(mode)
                        self.last_failure_day = day

            for mode in fraction_failures:
                # fail new fraction of components
                # possible failure times will also include whatever the current failure time is for the component, if its less then a defective one it doesn't change
                possible_failure_times = np.zeros((len(self.df), len(fraction_failures) + 1))
                possible_failure_times.fill(np.finfo(np.float32).max)
                # NOTE: i think i should just instead of doing the whole df, find the fraction, then sample that fraction from the components and just update those using the same method below
                for i, mode in enumerate(fraction_failures):
                    fail = self.case.config[self.level][ck.FAILURE][mode]

                    sample_ = np.random.random_sample(size=len(self.df))

                    defective = sample_ < fail[ck.FRAC]
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


class PartialFailure(Failure):
    """
    Specifies a decrease in the state of a component via a failure

    Unlike total failures, every defined partial failure will have its own object, instead of manaing all of them at once
    """

    def __init__(
        self,
        level: str,
        comp_level_df: pd.DataFrame,
        case: SamCase,
        mode: str,
        indep_monitoring: IndepMonitor = None,
    ):
        """
        Initalizes a partial failure instance

        Args:
            level (str): The component level this failure is apart of
            comp_level_df (:obj:`pd.DataFrame`): The component level dataframe containing the simulation data
            case (:obj:`SamCase`): The SAM case for this simulation
            mode (str): The name of the partial failure mode
            indep_monitoring (:obj:`IndepMonitoring`, Optional): For updating static monitoring during simulation
        """
        self.mode = mode
        super().__init__(level, comp_level_df, case, indep_monitoring=indep_monitoring)

    def initialize_components(self):
        component_info = self.case.config[self.level]
        df = self.df
        mode = self.mode
        failure_times = None

        # initalize failure mode by type
        df[f"failure_by_type_{mode}"] = 0
        fail = component_info[ck.PARTIAL_FAIL][mode]
        if fail.get(ck.FRAC, None) or fail.get(ck.DECAY_FRAC, None):
            frac = fail[ck.FRAC] if ck.FRAC in fail else fail[ck.DECAY_FRAC]
            # choose a percentage of components to be defective
            sample_ = np.random.random_sample(size=component_info[ck.NUM_COMPONENT])
            defective = sample_ < frac

            sample_ = sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

            # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
            failure_times = np.where(list(defective), sample_, np.nan)

        else:
            # setup failure times for each component
            failure_times = sample(fail[ck.DIST], fail[ck.PARAM], component_info[ck.NUM_COMPONENT])

        # initalize failures per day for this failure mode
        self.fails_per_day = {self.mode: np.zeros(self.case.config[ck.LIFETIME_YRS] * 365)}

        df[f"time_to_failure_{mode}"] = failure_times

    def reinitialize_components(self, df: pd.DataFrame) -> pd.DataFrame:
        component_info = self.case.config[self.level]
        num_repaired = len(df)
        fraction_failure = False
        failure_times = None
        mode = self.mode
        fail = component_info[ck.PARTIAL_FAIL][mode]

        if fail.get(ck.FRAC, None) or fail.get(ck.DECAY_FRAC, None):
            if fail.get(ck.FRAC, None):
                fraction_failure = True
                frac = fail[ck.FRAC]
            else:
                frac = fail[ck.DECAY_FRAC]

            # choose a percentage of modules to be defective
            sample_ = np.random.random_sample(size=num_repaired)
            defective = sample_ < frac

            sample_ = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

            # only give a possible failure time if the module is defective, otherwise it is set to nan, partial failure is not applied
            failure_times = np.where(list(defective), sample_, np.nan)
        else:
            # setup failure times for each component
            failure_times = sample(fail[ck.DIST], fail[ck.PARAM], num_repaired)

        df[f"time_to_failure_{mode}"] = failure_times

        # now, need to make sure that our fractional failure percentage is met for all components in this level
        # TODO: need to speed this up somehow
        if fraction_failure:
            # removes the diminishing effect where at the beginning of the simulation frac modules are a defective failure, then frac of frac is defective, etc.
            # NOTE: i think i should just instead of doing the whole df, find the fraction, then sample that fraction from the components and just update those using the same method below
            # number currently with failure mode is going to be the number non nan time_to_failures
            counts = self.df[f"time_to_failure_{mode}"].isna()
            update_df = self.df.loc[counts].copy()

            frac = (~counts).sum() / len(self.df)
            if frac >= fail[ck.FRAC]:
                return df

            sample_ = np.random.random_sample(size=len(update_df))
            # we just want the difference in fractions to bump it up to the failure fraction
            defective = sample_ < (fail[ck.FRAC] - frac)
            sample_ = sample(fail[ck.DIST], fail[ck.PARAM], len(update_df))

            # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
            failure_times = np.where(
                list(defective),
                sample_,
                np.nan,
            )

            update_df[f"time_to_failure_{mode}"] = failure_times
            self.df.loc[counts] = update_df

        return df

    def update(self, day: int):
        df = self.df

        # decrement time to failures
        df[f"time_to_failure_{self.mode}"] -= 1

        mask = (df["state"] == 1) & (df[f"time_to_failure_{self.mode}"] < 1)
        failed_comps = df.loc[mask].copy()

        if len(failed_comps) > 0:
            self.last_failure_day = day
            failed_comps["cumulative_failures"] += 1
            failed_comps[f"failure_by_type_{self.mode}"] += 1
            self.fails_per_day[self.mode][day] += len(failed_comps)

            warranty_mask = failed_comps["time_left_on_warranty"] <= 0
            failed_comps.loc[warranty_mask, "cumulative_oow_failures"] += 1

            failed_comps["state"] = 0

            # update time to detection times for component levels with only static monitoring
            # which will have None for monitor times
            try:
                if failed_comps["monitor_times"].isnull().any():
                    # monitor and time to detection will be the time to next static monitoring
                    indep_monitors = list(self.case.config[self.level][ck.INDEP_MONITOR].keys())
                    # next static monitoring is the min of the possible static monitors for this component level
                    failed_comps["monitor_times"] = np.amin(self.indep_monitoring.indep_monitoring[indep_monitors])
                    # in order to calculate the time to detection for component levels only monitoring by an
                    # independment monitoring with a threshold (no interval), need to instead
                    # set the nans that will be there to the day in the simulation when these components failed
                    # so it can be calculated later
                    failed_comps["monitor_times"] = failed_comps["monitor_times"].fillna(day)
                    failed_comps["time_to_detection"] = None  # failed_comps["monitor_times"].copy()
            # fails if no monitoring defined, faster then just doing a check if the column exists or whatever
            except KeyError:
                pass

            df.loc[mask] = failed_comps
        else:
            # check to see when last failure was for fraction failure, and update components with new failures
            # if its been longer then the mean time of the distribution
            # this is so if repairs arent occuring due to poor monitoring, failures are still occuring
            fail = self.case.config[self.level][ck.PARTIAL_FAIL][self.mode]
            if fail.get(ck.FRAC, None):
                # extract mean, since some distributions might not have mean defined in params
                if not self.mean:
                    self.mean = sample(fail[ck.DIST], fail[ck.PARAM], 10000).mean()

                if day > (self.mean + self.last_failure_day):
                    # fail new fraction of components
                    counts = self.df[f"time_to_failure_{self.mode}"].isna()
                    update_df = self.df.loc[counts].copy()

                    sample_ = np.random.random_sample(size=len(update_df))
                    # we just want the difference in fractions to bump it up to the failure fraction
                    defective = sample_ < fail[ck.FRAC]
                    sample_ = sample(fail[ck.DIST], fail[ck.PARAM], len(update_df))

                    # only give a possible failure time if the module is defective, otherwise it is set to numpy max float value (which won't be used)
                    failure_times = np.where(
                        list(defective),
                        sample_,
                        np.nan,
                    )

                    update_df[f"time_to_failure_{self.mode}"] = failure_times
                    self.df.loc[counts] = update_df
                    self.last_failure_day = day
