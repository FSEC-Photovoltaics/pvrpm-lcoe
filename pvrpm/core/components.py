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
    Data container for each component in the simulation
    """

    def __init__(self, case: SamCase):
        self.case = case

        self.comps = {}
        self.uptime = {}
        self.costs = {}
        # every component level will contain a dataframe containing the data for all the components in that level
        for c in ck.component_keys:
            if self.case.config.get(c, None):
                self.comps[c] = self.initialize_components(c)
                self.uptime[c] = 0
                self.costs[c] = np.zeros(self.case.config[c][ck.NUM_COMPONENT])

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
            # this files for certain parameter ranges, raising a runtime error
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
            - time_to_failure (int): Number of days until component fails
            - failure_type (str): The name of the failure type time_to_failure represents
            - time_to_repair (int): Number of days from failure until compoent is repaired
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

        component_ind = [i + 1 for i in range(component_info[ck.NUM_COMPONENT])]
        df = pd.DataFrame(index=component_ind)

        failure_modes = list(component_info.get(ck.FAILURE, {}).keys())
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
            return

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
        if len(repair_modes) == 1:
            # same repair mode for every repair
            repair = component_info[ck.REPAIR][repair_modes[0]]
            df["time_to_repair"] = self.sample(repair[ck.DIST], repair[ck.PARAM], component_info[ck.NUM_COMPONENT])
        else:
            modes = [repair_modes[i] for i in failure_ind]
            df["time_to_repair"] = np.array(
                [
                    self.sample(component_info[ck.REPAIR][m][ck.DIST], component_info[ck.REPAIR][m][ck.PARAM], 1)
                    for m in modes
                ]
            )

        return df

    def fail_component(self, row: pd.Series) -> pd.Series:
        """
        Changes state of a component to failed, incrementing failures and checking warranty

        Args:
            row (:obj:`pd.Series`): The row containing all the data for the component

        Returns:
            :obj:`pd.Series`: The updated row with failures
        """
        row["state"] = 0
        row["time_to_failure"] = 0
        row["cumulative_failures"] = row["cumulative_failures"] + 1
        mode = row["failure_type"]
        row[f"failure_by_type_{mode}"] = row[f"failure_by_type_{mode}"] + 1

        if row["time_left_on_warranty"] <= 0:
            row["cumulative_oow_failures"] = row["cumulative_oow_failures"] + 1

    def update_component(self, row):
        pass
