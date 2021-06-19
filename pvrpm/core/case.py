import json
import yaml
import os
from glob import glob
from typing import Optional, Union, Any

import PySAM.PySSC as pssc
import numpy as np

from pvrpm.core.logger import logger
from pvrpm.core.exceptions import CaseError
from pvrpm.core.utils import filename_to_module
from pvrpm.core.enums import ConfigKeys as ck


class SamCase:
    """"""

    def __init__(self, sam_json_dir: str, config: str):
        """"""
        self.ssc = pssc.PySSC()
        self.config = self.__load_config(config, type="yaml")
        self.daily_tracker_coeffs = None
        self.modules = {}

        # load the case jsons and pysam module objects for them
        first_module = None
        for path in glob(os.path.join(sam_json_dir, "*.json")):
            module_name = os.path.basename(path)
            try:
                module = filename_to_module(module_name)
            except AttributeError:
                raise CaseError(f"Couldn't find module for file {module_name}!")

            if not module:
                raise CaseError(f"Couldn't find module for file {module_name}!")

            module_name = module.__name__.replace("PySAM.", "")

            if not first_module:
                first_module = module.new()
                module = first_module
            else:
                module = module.from_existing(first_module)

            case_config = self.__load_config(path, type="json")
            for k, v in case_config.items():
                if k != "number_inputs":
                    module.value(k, v)

            self.modules[module_name] = module

        if not (self.modules and self.config):
            raise CaseError("There are errors in the configuration files, see logs.")

        self.__verify_case()
        self.__verify_config()

    @staticmethod
    def __load_config(path: str, type: str = "yaml") -> dict:
        """
        Loads a configuration from a YAML or JSON file and returns the dictionary.

        Args:
            path (str): String path to the file
            type (str): One of `yaml` or `json`, specifies the file to load

        Returns:
            :obj:`dict`: the data loaded from the file
        """
        try:
            with open(path, "r") as f:
                if type.lower().strip() == "yaml" or type.lower().strip() == "yml":
                    config = yaml.full_load(f)
                elif type.lower().strip() == "json":
                    config = json.load(f)
        except json.decoder.JSONDecodeError:
            logger.error(f"Theres an error reading the JSON configuration file: {e}")
            return None
        except yaml.scanner.ScannerError as e:
            logger.error(f"Theres an error reading the YAML configuration file: {e}")
            return None
        except yaml.parser.ParserError as e:
            logger.error(f"Theres an error reading the YAML configuration file: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"The configuration file at {path} couldn't be found!")
            return None

        return config

    def __verify_config(self) -> None:
        """
        Verifies loaded YAML configuration file.
        """
        # tracking check
        if self.config[ck.TRACKING] and not self.config.get(ck.TRACKER, False):
            raise CaseError(
                "Tracking modules loaded in SAM config but no trackers defined in the PVRPM YML configuration, please make sure it is setup!"
            )

        if not self.config[ck.TRACKING] and self.config.get(ck.TRACKER, False):
            raise CaseError(
                "No tracking modules loaded by tracker defined in PVRPM configuration, please remove it if not using it."
            )

        top_level_keys = set(ck.needed_keys)
        included_keys = set(self.config.keys()) & top_level_keys
        if included_keys != top_level_keys:
            raise CaseError(
                f"Missing required configuration options in the PVRPM YML: {top_level_keys - included_keys}"
            )

        if self.config[ck.NUM_REALIZATION] < 2:
            raise CaseError("Number of realizations must be greater than or equal to 2!")

        if self.config[ck.TRACKING] and self.config[ck.NUM_TRACKERS] <= 0:
            raise CaseError("If tracking is defined the number of trackers must be greater than 0!")

        if self.config[ck.NUM_COMBINERS] <= 0:
            raise CaseError("Number of combiners must be greater than 0!")

        if self.config[ck.NUM_TRANSFORMERS] <= 0:
            raise CaseError("Number of transformers must be greater than 0!")

        # Limitations of empirical P-value calculation
        max_p = (1 - (1 / self.config[ck.NUM_REALIZATION])) * 100
        if self.config[ck.P_VALUE] > max_p:
            raise CaseError(
                f"The maximum p-value that can be calculated with {self.config[ck.NUM_REALIZATION]} is {max_p}. Please either lower your desired p-value or increase the number of realizations."
            )

        for component in ck.component_keys:
            if not self.config.get(component, None):  # for non-needed components
                continue

            missing = []
            if not self.config[component].get(ck.NAME, None):
                missing.append(ck.NAME)
            if self.config[component].get(ck.CAN_FAIL, None) is None:
                missing.append(ck.CAN_FAIL)
            if self.config[component].get(ck.CAN_REPAIR, None) is None:
                missing.append(ck.CAN_REPAIR)
            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

            if self.config[component][ck.CAN_FAIL] and not self.config[component].get(ck.FAILURE, None):
                missing.append(ck.FAILURE)
            if self.config[component][ck.CAN_REPAIR] and not self.config[component].get(ck.REPAIR, None):
                missing.append(ck.REPAIR)
            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

            if self.config[component].get(ck.WARRANTY, None) and not self.config[component][ck.WARRANTY].get(
                ck.DAYS, None
            ):
                missing.append(ck.DAYS)

            for failure, fail_config in self.config[component].get(ck.FAILURE, {}).items():
                fails = set(ck.failure_keys)
                included = fails & set(fail_config.keys())
                if included != fails:
                    missing += list(fails - included)

                if fail_config.get(ck.DIST, None) in ck.dists:
                    if ck.MEAN not in fail_config[ck.PARAM]:
                        missing.append(ck.MEAN)
                    if fail_config[ck.DIST] != ck.EXPON and ck.STD not in fail_config[ck.PARAM]:
                        missing.append(ck.STD)

            for repair, repair_config in self.config[component].get(ck.REPAIR, {}).items():
                repairs_ = set(ck.repair_keys)
                included = repairs_ & set(repair_config.keys())
                if included != repairs_:
                    missing += list(repairs_ - included)

                if repair_config.get(ck.DIST, None) in ck.dists:
                    if ck.MEAN not in repair_config[ck.PARAM]:
                        missing.append(ck.MEAN)
                    if repair_config[ck.DIST] != ck.EXPON and ck.STD not in repair_config[ck.PARAM]:
                        missing.append(ck.STD)

            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

        if self.config[ck.TRACKING] and self.config[ck.TRACKER][ck.CAN_FAIL]:
            self.precalculate_tracker_losses()

    def __verify_case(self) -> None:
        """
        Verifies loaded module configuration from SAM and also sets class variables for some information about the case.
        """
        if not self.value("system_use_lifetime_output"):
            raise CaseError("Please modify your case to use lifetime mode!")

        if self.value("en_dc_lifetime_losses") or self.value("en_ac_lifetime_losses"):
            logger.warn("Lifetime daily DC and AC losses will be overridden for this run.")

        if self.value("om_fixed") != [0]:
            logger.warn(
                "There is a non-zero value in the fixed annual O&M costs input. These will be overwritten with the new values."
            )

        if self.value("dc_degradation") != [0]:
            logger.warn(
                "Degradation is set by the PVRPM script, you have entered a non-zero degradation to the degradation input. This script will set the degradation input to zero."
            )
            self.value("dc_degradation", [0])

        self.config[ck.NUM_MODULES] = 0
        self.config[ck.NUM_STRINGS] = 0
        self.config[ck.INVERTER_SIZE] = self.value("inverter_count")
        # assume the number of modules per string is the same for each subarray
        self.config[ck.MODULES_PER_STR] = self.value("subarray1_modules_per_string")
        self.config[ck.TRACKING] = False
        self.config[ck.MULTI_SUBARRAY] = False
        for sub in range(1, 5):
            if sub == 1 or self.value(f"subarray{sub}_enable"):  # subarry 1 is always enabled
                self.config[ck.NUM_MODULES] += self.value(f"subarray{sub}_modules_per_string") * self.value(
                    f"subarray{sub}_nstrings"
                )

                self.config[ck.NUM_STRINGS] += self.value(f"subarray{sub}_nstrings")

                if self.value(f"subarray{sub}_track_mode"):
                    self.config[ck.TRACKING] = True

                if sub > 1:
                    self.config[ck.MULTI_SUBARRAY] = True

        inverter = self.value("inverter_model")
        if inverter == 0:
            self.inverter_size = self.value("inv_snl_paco")
        elif inverter == 1:
            self.inverter_size = self.value("inv_ds_paco")
        elif inverter == 2:
            self.inverter_size = self.value("inv_pd_paco")
        else:
            raise CaseError("Unknown inverter model! Should be 0, 1, or 2")

        if self.config[ck.MULTI_SUBARRAY] and self.config[ck.TRACKING]:
            raise CaseError(
                "Tracker failures may only be modeled for a system consisting of a single subarray. Exiting simulation."
            )

        if self.config[ck.TRACKING]:
            if self.value("subarray1_track_mode") == 2 or self.value("subarray1_track_mode") == 3:
                raise CaseError(
                    "This script is not configured to run with 2-axis tracking or azimuth-axis tracking systems."
                )

        # assume 1 AC disconnect per inverter
        self.config[ck.NUM_INVERTERS] = self.value("inverter_count")
        self.config[ck.NUM_DISCONNECTS] = self.config[ck.NUM_INVERTERS]

        self.config[ck.STR_PER_COMBINER] = np.floor(self.config[ck.NUM_STRINGS] / self.config[ck.NUM_COMBINERS])

        self.config[ck.INVERTER_PER_TRANS] = np.floor(self.config[ck.NUM_INVERTERS] / self.config[ck.NUM_TRANSFORMERS])

        self.config[ck.LIFETIME_YRS] = self.value("analysis_period")

    def precalculate_tracker_losses(self):
        """
        {recalculate_tracker_losses calculates an array of coefficients (one for every day of the year) that account for the "benefit" of trackers on that particular day. This is used to determine how much power is lost if a tracker fails.
        """
        if self.value("subarray1_tilt") != 0:
            raise CaseError("This script can only model tracker failures for 0 degree tilt trackers.")

        user_analysis_period = self.value("analysis_period")
        self.value("analysis_period", 1)
        self.value("en_ac_lifetime_losses", 0)
        self.value("en_dc_lifetime_losses", 0)

        self.simulate()
        timeseries_with_tracker = self.output("dc_net")

        # calculate timeseries performance without trackers for one year
        user_tracking_mode = self.value("subarray1_track_mode")
        user_azimuth = self.value("subarray1_azimuth")
        user_tilt = self.value("subarray1_tilt")
        self.value("subarray1_track_mode", 0)  # fixed tilt

        if user_azimuth > 360 or user_azimuth < 0:
            raise CaseError("Azimuth must be between 0 and 360. Please adjust the azimuth and try again.")

        if self.config[ck.WORST_TRACKER]:
            # assume worst case tracker gets stuck to north. If axis is north-south, assume gets stuck to west.
            worse_case_az = user_azimuth

            if user_azimuth < 180:
                worst_case_az -= 90
            else:
                worst_case_az += 90

            if worst_case_az < 0:
                worst_case_az += 360
            if worst_case_az >= 360:
                worst_case_az -= 360

            self.value("subarray1_azimuth", worst_case_az)
            self.value("subarray1_tilt", self.value("subarray1_rotlim"))
        else:
            # assume average case is that tracker gets stuck flat
            self.value("subarray1_tilt", 0)

        self.simulate()
        timeseries_without_tracker = self.output("dc_net")

        # calculate daily loss statistics
        timestep = len(timeseries_with_tracker) / 8760
        self.daily_tracker_coeffs = np.zeros(365)
        curr_dc = 0
        # TODO make this more efficent
        for d in range(365):
            sum_without_tracker = 0  # kWh
            sum_with_tracker = 0  # kWh
            for h in range(24):
                for t in range(timestep):
                    sum_with_tracker += timeseries_with_tracker[curr_dc] / timestep
                    sum_without_tracker += timeseries_without_tracker[curr_dc] / timestep
                    curr_dc += 1

            self.daily_tracker_coeffs[d] = sum_without_tracker / sum_with_tracker

        self.value("analysis_period", user_analysis_period)
        self.value("subarray1_track_mode", user_tracking_mode)
        self.value("subarray1_azimuth", user_azimuth)
        self.value("subarray1_tilt", user_tilt)

    def simulate(self, verbose: int = 0) -> None:
        """
        Executes simulations for all modules in this case.

        Args:
            verbose (int): 0 for no log messages, 1 for simulation log messages
        """
        for m_name in self.config[ck.MODULE_ORDER]:
            self.modules[m_name].execute(verbose)

    def value(self, name: str, value: Optional[Any] = None) -> Union[None, float, dict, list, str]:
        """
        Get or set by string name a module value, without specifying the module the variable resides in.

        If there is no value provided, the value is returned for the variable name.

        This will search the module's data and update the variable if found. If the value is not found in all of the modules, an AttributeError is raised.

        Args:
            name (str): Name of the value
            value (Any, optional): Value to set variable to

        Returns:
            Value or the variable if value is None
        """
        for m_name in self.modules.keys():
            try:
                if value:
                    return self.modules[m_name].value(name, value)
                else:
                    return self.modules[m_name].value(name)
            except:
                pass

        raise AttributeError(f"Variable {name} not found or incorrect value datatype in {list(self.modules.keys())}")

    def output(self, name: str) -> Union[None, float, dict, list, str]:
        """
        Get an output variable by string name, without specifying the module the variable resides in.

        This will search all of the module's outputs. If the value is not found in all of the modules, an AttributeError is raised.

        Args:
            name (str): Name of the output

        Returns:
            The value of the output variable
        """
        for m_name in self.modules.keys():
            try:
                return getattr(self.modules[m_name].Outputs, name)
            except AttributeError:
                pass  # in case something else should be done
            except:
                # if this happens, value was found but was not set, which in PvSAM raises an exception
                # so, return None
                return None

        raise AttributeError(f"Output variable {name} not found in {list(self.modules.keys())}")
