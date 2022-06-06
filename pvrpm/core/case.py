import json
import yaml
import os
from glob import glob
from typing import Optional, Union, Any

import PySAM.PySSC as pssc
import numpy as np

from pvrpm.core.logger import logger
from pvrpm.core.exceptions import CaseError
from pvrpm.core.utils import filename_to_module, summarize_dc_energy, component_degradation
from pvrpm.core.enums import ConfigKeys as ck


class SamCase:
    """
    SAM Case loader, verifier, and simulation
    """

    def __init__(self, sam_json_dir: str, config: str, num_realizations: int = 0, results_folder: str = None):
        """"""
        self.ssc = pssc.PySSC()
        self.config = self.__load_config(config, type="yaml")
        self.sam_json_dir = sam_json_dir
        self.daily_tracker_coeffs = None
        self.modules = self.__load_modules()

        # will be calculated after base case simulation
        self.daylight_hours = None
        self.annual_daylight_hours = None
        self.base_lcoe = None
        self.base_npv = None
        self.base_ac_energy = None
        self.base_annual_energy = None
        self.base_dc_energy = None

        self.base_load = None
        self.base_tax_cash_flow = None
        self.base_losses = {}

        if not (self.modules and self.config):
            raise CaseError("There are errors in the configuration files, see logs.")

        # override results folder and number of realizations
        if num_realizations >= 2:
            self.config[ck.NUM_REALIZATION] = num_realizations
        if results_folder is not None:
            self.config[ck.RESULTS_FOLDER] = results_folder

        # lookup table for module order:
        self.module_orders = [
            ["Pvsamv1", "Grid", "Utilityrate5", "Cashloan"],
            ["Pvsamv1", "Grid", "Utilityrate5", "Merchantplant"],
            ["Pvsamv1", "Grid", "Utilityrate5", "Levpartflip"],
            ["Pvsamv1", "Grid", "Utilityrate5", "Equpartflip"],
            ["Belpe", "Pvsamv1", "Grid", "Utilityrate5", "Cashloan"],
            ["Pvsamv1", "Grid", "Utilityrate5", "Saleleaseback"],
            ["Pvsamv1", "Grid", "Utilityrate5", "Singleowner"],
            ["Pvsamv1", "Grid", "Utilityrate5", "HostDeveloper"],
            ["Belpe", "Pvsamv1", "Grid", "Utilityrate5", "Thirdpartyownership"],
        ]

        # lookup table for models that pvrpm cannot use
        self.bad_module_orders = [
            ["Pvsamv1", "Grid"],
            ["Pvsamv1", "Grid", "Lcoefcr"],
            ["Belpe", "Pvsamv1", "Grid", "Utilityrate5", "Thirdpartyownership"],
        ]

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
        except json.decoder.JSONDecodeError as e:
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

    def __load_modules(self) -> dict:
        """
        Loads the case modules and initalizes them with parameters define in the json

        Returns:
            :obj:`dict`: A dictionary containing the loaded and configured modules
        """
        first_module = None
        modules = {}
        for path in glob(os.path.join(self.sam_json_dir, "*.json")):
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
                    try:
                        module.value(k, v)
                    except AttributeError:
                        logger.warning(
                            f"Unknown key in module {module_name}: {k}\n Most likely you used the wrong SAM version to generate the JSONs for the case!"
                        )

            modules[module_name] = module

        return modules

    def __verify_config(self) -> None:
        """
        Verifies loaded YAML configuration file.
        """
        # helper function to check distribution parameters
        def check_params(component: str, name: str, config: dict):
            if ck.MEAN not in config[ck.PARAM]:
                raise CaseError(f"Mean parameter for {name} is missing!")

            if config[ck.DIST] == ck.WEIBULL:
                if ck.STD not in config[ck.PARAM] and ck.SHAPE not in config[ck.PARAM]:
                    raise CaseError(f"STD or SHAPE parameter for {name} is missing!")
            elif config[ck.DIST] != ck.EXPON and ck.STD not in config[ck.PARAM]:
                raise CaseError(f"STD parameter for {name} is missing!")

        # tracking check
        if self.config[ck.TRACKING] and not self.config.get(ck.TRACKER, False):
            raise CaseError(
                "Tracking modules loaded in SAM config but no trackers defined in the PVRPM YML configuration, please make sure it is setup!"
            )

        if not self.config[ck.TRACKING] and self.config.get(ck.TRACKER, False):
            logger.warning(
                "No tracking modules loaded for the SAM case, however there is a tracking configuration in the PVRPM configuration. This configuration will be ignored and no simulation with trackers will be performed."
            )

        top_level_keys = set(ck.needed_keys)
        included_keys = set(self.config.keys()) & top_level_keys
        if included_keys != top_level_keys:
            raise CaseError(
                f"Missing required configuration options in the PVRPM configuration: {top_level_keys - included_keys}"
            )

        if self.config[ck.NUM_REALIZATION] < 2:
            raise CaseError("Number of realizations must be greater than or equal to 2!")

        if self.config[ck.TRACKING] and self.config[ck.NUM_TRACKERS] <= 0:
            raise CaseError("If tracking is defined the number of trackers must be greater than 0!")

        if self.config[ck.NUM_COMBINERS] <= 0:
            raise CaseError("Number of combiners must be greater than 0!")

        # static monitoring
        if self.config.get(ck.INDEP_MONITOR, None):
            needed_keys = set(ck.indep_monitor_keys)
            for name, monitor_config in self.config[ck.INDEP_MONITOR].items():
                included_keys = set(monitor_config.keys()) & needed_keys
                unknown_keys = (
                    set(monitor_config.keys())
                    - needed_keys
                    - {ck.INTERVAL, ck.FAIL_THRESH, ck.DIST, ck.PARAM, ck.FAIL_PER_THRESH}
                )
                if included_keys != needed_keys:
                    raise CaseError(f"Independent monitoring for {name} is missing keys {needed_keys - included_keys}")
                if ck.INTERVAL not in monitor_config and (
                    ck.FAIL_THRESH not in monitor_config and ck.FAIL_PER_THRESH not in monitor_config
                ):
                    raise CaseError(
                        f"Independent monitoring for {name} is missing the interval and/or global_threshold/failure_per_threshold"
                    )

                if ck.DIST in monitor_config:
                    if not ck.PARAM in monitor_config:
                        raise CaseError(
                            f"Independent monitoring for {name} is missing the parameters for distribution."
                        )
                    if monitor_config[ck.DIST] in ck.dists:
                        check_params("", name, monitor_config)

                if unknown_keys:
                    logger.warning(f"Unknown keys in independent monitoring configuration: {unknown_keys}")
                for level in monitor_config[ck.LEVELS]:
                    if ck.INDEP_MONITOR not in self.config[level]:
                        self.config[level][ck.INDEP_MONITOR] = {}
                    if ck.INTERVAL in monitor_config:
                        self.config[level][ck.INDEP_MONITOR][name] = monitor_config[ck.INTERVAL]
                    else:
                        self.config[level][ck.INDEP_MONITOR][name] = None

        # cross level monitoring and compounding
        if self.config.get(ck.COMP_MONITOR, None):
            # parse levels in order from lowest -> highest to maintain priority on monitoring
            for component in ck.compound_levels:
                if not self.config[ck.COMP_MONITOR].get(component, None):
                    continue
                monitor_component_data = self.config[ck.COMP_MONITOR][component]

                for monitor_component, monitor_config in monitor_component_data.items():
                    needed_keys = set(ck.compound_keys)
                    included_keys = set(monitor_config.keys()) & needed_keys
                    unknown_keys = set(monitor_config.keys()) - needed_keys - {ck.FAIL_PER_THRESH} - {ck.FAIL_THRESH}
                    if included_keys != needed_keys:
                        raise CaseError(
                            f"Cross component monitoring under component {component}:{monitor_component} is missing keys {needed_keys - included_keys}"
                        )
                    # if monitor_config[ck.COMP_FUNC] not in ck.compound_funcs:
                    #    raise CaseError(
                    #        f"Compound function for {component}:{monitor_component} is not a valid function!"
                    #    )
                    if unknown_keys:
                        logger.warning(f"Unknown keys in cross level monitoring configuration: {unknown_keys}")

                    if (
                        not monitor_config.get(ck.FAIL_THRESH, None) is not None
                        and not monitor_config.get(ck.FAIL_PER_THRESH, None) is not None
                    ):
                        raise CaseError(
                            f"You must specify at least {ck.FAIL_THRESH} or {ck.FAIL_PER_THRESH} for {component}:{monitor_component}"
                        )

                    if monitor_config.get(ck.FAIL_THRESH, None) is not None and (
                        monitor_config[ck.FAIL_THRESH] < 0 or monitor_config[ck.FAIL_THRESH] > 1
                    ):
                        raise CaseError(
                            f"Global failure threshold for {component}:{monitor_component} must be between 0 and 1."
                        )

                    if monitor_config.get(ck.FAIL_PER_THRESH, None) is not None and (
                        monitor_config[ck.FAIL_PER_THRESH] < 0 or monitor_config[ck.FAIL_PER_THRESH] > 1
                    ):
                        raise CaseError(
                            f"{ck.FAIL_PER_THRESH} for {component}:{monitor_component} must be between 0 and 1."
                        )

                    if monitor_config[ck.DIST] in ck.dists:
                        check_params(component, monitor_component, monitor_config)

                    if not self.config[monitor_component].get(ck.COMP_MONITOR, None):
                        # add what level is monitoring this component level for compounding later
                        monitor_config[ck.LEVELS] = component
                        self.config[monitor_component][ck.COMP_MONITOR] = monitor_config

        for component in ck.component_keys:
            if not self.config.get(component, None):  # for non-needed components, needed ones checked already above
                continue

            missing = []
            if not self.config[component].get(ck.NAME, None):
                missing.append(ck.NAME)
            if self.config[component].get(ck.CAN_FAIL, None) is None:
                missing.append(ck.CAN_FAIL)
            if self.config[component].get(ck.CAN_REPAIR, None) is None:
                missing.append(ck.CAN_REPAIR)
            if self.config[component].get(ck.CAN_MONITOR, None) is None:
                missing.append(ck.CAN_MONITOR)
            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

            if self.config[component][ck.CAN_FAIL] and not self.config[component].get(ck.FAILURE, None):
                missing.append(ck.FAILURE)
            if self.config[component][ck.CAN_REPAIR] and not self.config[component].get(ck.REPAIR, None):
                missing.append(ck.REPAIR)
            if self.config[component][ck.CAN_MONITOR] and not self.config[component].get(ck.MONITORING, None):
                missing.append(ck.MONITORING)
            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

            if self.config[component].get(ck.WARRANTY, None) and not self.config[component][ck.WARRANTY].get(
                ck.DAYS, None
            ):
                missing.append(ck.DAYS)

            # check the number of repairs / monitoring is either 1 or equal to number of failures
            if self.config[component][ck.CAN_FAIL]:  # in case there are no failures for components that cant fail
                num_failure_modes = len(self.config[component].get(ck.FAILURE, {}))
                if self.config[component][ck.CAN_REPAIR]:
                    num_repair_modes = len(self.config[component].get(ck.REPAIR, {}))
                    if num_repair_modes != 1 and num_repair_modes != num_failure_modes:
                        raise CaseError(
                            f"Number of repairs for component '{component}' must be 1 or equal to the number of failures"
                        )
                if self.config[component][ck.CAN_MONITOR]:
                    num_monitor_modes = len(self.config[component].get(ck.MONITORING, {}))
                    if num_monitor_modes != 1 and num_monitor_modes != num_failure_modes:
                        raise CaseError(
                            f"Number of monitoring modes for component '{component}' must be 1 or equal to the number of failures"
                        )
                # check concurrent failures and repairs
                num_failure_modes = len(self.config[component].get(ck.PARTIAL_FAIL, {}))
                if self.config[component][ck.CAN_REPAIR]:
                    num_repair_modes = len(self.config[component].get(ck.PARTIAL_REPAIR, {}))
                    if num_repair_modes != 1 and num_repair_modes != num_failure_modes:
                        raise CaseError(
                            f"Number of concurrent repairs for component '{component}' must be 1 or equal to the number of concurrent failures"
                        )

            for failure, fail_config in self.config[component].get(ck.FAILURE, {}).items():
                fails = set(ck.failure_keys)
                if component == ck.INVERTER:
                    # inverters may have cost_per_watt specified instead of cost
                    fails.discard(ck.COST)

                included = fails & set(fail_config.keys())
                if included != fails:
                    missing += list(fails - included)

                unknown_keys = set(fail_config.keys()) - fails - {ck.FRAC, ck.COST, ck.COST_PER_WATT, ck.DECAY_FRAC}
                if unknown_keys:
                    logger.warning(f"Unknown keys in failure configuration {failure}: {unknown_keys}")

                keys = set(fail_config.keys())
                if ck.FRAC in keys and ck.DECAY_FRAC in keys:
                    raise CaseError(f"Must specify either `fraction` or `decay_fraction`, not both for '{component}'")

                # update cost for inverters
                if component == ck.INVERTER:
                    if fail_config.get(ck.COST, None) is None and fail_config.get(ck.COST_PER_WATT, None) is None:
                        missing.append(ck.COST)

                    if fail_config.get(ck.COST_PER_WATT, None) is not None:
                        # calculate costs based on cents/watt
                        self.config[component][ck.FAILURE][failure][ck.COST] = (
                            fail_config[ck.COST_PER_WATT] * self.config[ck.INVERTER_SIZE]
                        )

                if fail_config.get(ck.DIST, None) in ck.dists:
                    check_params(component, failure, fail_config)

            # partial failure check
            for failure, fail_config in self.config[component].get(ck.PARTIAL_FAIL, {}).items():
                fails = set(ck.partial_failure_keys)
                if component == ck.INVERTER:
                    # inverters may have cost_per_watt specified instead of cost
                    fails.discard(ck.COST)

                included = fails & set(fail_config.keys())
                if included != fails:
                    missing += list(fails - included)

                unknown_keys = set(fail_config.keys()) - fails - {ck.FRAC, ck.COST, ck.COST_PER_WATT, ck.DECAY_FRAC}
                if unknown_keys:
                    logger.warning(f"Unknown keys in concurrent failure configuration {failure}: {unknown_keys}")

                keys = set(fail_config.keys())
                if ck.FRAC in keys and ck.DECAY_FRAC in keys:
                    raise CaseError(f"Must specify either `fraction` or `decay_fraction`, not both for '{component}'")

                # update cost for inverters
                if component == ck.INVERTER:
                    if fail_config.get(ck.COST, None) is None and fail_config.get(ck.COST_PER_WATT, None) is None:
                        missing.append(ck.COST)

                    if fail_config.get(ck.COST_PER_WATT, None) is not None:
                        # calculate costs based on cents/watt
                        self.config[component][ck.PARTIAL_FAIL][failure][ck.COST] = (
                            fail_config[ck.COST_PER_WATT] * self.config[ck.INVERTER_SIZE]
                        )

                if fail_config.get(ck.DIST, None) in ck.dists:
                    check_params(component, failure, fail_config)

            for monitor, monitor_config in self.config[component].get(ck.MONITORING, {}).items():
                monitor_ = set(ck.monitoring_keys)
                included = monitor_ & set(monitor_config.keys())
                if included != monitor_:
                    missing += list(monitor_ - included)

                unknown_keys = set(monitor_config.keys()) - monitor_
                if unknown_keys:
                    logger.warning(f"Unknown keys in monitoring configuration {monitor}: {unknown_keys}")

                if monitor_config.get(ck.DIST, None) in ck.dists:
                    check_params(component, monitor, monitor_config)

            for repair, repair_config in self.config[component].get(ck.REPAIR, {}).items():
                repairs_ = set(ck.repair_keys)
                included = repairs_ & set(repair_config.keys())
                if included != repairs_:
                    missing += list(repairs_ - included)

                unknown_keys = set(repair_config.keys()) - repairs_
                if unknown_keys:
                    logger.warning(f"Unknown keys in repair configuration {repair}: {unknown_keys}")

                if repair_config.get(ck.DIST, None) in ck.dists:
                    check_params(component, repair, repair_config)

            # partial repairs
            for repair, repair_config in self.config[component].get(ck.PARTIAL_REPAIR, {}).items():
                repairs_ = set(ck.partial_repair_keys)
                included = repairs_ & set(repair_config.keys())
                if included != repairs_:
                    missing += list(repairs_ - included)

                unknown_keys = set(repair_config.keys()) - repairs_
                if unknown_keys:
                    logger.warning(f"Unknown keys in concurrent repair configuration {repair}: {unknown_keys}")

                if repair_config.get(ck.DIST, None) in ck.dists:
                    check_params(component, repair, repair_config)

            if missing:
                raise CaseError(f"Missing configurations for component '{component}': {missing}")

            if self.config[ck.STR_PER_COMBINER] * self.config[ck.NUM_COMBINERS] != self.num_strings:
                raise CaseError("There must be an integer number of strings per combiner!")

            if self.config[ck.INVERTER_PER_TRANS] * self.config[ck.NUM_TRANSFORMERS] != self.num_inverters:
                raise CaseError("There must be an integer number of inverters per transformer!")

            # add the number of each component to its configuration information
            if component == ck.MODULE:
                self.config[component][ck.NUM_COMPONENT] = int(self.num_modules)
            elif component == ck.STRING:
                self.config[component][ck.NUM_COMPONENT] = int(self.num_strings)
            elif component == ck.COMBINER:
                self.config[component][ck.NUM_COMPONENT] = self.config[ck.NUM_COMBINERS]
            elif component == ck.INVERTER:
                self.config[component][ck.NUM_COMPONENT] = int(self.num_inverters)
            elif component == ck.DISCONNECT:
                self.config[component][ck.NUM_COMPONENT] = int(self.num_disconnects)
            elif component == ck.TRANSFORMER:
                self.config[component][ck.NUM_COMPONENT] = self.config[ck.NUM_TRANSFORMERS]
            elif component == ck.GRID:
                self.config[component][ck.NUM_COMPONENT] = 1
            elif component == ck.TRACKER:
                self.config[component][ck.NUM_COMPONENT] = self.config[ck.NUM_TRACKERS]

        # make directory for results if it doesnt exist
        os.makedirs(self.config[ck.RESULTS_FOLDER], exist_ok=True)

        if self.config[ck.TRACKING] and self.config[ck.TRACKER][ck.CAN_FAIL]:
            self.precalculate_tracker_losses()

    def __verify_case(self) -> None:
        """
        Verifies loaded module configuration from SAM and also sets class variables for some information about the case.
        """
        # since we are finding order simulation now, remove set order in config for old pvrpm config files
        self.config[ck.MODULE_ORDER] = None

        # setup module order for simulation, also
        # need to check that an LCOE calculator that supports lifetime is used
        # in this case its only 1 unusable calculator and if no calculator is present
        my_modules = list(self.modules.keys())
        for module_loadout in self.module_orders:
            if len(module_loadout) != len(self.modules):
                continue
            found = True
            for module in module_loadout:
                if module not in my_modules:
                    found = False
                    break

            if found:
                self.config[ck.MODULE_ORDER] = module_loadout
                break

        if self.config[ck.MODULE_ORDER] is None:
            for module_loadout in self.bad_module_orders:
                if len(module_loadout) != len(self.modules):
                    continue

                found = True
                for module in module_loadout:
                    if module not in my_modules:
                        found = False
                        break

                if found:
                    raise CaseError(
                        "You have either selected the `LCOE Calculator (FCR Method)`, `Third Party Owner - Host` or `No Financial Model` for your financial model, which PVRPM does not support. Please select a supported financial model."
                    )

            raise CaseError(
                "You have selected an unknown financial model or are not using the `Detailed Photovoltaic Model`. Please update your model to a supported model."
            )

        if self.value("en_dc_lifetime_losses") or self.value("en_ac_lifetime_losses"):
            logger.warning("Lifetime daily DC and AC losses will be overridden for this run.")

        if self.value("om_fixed") != [0]:
            logger.warning(
                "There is a non-zero value in the fixed annual O&M costs input. These will be overwritten with the new values."
            )

        if self.value("dc_degradation") != [0]:
            logger.warning(
                "Degradation is set by the PVRPM script, you have entered a non-zero degradation to the degradation input. This script will set the degradation input to zero."
            )
            self.value("dc_degradation", [0])

        if ck.NUM_TRANSFORMERS not in self.config or self.config[ck.NUM_TRANSFORMERS] < 1:
            raise CaseError("Number of transformers must be greater than 0!")

        self.num_modules = 0
        self.num_strings = 0
        # assume the number of modules per string is the same for each subarray
        self.config[ck.MODULES_PER_STR] = int(self.value("subarray1_modules_per_string"))
        self.config[ck.TRACKING] = False
        self.config[ck.MULTI_SUBARRAY] = 0
        for sub in range(1, 5):
            if sub == 1 or self.value(f"subarray{sub}_enable"):  # subarry 1 is always enabled
                self.num_modules += self.value(f"subarray{sub}_modules_per_string") * self.value(
                    f"subarray{sub}_nstrings"
                )

                self.num_strings += self.value(f"subarray{sub}_nstrings")

                if self.value(f"subarray{sub}_track_mode"):
                    self.config[ck.TRACKING] = True

                self.config[ck.MULTI_SUBARRAY] += 1

        inverter = self.value("inverter_model")
        if inverter == 0:
            self.config[ck.INVERTER_SIZE] = self.value("inv_snl_paco")
        elif inverter == 1:
            self.config[ck.INVERTER_SIZE] = self.value("inv_ds_paco")
        elif inverter == 2:
            self.config[ck.INVERTER_SIZE] = self.value("inv_pd_paco")
        else:
            raise CaseError("Unknown inverter model! Should be 0, 1, or 2")

        if self.config[ck.MULTI_SUBARRAY] > 1 and self.config[ck.TRACKING]:
            raise CaseError(
                "Tracker failures may only be modeled for a system consisting of a single subarray. Exiting simulation."
            )

        if self.config[ck.TRACKING]:
            if self.value("subarray1_track_mode") == 2 or self.value("subarray1_track_mode") == 3:
                raise CaseError(
                    "This script is not configured to run with 2-axis tracking or azimuth-axis tracking systems."
                )

        # assume 1 AC disconnect per inverter
        self.num_inverters = self.value("inverter_count")
        self.num_disconnects = self.num_inverters

        self.config[ck.STR_PER_COMBINER] = int(np.floor(self.num_strings / self.config[ck.NUM_COMBINERS]))

        self.config[ck.COMBINER_PER_INVERTER] = int(np.floor(self.config[ck.NUM_COMBINERS] / self.num_inverters))

        self.config[ck.INVERTER_PER_TRANS] = int(np.floor(self.num_inverters / self.config[ck.NUM_TRANSFORMERS]))

        self.config[ck.LIFETIME_YRS] = int(self.value("analysis_period"))

    # for pickling
    def __getstate__(self) -> dict:
        """
        Converts the case into a dictionary for pickling
        """
        state = self.__dict__.copy()
        del state["modules"]
        del state["ssc"]

        return state

    def __setstate__(self, state: dict) -> None:
        """
        Creates the object from a dictionary
        """
        self.__dict__ = state
        self.ssc = pssc.PySSC()
        self.modules = self.__load_modules()

    def get_npv(self):
        """
        Returns the NPV for the case after a simulation has been ran, regardless of financial model used.
        """
        try:
            return self.output("npv")
        except AttributeError:
            pass

        try:
            return np.array(self.output("cf_project_return_aftertax_npv")).sum()
        except AttributeError:
            pass

        try:
            return self.output("tax_investor_aftertax_npv")
        except AttributeError:
            return None

    def precalculate_tracker_losses(self):
        """
        Precalculate_tracker_losses calculates an array of coefficients (one for every day of the year) that account for the "benefit" of trackers on that particular day. This is used to determine how much power is lost if a tracker fails.
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
            worst_case_az = user_azimuth

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

        # summarize it to daily energy
        sum_with_tracker = summarize_dc_energy(timeseries_with_tracker, 365)
        sum_without_tracker = summarize_dc_energy(timeseries_without_tracker, 365)

        # calculate daily loss statistics
        self.daily_tracker_coeffs = sum_without_tracker / sum_with_tracker

        self.value("analysis_period", user_analysis_period)
        self.value("subarray1_track_mode", user_tracking_mode)
        self.value("subarray1_azimuth", user_azimuth)
        self.value("subarray1_tilt", user_tilt)

    def base_case_sim(self) -> None:
        """
        Runs the base case simulation for this case, with no failures and optimal lifetime losses

        This also sets base case output parameters of this object
        """
        lifetime = self.config[ck.LIFETIME_YRS]

        # run the dummy base case
        self.value("en_dc_lifetime_losses", 0)
        self.value("en_ac_lifetime_losses", 0)

        self.value("om_fixed", [0])

        module_degradation_rate = self.config[ck.MODULE].get(ck.DEGRADE, 0) / 365

        degrade = [(1 - component_degradation(module_degradation_rate, i)) * 100 for i in range(lifetime * 365)]

        self.value("en_dc_lifetime_losses", 1)
        self.value("dc_lifetime_losses", degrade)

        self.simulate()

        self.base_lcoe = self.output("lcoe_real")
        self.base_npv = self.get_npv()

        # ac energy
        # remove the first element from cf_energy_net because it is always 0, representing year 0
        self.base_annual_energy = self.output("cf_energy_net")[1:]
        self.base_ac_energy = self.value("gen")
        self.base_dc_energy = summarize_dc_energy(self.output("dc_net"), lifetime)

        # other outputs from base case (for graphing)
        try:
            self.base_load = self.value("load")
        except AttributeError:
            self.base_load = None

        try:
            self.base_tax_cash_flow = self.output("cf_after_tax_cash_flow")
        except AttributeError:
            self.base_tax_cash_flow = self.output("cf_pretax_cashflow")

        for loss in ck.losses:
            try:
                self.base_losses[loss] = self.output(loss)
            except:
                self.base_losses[loss] = 0

        # calculate availability using sun hours
        # contains every hour in the year and whether is sun up, down, sunrise, sunset
        sunup = self.value("sunup")

        # 0 sun is down, 1 sun is up, 2 surnise, 3 sunset, we only considered sun up (1)
        sunup = np.array(sunup)

        # determine the frequency of the data, same as frequncy of supplied weather file
        total = len(sunup)
        if total == 8760:
            freq = 1
        else:
            freq = 0
            while total > 8760:
                freq += 1
                total /= freq

        # sometimes it gives half hourly or quater hourly data, so just pull out the hourly
        sunup = np.reshape(sunup[0::freq], (365, 24))

        # zero out every value except where the value is 1 (for sunup)
        sunup = np.where(sunup == 1, sunup, 0)
        # sum up daylight hours for each day
        self.daylight_hours = np.sum(sunup, axis=1)
        self.annual_daylight_hours = np.sum(self.daylight_hours)

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

        Note:
            Some modules have the same keys, this function will return the first key found in the module order specified in the configuration. Because of the way modules share data in PySAM, setting the value in the first module will propagate it to the other modules.
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
