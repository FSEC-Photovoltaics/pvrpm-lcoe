import json
import yaml
import os
from glob import glob

import pandas as pd
import numpy as np
import scipy

import PySAM.PySSC as pssc

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
        self.__verify_config()

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

    @staticmethod
    def __load_config(path: str, type: str = "yaml") -> dict:
        """
        Loads a configuration from a YAML or JSON file and returns the dictionary

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
        except FileNotFoundError:
            logger.error(f"The configuration file at {path} couldn't be found!")
            return None

        return config

    def __verify_config(self) -> None:
        """
        Verifies loaded YAML configuration file
        """
        pass

    def __verify_case(self) -> None:
        """
        Verifies loaded module configuration from SAM
        """
        pass

    def simulate(self, verbose: int = 0) -> None:
        """
        Executes simulations for all modules in this case

        Args:
            verbose (int): 0 for no log messages, 1 for simulation log messages
        """
        for m_name in self.config[ck.MODULE_ORDER]:
            self.modules[m_name].execute(verbose)
