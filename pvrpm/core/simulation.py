import os

import pandas as pd
import numpy as np
import scipy

from pvrpm.core.enums import ConfigKeys as ck
from pvrpm.core.case import SamCase


def pvrpm_sim(case: SamCase):
    """
    Run the PVRPM simulation on a specific case. Results will be saved to the folder specified in the configuration.

    Args:
        case (:obj:`SamCase`): The loaded and verified case to use with the simulation
    """
    pass
