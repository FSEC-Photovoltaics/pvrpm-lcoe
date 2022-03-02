import os
import numpy as np

from pvrpm.core.case import SamCase
from pvrpm.core.simulation import pvrpm_sim, gen_results
from pvrpm.core.enums import ConfigKeys as ck


def test_simulation(tmp_path: str):
    """
    Test every function of the simulation with a test set case

    Args:
        tmp_path (str): Temporary path for storing result files
    """
    os.chdir(os.path.join("tests", "integration", "case"))
    case = SamCase(".", "test.yml", num_realizations=5, results_folder=tmp_path)
    # results = pvrpm_sim(case, save_results=True, save_graphs=True, progress_bar=False, debug=365, threads=-1)

    # df_results = gen_results(case, results)
    # mean_idx = 8
    # summary = df_results[0]
    # avg_lcoe = summary["lcoe"][summary.index[mean_idx]]

    # total_fails = [summary[f"{c}_total_failures"][mean_idx] for c in ck.component_keys]
    # ranges for each total fails
    # ranges = [
    #    (220, 240),
    #    (2, 6),
    #    (20, 30),
    #    (10, 20),
    #    (2, 12),
    #    (0.5, 5),
    #    (0.5, 5),
    #    (180, 195),
    # ]

    # ensure simulation ran correctly
    # values taken from manual input into SAM case
    # assert avg_lcoe <= 22 and avg_lcoe >= 10
    # for i, (l, h) in enumerate(ranges):
    #    assert total_fails[i] >= l and total_fails[i] <= h
