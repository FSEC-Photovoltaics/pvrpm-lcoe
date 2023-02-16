import os
import numpy as np
import json
from glob import glob

from pvrpm.core.case import SamCase
from pvrpm.core.simulation import pvrpm_sim, gen_results
from pvrpm.core.enums import ConfigKeys as ck

# first range is for LCOE, rest is for failures of every level
lcoe_ranges = {
    "commercial": (8, 15),
    "merchant-plant": (145, 160),
    "ppa-partnership": (102, 107),
    "ppa-partnership-debt": (232, 237),
    "ppa-sale-leaseback": (102, 107),
    "ppa-single": (150, 155),
    "residential": (20, 25),
    "third-party-host-dev": (147, 153),
}


fail_ranges = [
    (220, 240),
    (2, 6),
    (30, 40),
    (10, 20),
    (2, 12),
    (0.5, 5),
    (0.5, 5),
    (180, 195),
]


def test_simulation(tmp_path: str):
    """
    Test every function of the simulation with a test set case

    Args:
        tmp_path (str): Temporary path for storing result files
    """
    os.chdir(os.path.join("tests", "integration", "case"))
    for case_name, lcoe_range in lcoe_ranges.items():
        os.chdir(case_name)
        # make sure weather file is correct:
        for p in glob("*.json"):
            with open(p, "r") as f:
                data = json.load(f)

            if "solar_resource_file" in data:
                data["solar_resource_file"] = "weather_file.csv"

                with open(p, "w") as f:
                    json.dump(data, f)

                break

        case = SamCase(
            ".",
            os.path.join("..", "test.yml"),
            num_realizations=5,
            results_folder=tmp_path,
        )
        results = pvrpm_sim(
            case,
            save_results=True,
            save_graphs=True,
            progress_bar=False,
            debug=365,
            threads=-1,
        )

        df_results = gen_results(case, results)
        mean_idx = 8
        summary = df_results[0]
        avg_lcoe = summary["lcoe_real"][summary.index[mean_idx]]

        total_fails = [summary[f"{c}_total_failures"][mean_idx] for c in ck.component_keys]

        # ensure simulation ran correctly
        # values taken from manual input into SAM case

        assert avg_lcoe <= lcoe_range[1] and avg_lcoe >= lcoe_range[0]
        for i, (l, h) in enumerate(fail_ranges):
            assert total_fails[i] >= l and total_fails[i] <= h

        os.chdir("..")
