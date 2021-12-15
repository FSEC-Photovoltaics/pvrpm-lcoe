import numpy as np

from pvrpm.core import utils
from pvrpm.core.enums import ConfigKeys as ck


def test_load_modules():
    """
    Tests the loading of PySam modules
    """
    utils.load_pysam_modules()

    # try a few modules to make sure modules were loaded properly
    assert utils.filename_to_module("  Test_pvsamv1.json   ").__name__ == "PySAM.Pvsamv1"
    assert utils.filename_to_module("  Test_cashloan.json   ").__name__ == "PySAM.Cashloan"
    assert utils.filename_to_module("  Test_BeLpE.json   ").__name__ == "PySAM.Belpe"


def test_summarize_dc():
    """
    Tests the summarize DC energy function
    """
    input_array = np.array([1 for _ in range(365 * 25)])

    output_array = utils.summarize_dc_energy(input_array, 25)

    # since input was all ones, sum of output should be 365 * 25
    assert int(np.sum(output_array)) == 365 * 25


def test_component_degrade():
    """
    Tests the component degradation function
    """
    per_day = 0.5 / 365
    day = 365

    result = utils.component_degradation(per_day, day)

    assert result >= 0.995


def test_sampling():
    """
    Tests the sampling function for various distributions
    """
    parameters = {"mean": 365, "std": 180}
    tolerance = 5
    n = 100000

    for d in ck.dists:
        print(f"Checking distribution: {d}")
        sample = utils.sample(d, parameters, n)
        mean = int(sample.mean())
        std = int(sample.std())
        assert mean >= parameters["mean"] - tolerance and mean <= parameters["mean"] + tolerance
        # exponential doesn't set std from input, and uniform function the std is used to calculate the range of values
        # so don't need to check std
        if d != "exponential" and d != "uniform":
            assert std >= parameters["std"] - tolerance and std <= parameters["std"] + tolerance

    # try weibull using shape
    parameters["shape"] = 3
    del parameters["std"]
    sample = utils.sample("weibull", parameters, n)
    mean = int(sample.mean())
    assert mean >= parameters["mean"] - tolerance and mean <= parameters["mean"] + tolerance

    # try non-supported dist
    parameters = {"loc": 365, "scale": 180}
    sample = utils.sample("norm", parameters, n)
    mean = int(sample.mean())
    std = int(sample.std())
    assert mean >= parameters["loc"] - tolerance and mean <= parameters["loc"] + tolerance
    assert std >= parameters["scale"] - tolerance and std <= parameters["scale"] + tolerance
