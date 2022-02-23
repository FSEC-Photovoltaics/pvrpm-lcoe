import click

from pvrpm.core.logger import logger, init_logger
from pvrpm.core.exceptions import CaseError
from pvrpm.core.case import SamCase
from pvrpm.core.simulation import pvrpm_sim


@click.group()
def main():
    """
    Perform cost modeling for PV systems using SAM and PVRPM
    """
    pass


@main.command()
@click.argument("config")
@click.option("--case", metavar="<path>", help="Path to directory containing json export from SAM for the case")
@click.option(
    "--threads",
    metavar="<num_threads>",
    help="Number of threads to use for paralized simulations, set to -1 to use all CPU threads",
    default=1,
)
@click.option(
    "--realization",
    "--realizations",
    metavar="<num_realizations>",
    help="Set the number of realizations for this run, overrides configuration file value",
    default=0,
)
@click.option(
    "--results",
    metavar="<path/to/results/folder>",
    help="Folder to use for saving the results, overrides configuration file value",
    default=None,
)
@click.option("--trace", default=False, is_flag=True, help="Enable debug stack traces")
@click.option(
    "--debug",
    default=0,
    help="Save simulation state every specified number of days for debugging. Saved to results folder specified",
)
@click.option("--noprogress", default=False, is_flag=True, help="Disable progress bars for realizations")
def run(
    case: str, threads: int, realization: int, results: str, config: str, trace: bool, debug: int, noprogress: bool
):
    """
    Run the PVRPM LCOE cost model for the case

    The config YAML file should specify module order of simulation
    """
    try:
        sam_case = SamCase(case, config, num_realizations=int(realization), results_folder=results)
    except CaseError as e:
        logger.error(f"Error loading and verifying case: {e}")
        return

    if trace:
        pvrpm_sim(
            sam_case, save_results=True, save_graphs=True, threads=threads, debug=debug, progress_bar=not noprogress
        )
    else:
        try:
            pvrpm_sim(
                sam_case, save_results=True, save_graphs=True, threads=threads, debug=debug, progress_bar=not noprogress
            )
        except Exception as e:
            logger.error(f"There was an error in performing the simulation: {e}")


@main.command()
@click.argument("config")
@click.option("--case", metavar="<path>", help="Path to directory containing json export from SAM for the case")
@click.option("--verbose", default=0, is_flag=True, help="Enable verbosity in SAM simulation")
def sim(case: str, config: str, verbose: int):
    """
    Load the SAM case and test the basic SAM simulation

    The config YAML file should specify module order of simulation
    """

    try:
        sam_case = SamCase(case, config)
    except CaseError as e:
        logger.error(f"Error loading and verifying case: {e}")
        return

    try:
        sam_case.simulate(verbose=verbose)
    except Exception as e:
        logger.error(f"There was an error in running the simulation: {e}")


@main.command()
@click.argument("config")
@click.option("--case", metavar="<path>", help="Path to directory containing json export from SAM for the case")
def verify(case: str, config: str):
    """
    Verify the case and pvrpm configuration files
    """
    try:
        sam_case = SamCase(case, config)
        logger.info("Configuration verified successfully!")
    except CaseError as e:
        logger.error(f"Error loading and verifying case: {e}")
