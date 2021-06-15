import click
import sys

from pvrpm.core.logger import logger, init_logger
from pvrpm.core.exceptions import CaseError
from pvrpm.core.case import SamCase

init_logger()


@click.group()
def main():
    """
    Perform cost modeling for PV systems using SAM and PVRPM
    """
    pass


@main.command()
@click.argument("config")
@click.option("--case", metavar="<path>", help="Path to directory containing json export from SAM for the case")
@click.option("--verbose", default=0, is_flag=True, help="Enable verbosity in SAM simulation")
def sim(case: str, config: str, verbose: int):
    """
    Load the SAM case and run the basic SAM simulation

    The config YAML file should specify module order of simulation
    """

    try:
        sam_case = SamCase(case, config)
    except CaseError as e:
        logger.error(f"Error loading and verifying case: {e}")

    try:
        sam_case.simulate(verbose=verbose)
    except Exception as e:
        logger.error(f"There was an error in running the simulation: {e}")
