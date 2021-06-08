import click
import sys

from pvrpm.core.logger import logger, init_logger

init_logger()


@click.group()
def main():
    """
    Perform cost modeling for PV systems using SAM and PVRPM
    """
    pass
