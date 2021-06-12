from typing import Any

import PySAM as pysam


# override to getattr to get modules case insensitve
def getattr_override(obj: Any, attr: str) -> Any:
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def filename_to_module(filename: str):
    """
    Takes the filename of an exported json file from SAM, extracts the module name, and returns a callback to that module that can be used to create an object

    Args:
        filename (str): Filename of the exported case

    Returns:
        :obj:`PySAM`: PySAM object the file represents
    """
    # SAM case file exporting should be underscores, with the last word being the module type
    module_str = filename.strip().split("_")[-1].split(".")[0].strip()

    return getattr_override(pysam, module_str)
