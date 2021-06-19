from typing import Any
import pkgutil
import importlib

# override to getattr to get modules case insensitve
def getattr_override(obj: Any, attr: str) -> Any:
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


# TODO: there has to be a better way to do this...
def load_pysam_modules():
    """
    Loads ALL of PySAM's modules manually and globalizes them

    This is needed because PySAM is a wrapper for the ssc and sdk of SAM, which includes dynamic modules that are not properly defined for pybind, so using pkgutil's walk_packages function does not work (import error). Since the modules need to be loaded in order for getattr to find it, this must be done once when the program starts
    """
    global pysam
    import PySAM as pysam

    for loader, module_name, is_pkg in pkgutil.walk_packages(pysam.__path__):
        try:
            importlib.import_module(f"{pysam.__name__}.{module_name}")
        except:
            pass


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
