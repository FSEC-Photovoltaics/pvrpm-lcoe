from typing import Any


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
    import PySAM.Annualoutput
    import PySAM.Battery
    import PySAM.BatteryStateful
    import PySAM.Battwatts
    import PySAM.Belpe
    import PySAM.Biomass
    import PySAM.Cashloan
    import PySAM.CbConstructionFinancing
    import PySAM.CbEmpiricalHceHeatLoss
    import PySAM.CbMsptSystemCosts
    import PySAM.DsgFluxPreprocess
    import PySAM.Equpartflip
    import PySAM.Fuelcell
    import PySAM.GenericSystem
    import PySAM.Geothermal
    import PySAM.GeothermalCosts
    import PySAM.Grid
    import PySAM.Hcpv
    import PySAM.HostDeveloper
    import PySAM.Iec61853interp
    import PySAM.Iec61853par
    import PySAM.InvCecCg
    import PySAM.IphToLcoefcr
    import PySAM.Ippppa
    import PySAM.Irradproc
    import PySAM.IsccDesignPoint
    import PySAM.Layoutarea
    import PySAM.Lcoefcr
    import PySAM.Levpartflip
    import PySAM.LinearFresnelDsgIph
    import PySAM.Merchantplant
    import PySAM.MhkCosts
    import PySAM.MhkTidal
    import PySAM.MhkWave
    import PySAM.Poacalib
    import PySAM.Pv6parmod
    import PySAM.PvGetShadeLossMpp
    import PySAM.Pvsamv1
    import PySAM.Pvsandiainv
    import PySAM.Pvwattsv1
    import PySAM.Pvwattsv11ts
    import PySAM.Pvwattsv1Poa
    import PySAM.Pvwattsv5
    import PySAM.Pvwattsv51ts
    import PySAM.Pvwattsv7
    import PySAM.Saleleaseback
    import PySAM.Sco2AirCooler
    import PySAM.Sco2CompCurves
    import PySAM.Sco2CspSystem
    import PySAM.Sco2CspUdPcTables
    import PySAM.Sco2DesignCycle
    import PySAM.Sco2DesignPoint
    import PySAM.Singlediode
    import PySAM.Singlediodeparams
    import PySAM.Singleowner
    import PySAM.SixParsolve
    import PySAM.Snowmodel
    import PySAM.Solarpilot
    import PySAM.Swh
    import PySAM.TcsdirectSteam
    import PySAM.Tcsdish
    import PySAM.TcsgenericSolar
    import PySAM.Tcsiscc
    import PySAM.TcslinearFresnel
    import PySAM.TcsmoltenSalt
    import PySAM.TcsMSLF

    import PySAM.TcstroughEmpirical
    import PySAM.TcstroughPhysical
    import PySAM.Thermalrate
    import PySAM.Thirdpartyownership
    import PySAM.Timeseq
    import PySAM.TroughPhysical
    import PySAM.TroughPhysicalCspSolver
    import PySAM.TroughPhysicalProcessHeat
    import PySAM.UiTesCalcs
    import PySAM.UiUdpcChecks
    import PySAM.UserHtfComparison
    import PySAM.Utilityrate
    import PySAM.Utilityrate2
    import PySAM.Utilityrate3
    import PySAM.Utilityrate4
    import PySAM.Utilityrate5
    import PySAM.WaveFileReader
    import PySAM.Wfcheck
    import PySAM.Wfcsvconv
    import PySAM.Wfreader
    import PySAM.Windbos
    import PySAM.Windcsm
    import PySAM.WindFileReader
    import PySAM.WindObos
    import PySAM.Windpower


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
