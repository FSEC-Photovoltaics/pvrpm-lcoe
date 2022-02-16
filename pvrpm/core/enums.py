class ConfigKeys:
    MODULE_ORDER = "module_order"
    NUM_REALIZATION = "num_realizations"
    NUM_COMBINERS = "num_combiners"
    NUM_TRANSFORMERS = "num_transformers"
    NUM_TRACKERS = "num_trackers"

    RESULTS_FOLDER = "results_folder"
    CONF_INTERVAL = "conf_interval"

    LABOR_RATE = "present_day_labor_rate"
    INFLATION = "inflation"
    WORST_TRACKER = "use_worst_case_tracker"

    MODULE = "module"
    NAME = "name"
    CAN_FAIL = "can_fail"
    CAN_REPAIR = "can_repair"
    CAN_MONITOR = "can_monitor"

    WARRANTY = "warranty"
    DAYS = "days"

    FAILURE = "failures"
    PARTIAL_FAIL = "concurrent_failures"
    DIST = "distribution"
    PARAM = "parameters"
    MEAN = "mean"
    STD = "std"
    SHAPE = "shape"
    LABOR = "labor_time"
    COST = "cost"
    COST_PER_WATT = "cost_per_watt"
    FRAC = "fraction"
    DECAY_FRAC = "decay_fraction"

    REPAIR = "repairs"
    PARTIAL_REPAIR = "concurrent_repairs"
    MONITORING = "monitoring"
    DEGRADE = "degradation"

    STRING = "string"
    COMBINER = "combiner"
    INVERTER = "inverter"
    DISCONNECT = "disconnect"
    TRANSFORMER = "transformer"
    GRID = "grid"
    TRACKER = "tracker"

    # static monitoring
    INDEP_MONITOR = "indep_monitoring"
    INTERVAL = "interval"
    LEVELS = "levels"

    # cross level monitoring
    COMP_MONITOR = "component_level_monitoring"
    FAIL_THRESH = "global_threshold"
    FAIL_PER_THRESH = "failure_per_threshold"
    COMP_FUNC = "compounding_function"
    COMP_PARAM = "compound_parameters"
    # compounding functions parameters
    THRESH = "threshold"
    STEP = "step"
    BASE = "base"
    CONST = "constant"
    SLOPE = "slope"
    # compound functions
    LOG = "log"
    LINEAR = "linear"

    # supported distribution functions
    NORMAL = "normal"
    EXPON = "exponential"
    WEIBULL = "weibull"
    LOGNORM = "lognormal"
    UNIFORM = "uniform"

    # added during case setup
    NUM_COMPONENT = "count"
    TRACKING = "is_tracking_system"
    MULTI_SUBARRAY = "has_multiple_subarrays"
    MODULES_PER_STR = "num_modules_per_string"
    INVERTER_SIZE = "inverter_size"
    COMBINER_PER_INVERTER = "num_combiners_per_inverter"
    STR_PER_COMBINER = "num_strings_per_combiner"
    INVERTER_PER_TRANS = "num_inverters_per_transformer"
    LIFETIME_YRS = "system_lifetime_yrs"

    # for error checking
    needed_keys = [
        MODULE_ORDER,
        NUM_REALIZATION,
        NUM_COMBINERS,
        NUM_TRANSFORMERS,
        NUM_TRACKERS,
        RESULTS_FOLDER,
        CONF_INTERVAL,
        LABOR_RATE,
        INFLATION,
        WORST_TRACKER,
        MODULE,
        STRING,
        COMBINER,
        INVERTER,
        DISCONNECT,
        TRANSFORMER,
        GRID,
    ]

    component_keys = [
        MODULE,
        STRING,
        COMBINER,
        INVERTER,
        DISCONNECT,
        TRANSFORMER,
        GRID,
        TRACKER,
    ]

    failure_keys = [
        DIST,
        PARAM,
        LABOR,
        COST,
    ]

    partial_failure_keys = [
        DIST,
        PARAM,
        LABOR,
        COST,
    ]

    monitoring_keys = [
        DIST,
        PARAM,
    ]

    repair_keys = [
        DIST,
        PARAM,
    ]

    partial_repair_keys = [
        DIST,
        PARAM,
    ]

    dists = [
        NORMAL,
        EXPON,
        WEIBULL,
        LOGNORM,
        UNIFORM,
    ]

    indep_monitor_keys = [
        COST,
        LEVELS,
        LABOR,
    ]

    compound_funcs = [
        STEP,
        LOG,
        LINEAR,
        EXPON,
        CONST,
    ]

    compound_keys = [
        #        FAIL_THRESH,
        #        COMP_FUNC,
        #        COMP_PARAM,
        DIST,
        PARAM,
    ]

    compound_levels = [
        STRING,
        COMBINER,
        INVERTER,
        DISCONNECT,
        TRANSFORMER,
        GRID,
    ]

    # for output generation
    losses = [
        "annual_poa_shading_loss_percent",
        "annual_poa_soiling_loss_percent",
        "annual_poa_cover_loss_percent",
        "annual_dc_module_loss_percent",
        "annual_dc_mppt_clip_loss_percent",
        "annual_dc_mismatch_loss_percent",
        "annual_dc_diodes_loss_percent",
        "annual_dc_wiring_loss_percent",
        "annual_dc_tracking_loss_percent",
        "annual_dc_nameplate_loss_percent",
        "annual_dc_optimizer_loss_percent",
        "annual_dc_perf_adj_loss_percent",
        "annual_ac_inv_clip_loss_percent",
        "annual_ac_inv_pso_loss_percent",
        "annual_ac_inv_pnt_loss_percent",
        "annual_ac_inv_eff_loss_percent",
        "ac_loss",
        "annual_transmission_loss_percent",
        "annual_ac_perf_adj_loss_percent",
        "annual_xfmr_loss_percent",
    ]
