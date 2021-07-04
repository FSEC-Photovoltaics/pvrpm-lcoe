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

    WARRANTY = "warranty"
    DAYS = "days"

    FAILURE = "failures"
    DIST = "distribution"
    PARAM = "parameters"
    MEAN = "mean"
    STD = "std"
    LABOR = "labor_time"
    COST = "cost"
    FRAC = "fraction"

    REPAIR = "repairs"
    DEGRADE = "degradation"

    STRING = "string"
    COMBINER = "combiner"
    INVERTER = "inverter"
    DISCONNECT = "disconnect"
    TRANSFORMER = "transformer"
    GRID = "grid"
    TRACKER = "tracker"

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

    repair_keys = [
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

    # for output generation
    losses = [
        "acwiring_loss",
        "dcoptimizer_loss",
        "subarray1_dcwiring_loss",
        "subarray1_diodeconn_loss",
        "subarray1_mismatch_loss",
        "subarray1_nameplate_loss",
        "subarray1_rear_irradiance_loss",
        "subarray1_soiling",
        "subarray1_tracking_loss",
        "subarray2_dcwiring_loss",
        "subarray2_diodeconn_loss",
        "subarray2_mismatch_loss",
        "subarray2_nameplate_loss",
        "subarray2_rear_irradiance_loss",
        "subarray2_soiling",
        "subarray2_tracking_loss",
        "subarray3_dcwiring_loss",
        "subarray3_diodeconn_loss",
        "subarray3_mismatch_loss",
        "subarray3_nameplate_loss",
        "subarray3_rear_irradiance_loss",
        "subarray3_soiling",
        "subarray3_tracking_loss",
        "subarray4_dcwiring_loss",
        "subarray4_diodeconn_loss",
        "subarray4_mismatch_loss",
        "subarray4_nameplate_loss",
        "subarray4_rear_irradiance_loss",
        "subarray4_soiling",
        "subarray4_tracking_loss",
        "transformer_load_loss",
        "transformer_no_load_loss",
        "transmission_loss",
    ]
