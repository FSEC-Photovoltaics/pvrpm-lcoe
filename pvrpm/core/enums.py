class ConfigKeys:
    MODULE_ORDER = "module_order"
    NUM_REALIZATION = "num_realizations"
    NUM_COMBINERS = "num_combiners"
    NUM_TRANSFORMERS = "num_transformers"
    NUM_TRACKERS = "num_trackers"

    RESULTS_FOLDER = "results_folder"
    P_VALUE = "p_value"
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
        P_VALUE,
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
