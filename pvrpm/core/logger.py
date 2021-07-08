import logging

if "logger" not in locals() and "logger" not in globals():
    logger = logging.getLogger("pvrpm")


def init_logger():
    """
    Initalizes logger for module
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s--%(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)
