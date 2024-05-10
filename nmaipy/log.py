import logging
import sys
from typing import Optional, Union


def get_logger() -> logging.Logger:
    """
    Get package logger
    """
    logger = logging.getLogger("nmaipy")
    logger.propagate = False
    return logger


def configure_logger(log_level: Optional[Union[int, str]] = logging.INFO):
    """
    Configure logger
    """
    logger = get_logger()
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setStream(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - %(filename)s:%(lineno)d | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
