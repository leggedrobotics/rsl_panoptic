# import packages
import argparse
import logging


# Initialize logger for the project
import logging as python_logging
import sys

_logger = python_logging.getLogger("ssl_panoptic")
_logger.setLevel(python_logging.INFO)

console_handler = python_logging.StreamHandler(sys.stdout)
console_handler.setFormatter(python_logging.Formatter("%(module)s: %(message)s"))
_logger.addHandler(console_handler)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Change logging level to debug",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Change logging level to debug",
    )
    return parser


def logger_level(argument: argparse.Namespace) -> None:
    """
    Function to change "ml_utils" logger level to debug

    Parameters
    ----------
    argument: is parsed

    Returns
    -------
    logger: logger handle pointing to sys.stdout on defined logger level

    """
    if argument.debug:
        _logger.setLevel(logging.DEBUG)
    if argument.verbose:
        _logger.setLevel(logging.NOTSET)
