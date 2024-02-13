"""Module for logging"""
import logging


def init_logger(file_name: str, level=logging.INFO, init_file_handler=False):
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers if present to avoid duplication
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add the handlers to the logger
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    if init_file_handler:
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
