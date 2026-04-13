import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "app",
    log_file: str = "app.log",
    level: int = logging.DEBUG,
    to_console: bool = True,
):
    """
    Creates and configures a reusable logger with UTF-8 support.

    Args:
        name (str): Logger name
        log_file (str): File to write logs to
        level (int): Logging level
        to_console (bool): Whether to also print logs to console

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    # If already configured → update level and return
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if to_console:
        stream = sys.stdout

        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass  # safe fallback

        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None):
    """
    Quick access to an existing logger or default one.
    """
    return logging.getLogger(name if name else "app")