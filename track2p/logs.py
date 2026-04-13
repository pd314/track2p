import logging
import os


def setup_logger(
    name: str = "app",
    log_file: str = "app.log",
    level: int = logging.DEBUG,
    to_console: bool = True,
):
    """
    Creates and configures a reusable logger.

    Args:
        name (str): Logger name
        log_file (str): File to write logs to
        level (int): Logging level
        to_console (bool): Whether to also print logs to console

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Log format (includes file name, line number, function, time, level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger.addHandler(file_handler)

    # Optional console handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None):
    """
    Quick access to an existing logger or default one.
    """
    return logging.getLogger(name if name else "app")