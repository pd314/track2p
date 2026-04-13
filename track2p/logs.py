import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "app",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    to_console: bool = True,
    force_reinit: bool = False,
):
    logger = logging.getLogger(name)

    if logger.handlers and not force_reinit:
        logger.setLevel(level)
        return logger

    if force_reinit:
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # FILE HANDLER
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # CONSOLE HANDLER
    if to_console:
        stream = sys.stdout
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str | None = None):
    """Simple wrapper around standard logging system."""
    return logging.getLogger(name if name else "app")