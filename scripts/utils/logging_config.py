"""Centralized logging configuration for the scraping pipeline."""

import logging
import sys
from pathlib import Path

_configured = False


def setup_logging(
    name: str = "geo-scraper",
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure and return a logger with file + stderr handlers.

    Args:
        name: Logger name (used as log file prefix).
        level: Logging level.
        log_dir: Directory for log files. If None, file logging is skipped.

    Returns:
        Configured logger instance.
    """
    global _configured

    if _configured:
        return logging.getLogger(name)
    _configured = True

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger so all loggers (including per-class ones) get output
    root = logging.getLogger()
    root.setLevel(level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    return logging.getLogger(name)
