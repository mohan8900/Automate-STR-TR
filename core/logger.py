"""
Structured JSON logger using loguru.
All modules use get_logger(__name__) to get a named logger.
"""
from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger as _root_logger


_configured = False


def _setup_logger(log_level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    Path("logs").mkdir(exist_ok=True)

    _root_logger.remove()

    # Console — human-readable
    _root_logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{extra[module]}</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # File — JSON for structured search / Grafana Loki
    _root_logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        serialize=True,  # JSON output
    )

    # Separate error log
    _root_logger.add(
        "logs/errors.log",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
    )

    _configured = True


def get_logger(module: str = "app", log_level: str = "INFO"):
    """Return a loguru logger bound to the given module name."""
    _setup_logger(log_level)
    return _root_logger.bind(module=module)
