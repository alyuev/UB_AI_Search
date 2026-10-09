"""Logging helpers: rotating file handlers under logs/ plus console output."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config import LOGS_DIR

_configured: dict[str, logging.Logger] = {}

_VALID_NAMES = {"app", "index", "analysis"}


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to logs/<name>.log (rotating, utf-8) + console.

    `name` should be one of "app", "index", "analysis" but any name is
    accepted and mapped to logs/<name>.log.
    """
    if name in _configured:
        return _configured[name]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"smartsearch.{name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = LOGS_DIR / f"{name}.log"
    fh = RotatingFileHandler(
        file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _configured[name] = logger
    return logger
