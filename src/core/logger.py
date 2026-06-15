"""Structured logging helper for the application.

Provides a small convenience around the stdlib `logging` module and the
application `settings` so modules can import and call `get_logger(__name__)`.
"""
from __future__ import annotations

import logging
from logging import Logger

from src.core.config import settings


def get_logger(name: str) -> Logger:
	"""Return a configured logger for `name`.

	The logger is configured once via ``logging.basicConfig`` using the
	`log_level` from the application settings. Callers may further
	configure handlers if needed.
	"""
	level_name = (settings.log_level or "INFO").upper()
	level = getattr(logging, level_name, logging.INFO)

	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
	)

	return logging.getLogger(name)
