"""Logging configuration for PaperBanana."""

from __future__ import annotations

import logging

import structlog


def configure_logging(*, verbose: bool = False) -> None:
    """Configure structlog output level.

    Args:
        verbose: If True, show detailed agent progress and timing at DEBUG level.
                 If False (default), suppress logs below WARNING for clean output.
    """
    level = logging.DEBUG if verbose else logging.WARNING

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
