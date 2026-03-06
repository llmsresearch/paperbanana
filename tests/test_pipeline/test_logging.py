"""Tests for logging configuration."""

from __future__ import annotations

import structlog

from paperbanana.core.logging import configure_logging


def test_configure_logging_default_suppresses_info():
    """Test that default logging sets filtering at WARNING level."""
    configure_logging(verbose=False)
    logger = structlog.get_logger().bind()
    assert "FilteringAtWarning" in type(logger).__name__


def test_configure_logging_verbose_enables_debug():
    """Test that verbose logging sets filtering at DEBUG level."""
    configure_logging(verbose=True)
    logger = structlog.get_logger().bind()
    assert "FilteringAtDebug" in type(logger).__name__


def test_configure_logging_verbose_false_then_true():
    """Test that logging can be reconfigured from quiet to verbose."""
    configure_logging(verbose=False)
    logger = structlog.get_logger().bind()
    assert "FilteringAtWarning" in type(logger).__name__

    configure_logging(verbose=True)
    logger = structlog.get_logger().bind()
    assert "FilteringAtDebug" in type(logger).__name__
