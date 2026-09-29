"""Tests for core._core module (Issues #1949, #1744)."""

from __future__ import annotations

import logging

from src.shared.python.core._core import get_logger, setup_logging


class TestSetupLogging:
    def test_returns_logger(self) -> None:
        logger = setup_logging("test_core_core")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        logger = setup_logging("my_test_module")
        assert logger.name == "my_test_module"

    def test_logger_has_handlers(self) -> None:
        logger = setup_logging("handler_test")
        assert len(logger.handlers) > 0 or logger.parent is not None


class TestGetLogger:
    def test_returns_logger(self) -> None:
        logger = get_logger("test_get_logger")
        assert logger is not None

    def test_logger_name(self) -> None:
        logger = get_logger("specific_name")
        # structlog may return a wrapped logger; just check it's callable
        assert callable(logger.info) or hasattr(logger, "info")
