"""Unit tests for core module (logging and exceptions).

TEST-001: Added test coverage for core.py (previously 0% coverage).
Issue #2061: Updated to reflect stdlib-only logging in _core.py.
"""

import logging

import pytest

from src.shared.python.core import (
    DataFormatError,
    EngineNotFoundError,
    GolfModelingError,
    get_logger,
    setup_logging,
    setup_structured_logging,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_golf_modeling_error(self) -> None:
        """Test GolfModelingError is a proper exception."""
        error = GolfModelingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_engine_not_found_error(self) -> None:
        """Test EngineNotFoundError inherits from GolfModelingError."""
        error = EngineNotFoundError("Drake")
        # New error format wraps the engine name
        assert "Drake" in str(error)
        assert isinstance(error, GolfModelingError)
        assert isinstance(error, Exception)

    def test_data_format_error(self) -> None:
        """Test DataFormatError inherits from GolfModelingError."""
        error = DataFormatError("Invalid CSV format")
        assert str(error) == "Invalid CSV format"
        assert isinstance(error, GolfModelingError)
        assert isinstance(error, Exception)

    def test_exceptions_can_be_raised(self) -> None:
        """Test that exceptions can be raised and caught."""
        with pytest.raises(GolfModelingError) as exc_info:
            raise GolfModelingError("Test message")
        assert "Test message" in str(exc_info.value)

        with pytest.raises(EngineNotFoundError) as exc_info:
            raise EngineNotFoundError("Engine missing")
        assert "Engine missing" in str(exc_info.value)

        with pytest.raises(DataFormatError) as exc_info:
            raise DataFormatError("Bad data")
        assert "Bad data" in str(exc_info.value)


class TestLegacyLogging:
    """Test legacy setup_logging function."""

    def test_setup_logging_returns_logger(self) -> None:
        """Test setup_logging returns a configured logger."""
        logger = setup_logging("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_setup_logging_default_level(self) -> None:
        """Test setup_logging uses INFO level by default."""
        logger = setup_logging("test_default_level")
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self) -> None:
        """Test setup_logging with custom log level."""
        logger = setup_logging("test_custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logging_adds_handler(self) -> None:
        """Test setup_logging adds a StreamHandler."""
        logger = setup_logging("test_handler")
        assert len(logger.handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_setup_logging_idempotent(self) -> None:
        """Test calling setup_logging twice doesn't duplicate handlers."""
        logger1 = setup_logging("test_idempotent")
        initial_handlers = len(logger1.handlers)

        logger2 = setup_logging("test_idempotent")
        assert logger1 is logger2  # Same logger instance
        assert len(logger2.handlers) == initial_handlers  # No duplicate handlers


class TestGetLogger:
    """Test get_logger returns a stdlib Logger.

    Issue #2061: _core.get_logger now returns logging.Logger (stdlib) instead
    of a structlog BoundLogger.  This makes logging consistent with the rest of
    the codebase which uses stdlib logging directly.  Structured-logging
    configuration (including optional structlog processors) is handled by
    logging_pkg.logging_config.setup_logging() at application start-up.
    """

    def test_get_logger_returns_stdlib_logger(self) -> None:
        """Test get_logger returns a stdlib Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_name(self) -> None:
        """Test get_logger assigns the correct name."""
        logger = get_logger("specific_name")
        assert logger.name == "specific_name"

    def test_get_logger_has_standard_methods(self) -> None:
        """Test logger has the standard logging methods."""
        logger = get_logger("test_methods")
        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)

    def test_get_logger_multiple_calls_same_instance(self) -> None:
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2

    def test_get_logger_different_names_different_instances(self) -> None:
        """Test that different names return different loggers."""
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2


class TestSetupStructuredLogging:
    """Test setup_structured_logging delegates correctly.

    Issue #2061: setup_structured_logging now delegates to
    logging_pkg.logging_config.setup_logging() rather than configuring
    structlog directly.
    """

    def test_setup_structured_logging_basic(self) -> None:
        """Test basic setup_structured_logging call does not raise."""
        setup_structured_logging()

    def test_setup_structured_logging_with_dev_mode(self) -> None:
        """Test setup with development mode enabled."""
        setup_structured_logging(dev_mode=True, json_output=False)
        logger = get_logger("test_dev")
        assert logger is not None

    def test_setup_structured_logging_with_json(self) -> None:
        """Test setup with JSON output mode."""
        setup_structured_logging(dev_mode=False, json_output=True)
        logger = get_logger("test_json")
        assert logger is not None

    def test_setup_structured_logging_custom_level(self) -> None:
        """Test setup with custom log level."""
        setup_structured_logging(level=logging.DEBUG)
        logger = get_logger("test_level")
        assert logger is not None

    def test_setup_structured_logging_idempotent(self) -> None:
        """Test that calling setup_structured_logging multiple times is safe."""
        setup_structured_logging()
        setup_structured_logging()  # Should not raise or reconfigure
        logger = get_logger("test_idempotent_struct")
        assert logger is not None


class TestLoggingCompatibility:
    """Test compatibility between legacy and get_logger-based logging."""

    def test_both_functions_return_loggers(self) -> None:
        """Test that setup_logging and get_logger both return Logger instances."""
        legacy_logger = setup_logging("legacy_module")
        stdlib_logger = get_logger("stdlib_module")

        assert isinstance(legacy_logger, logging.Logger)
        assert isinstance(stdlib_logger, logging.Logger)

    def test_loggers_log_without_error(self) -> None:
        """Test that loggers from both sources can log without error."""
        legacy_logger = setup_logging("legacy_compat")
        stdlib_logger = get_logger("stdlib_compat")

        # Both should work
        legacy_logger.info("Legacy log message")
        stdlib_logger.info("stdlib log message")

    def test_same_module_name_same_logger(self) -> None:
        """Test using same module name with both APIs returns same logger."""
        legacy = setup_logging("test_module_compat")
        stdlib = get_logger("test_module_compat")

        # Both APIs return the same underlying stdlib logger
        assert legacy is stdlib

    def test_logger_exception_logging(self) -> None:
        """Test logger can log exceptions."""
        logger = get_logger("test_exception")
        try:
            raise ValueError("Test exception")
        except ValueError:
            # Should not raise when logging exception
            logger.error("error_occurred", exc_info=True)

    def test_multiple_loggers_independent(self) -> None:
        """Test that multiple loggers are independent."""
        logger1 = get_logger("module1_compat")
        logger2 = get_logger("module2_compat")

        # Both should work independently
        logger1.info("event1")
        logger2.info("event2")
