"""Unit tests for core module (logging and exceptions).

TEST-001: Added test coverage for core.py (previously 0% coverage).
OBS-001: Added tests for new structured logging functionality.
"""

import logging

import pytest

from shared.python.core import (
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
        error = EngineNotFoundError("Drake not found")
        assert str(error) == "Drake not found"
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


class TestStructuredLogging:
    """Test structured logging with structlog."""

    def setup_method(self) -> None:
        """Reset structured logging configuration before each test."""
        # Reset the global configuration flag
        import shared.python.core as core_module
        core_module._structured_logging_configured = False

    def test_setup_structured_logging_basic(self) -> None:
        """Test basic setup_structured_logging call."""
        setup_structured_logging()
        # Should not raise any exceptions

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
        # Logger should be created without errors
        assert logger is not None

    def test_setup_structured_logging_idempotent(self) -> None:
        """Test that calling setup_structured_logging multiple times is safe."""
        setup_structured_logging()
        setup_structured_logging()  # Should not raise or reconfigure
        logger = get_logger("test_idempotent_struct")
        assert logger is not None

    def test_get_logger_returns_structlog_logger(self) -> None:
        """Test get_logger returns a structlog BoundLogger."""
        logger = get_logger("test_module")
        # Check that it has structlog's bind method
        assert hasattr(logger, "bind")

    def test_get_logger_auto_configures(self) -> None:
        """Test get_logger auto-configures if not already configured."""
        # Don't call setup_structured_logging first
        logger = get_logger("test_auto_config")
        assert logger is not None
        assert hasattr(logger, "bind")

    def test_logger_supports_structured_data(self) -> None:
        """Test logger can handle keyword arguments."""
        logger = get_logger("test_structured")
        # Should not raise exceptions
        logger.info("test_event", key1="value1", key2=42, key3=True)

    def test_logger_bind_creates_context(self) -> None:
        """Test logger.bind creates logger with persistent context."""
        logger = get_logger("test_bind")
        bound_logger = logger.bind(request_id="test-123", user="alice")
        assert bound_logger is not None
        # Should not raise when logging
        bound_logger.info("test_event", action="test")

    def test_logger_different_levels(self) -> None:
        """Test logger supports different log levels."""
        logger = get_logger("test_levels")
        # Should not raise exceptions
        logger.debug("debug_event", level="debug")
        logger.info("info_event", level="info")
        logger.warning("warning_event", level="warning")
        logger.error("error_event", level="error")

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
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Both should work independently
        logger1.info("event1", module="module1")
        logger2.info("event2", module="module2")

    def test_structured_logging_processors_configured(self) -> None:
        """Test that processors are properly configured."""
        setup_structured_logging(dev_mode=True)
        # Verify structlog is configured (doesn't raise)
        logger = get_logger("test_processors")
        logger.info("test", timestamp=True, caller=True)


class TestLoggingCompatibility:
    """Test compatibility between legacy and structured logging."""

    def test_both_logging_systems_coexist(self) -> None:
        """Test that legacy and structured logging can coexist."""
        legacy_logger = setup_logging("legacy_module")
        structured_logger = get_logger("structured_module")

        # Both should work
        legacy_logger.info("Legacy log message")
        structured_logger.info("structured_event", key="value")

    def test_same_module_name_different_loggers(self) -> None:
        """Test using same module name with both systems."""
        legacy = setup_logging("test_module_compat")
        structured = get_logger("test_module_compat")

        # Should not interfere with each other
        legacy.info("Legacy message")
        structured.info("structured_message", type="structured")
