"""Tests for src.shared.python.logging_pkg.logging_config (Issues #1949, #1744)."""

from __future__ import annotations

import logging

from src.shared.python.logging_pkg.logging_config import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_BYTES,
    DETAILED_LOG_FORMAT,
    SIMPLE_LOG_FORMAT,
    LogLevel,
    SensitiveDataFilter,
    configure_test_logging,
    get_logger,
    setup_logging,
)

# ---------------------------------------------------------------------------
# LogLevel enum
# ---------------------------------------------------------------------------


class TestLogLevel:
    def test_has_standard_levels(self) -> None:
        assert LogLevel.DEBUG
        assert LogLevel.INFO
        assert LogLevel.WARNING
        assert LogLevel.ERROR

    def test_debug_lower_than_info(self) -> None:
        assert LogLevel.DEBUG.value < LogLevel.INFO.value

    def test_info_lower_than_warning(self) -> None:
        assert LogLevel.INFO.value < LogLevel.WARNING.value

    def test_warning_lower_than_error(self) -> None:
        assert LogLevel.WARNING.value < LogLevel.ERROR.value

    def test_values_match_stdlib(self) -> None:
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------


class TestLogFormatConstants:
    def test_default_format_non_empty(self) -> None:
        assert len(DEFAULT_LOG_FORMAT) > 0

    def test_simple_format_non_empty(self) -> None:
        assert len(SIMPLE_LOG_FORMAT) > 0

    def test_detailed_format_non_empty(self) -> None:
        assert len(DETAILED_LOG_FORMAT) > 0

    def test_default_max_bytes_positive(self) -> None:
        assert DEFAULT_MAX_BYTES > 0

    def test_default_backup_count_positive(self) -> None:
        assert DEFAULT_BACKUP_COUNT > 0


# ---------------------------------------------------------------------------
# SensitiveDataFilter — redaction
# ---------------------------------------------------------------------------


class TestSensitiveDataFilter:
    def _make_record(self, msg: str, args: tuple = ()) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=args,
            exc_info=None,
        )
        return record

    def test_filter_returns_true(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("plain message")
        assert flt.filter(record) is True

    def test_redacts_password(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("password=secret123")
        flt.filter(record)
        assert "secret123" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_api_key(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("api_key=abc123xyz")
        flt.filter(record)
        assert "abc123xyz" not in record.msg
        assert "REDACTED" in record.msg

    def test_redacts_access_token(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("access_token=tok_abcdefg")
        flt.filter(record)
        assert "tok_abcdefg" not in record.msg

    def test_plain_text_unchanged(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("user logged in from 192.168.1.1")
        flt.filter(record)
        assert record.msg == "user logged in from 192.168.1.1"

    def test_handles_percent_formatting(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("connecting to %s", ("db.example.com",))
        flt.filter(record)
        assert "db.example.com" in record.msg
        # args should be cleared after formatting
        assert record.args is None

    def test_case_insensitive_password(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("Password=MySuperSecret")
        flt.filter(record)
        assert "MySuperSecret" not in record.msg

    def test_multiple_sensitive_fields(self) -> None:
        flt = SensitiveDataFilter()
        record = self._make_record("api_key=key1 password=pass1")
        flt.filter(record)
        assert "key1" not in record.msg
        assert "pass1" not in record.msg


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger(self) -> None:
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger) or hasattr(logger, "info")

    def test_logger_name_set(self) -> None:
        logger = get_logger("my_test_module")
        # Standard logging.Logger has .name; structlog bound loggers may not
        if isinstance(logger, logging.Logger):
            assert logger.name == "my_test_module"

    def test_none_name_returns_root(self) -> None:
        logger = get_logger(None)
        assert logger is not None

    def test_can_log_info(self) -> None:
        logger = get_logger("test_can_log")
        logger.info("test message — no exception expected")  # should not raise


# ---------------------------------------------------------------------------
# configure_test_logging
# ---------------------------------------------------------------------------


class TestConfigureTestLogging:
    def test_runs_without_error(self) -> None:
        configure_test_logging()  # should not raise

    def test_sets_level(self) -> None:
        configure_test_logging(level=LogLevel.WARNING)
        root = logging.getLogger()
        assert root.level <= logging.WARNING or root.level == logging.WARNING


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_runs_without_error(self) -> None:
        setup_logging()  # should not raise

    def test_custom_level_does_not_raise(self) -> None:
        # Just verifying no exception — root level can be affected by other workers
        setup_logging(level=LogLevel.ERROR)  # should not raise

    def test_redacts_password_comma_separated(self) -> None:
        """Tests that a password in a comma-separated string is redacted without leaking the suffix."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connected: password=abc,def, user=test",
            args=(),
            exc_info=None,
        )
        flt = SensitiveDataFilter()
        flt.filter(record)
        assert "password=***REDACTED***" in record.msg
        assert "def" in record.msg

    def test_redacts_password_json(self) -> None:
        """Tests that a password in a JSON payload is redacted correctly."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='{"password":"abc,def"}',
            args=(),
            exc_info=None,
        )
        flt = SensitiveDataFilter()
        flt.filter(record)
        assert '{"password":"***REDACTED***"}' in record.msg
