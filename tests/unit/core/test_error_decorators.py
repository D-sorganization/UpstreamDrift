"""Comprehensive tests for src.shared.python.core.error_decorators module.

Tests all decorators, context managers, and utility functions for error handling.
"""

from __future__ import annotations

import logging

import pytest

from src.shared.python.core.error_decorators import (
    ErrorContext,
    check_module_available,
    handle_import_error,
    log_errors,
    retry_on_error,
    safe_import,
    validate_args,
)

# ============================================================================
# Tests for @log_errors decorator
# ============================================================================


class TestLogErrors:
    """Tests for @log_errors decorator."""

    def test_success_passes_through(self) -> None:
        """Decorated function returns value on success."""

        @log_errors("should not trigger")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_reraise_true_propagates_exception(self) -> None:
        """With reraise=True, exception is propagated after logging."""

        @log_errors("operation failed", reraise=True)
        def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

    def test_reraise_false_returns_default(self) -> None:
        """With reraise=False, returns default_return on error."""

        @log_errors("operation failed", reraise=False, default_return=-1)
        def fail() -> int:
            raise RuntimeError("boom")

        assert fail() == -1

    def test_reraise_false_default_none(self) -> None:
        """With reraise=False and no default, returns None."""

        @log_errors("op failed", reraise=False)
        def fail() -> None:
            raise RuntimeError("boom")

        assert fail() is None

    def test_preserves_function_name(self) -> None:
        """Decorator preserves __name__ via functools.wraps."""

        @log_errors("test")
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_logs_error_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Decorator logs the custom error message."""

        @log_errors("custom msg", reraise=False)
        def fail() -> None:
            raise ValueError("inner error")

        with caplog.at_level(logging.ERROR):
            fail()

        assert any("custom msg" in r.message for r in caplog.records)

    def test_passes_kwargs(self) -> None:
        """Decorated function receives keyword arguments correctly."""

        @log_errors("test")
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        assert greet("World", greeting="Hi") == "Hi, World!"


# ============================================================================
# Tests for @handle_import_error decorator
# ============================================================================


class TestHandleImportError:
    """Tests for @handle_import_error decorator."""

    def test_success_returns_value(self) -> None:
        """Returns value when no ImportError occurs."""

        @handle_import_error(module_name="os")
        def import_os() -> str:
            import os  # noqa: F811

            return os.name

        result = import_os()
        assert result is not None

    def test_import_failure_returns_default(self) -> None:
        """Returns default_value when ImportError occurs."""

        @handle_import_error(
            module_name="nonexistent_module_xyz", default_value="fallback"
        )
        def import_missing() -> str:
            raise ImportError("No module named 'nonexistent_module_xyz'")

        result = import_missing()
        assert result == "fallback"

    def test_import_failure_default_none(self) -> None:
        """Returns None when ImportError occurs and no default set."""

        @handle_import_error(module_name="missing")
        def import_missing() -> None:
            raise ImportError("not found")

        assert import_missing() is None

    def test_non_import_error_propagates(self) -> None:
        """Non-ImportError exceptions are not caught."""

        @handle_import_error(module_name="test")
        def fail() -> None:
            raise ValueError("not an import error")

        with pytest.raises(ValueError, match="not an import error"):
            fail()

    def test_preserves_function_name(self) -> None:
        """Decorator preserves __name__."""

        @handle_import_error(module_name="test")
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"


# ============================================================================
# Tests for @retry_on_error decorator
# ============================================================================


class TestRetryOnError:
    """Tests for @retry_on_error decorator."""

    def test_success_no_retry(self) -> None:
        """Function that succeeds is not retried."""
        call_count = 0

        @retry_on_error(max_attempts=3)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_failure(self) -> None:
        """Function is retried on failure."""
        call_count = 0

        @retry_on_error(max_attempts=3, exceptions=(ValueError,), delay=0.0)
        def sometimes_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = sometimes_fail()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self) -> None:
        """Raises exception when all retries exhausted."""

        @retry_on_error(max_attempts=2, exceptions=(ValueError,), delay=0.0)
        def always_fail() -> None:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fail()

    def test_only_catches_specified_exceptions(self) -> None:
        """Only retries on specified exception types."""

        @retry_on_error(max_attempts=3, exceptions=(IOError,), delay=0.0)
        def fail_different() -> None:
            raise ValueError("wrong type")

        with pytest.raises(ValueError):
            fail_different()

    def test_preserves_function_name(self) -> None:
        """Decorator preserves __name__."""

        @retry_on_error()
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"


# ============================================================================
# Tests for ErrorContext context manager
# ============================================================================


class TestErrorContext:
    """Tests for ErrorContext context manager."""

    def test_success_no_exception(self) -> None:
        """Context manager completes without error."""
        with ErrorContext("test operation"):
            result = 1 + 1
        assert result == 2

    def test_reraise_true_propagates(self) -> None:
        """With reraise=True, exception is re-raised."""
        with (
            pytest.raises(ValueError, match="inner"),
            ErrorContext("test op", reraise=True),
        ):
            raise ValueError("inner")

    def test_reraise_false_suppresses(self) -> None:
        """With reraise=False, exception is suppressed."""
        with ErrorContext("test op", reraise=False):
            raise RuntimeError("suppressed")
        # Should reach here without error

    def test_logs_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Context manager logs the operation name on error."""
        with (
            caplog.at_level(logging.ERROR),
            ErrorContext("loading config", reraise=False),
        ):
            raise OSError("disk error")

        assert any("loading config" in r.message for r in caplog.records)

    def test_log_success_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        """With log_success=True, logs on successful completion."""
        with (
            caplog.at_level(logging.DEBUG),
            ErrorContext("batch processing", log_success=True),
        ):
            pass

        # Should have logged success
        log_messages = [r.message for r in caplog.records]
        assert any("batch processing" in msg for msg in log_messages)


# ============================================================================
# Tests for @validate_args decorator
# ============================================================================


class TestValidateArgs:
    """Tests for @validate_args decorator."""

    def test_valid_args_pass(self) -> None:
        """Validation passes when args are valid."""

        @validate_args(x=lambda x: x > 0)
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10

    def test_invalid_args_raise_error(self) -> None:
        """Validation raises ValueError for invalid args."""

        @validate_args(x=lambda x: x > 0)
        def double(x: int) -> int:
            return x * 2

        with pytest.raises(ValueError):
            double(-1)

    def test_multiple_validators(self) -> None:
        """Multiple args can be validated simultaneously."""

        @validate_args(x=lambda x: x > 0, y=lambda y: isinstance(y, str))
        def process(x: int, y: str) -> str:
            return f"{y}: {x}"

        assert process(5, "value") == "value: 5"

    def test_multiple_validator_failure(self) -> None:
        """Fails if any validator returns False."""

        @validate_args(x=lambda x: x > 0, y=lambda y: isinstance(y, str))
        def process(x: int, y: str) -> str:
            return f"{y}: {x}"

        with pytest.raises(ValueError):
            process(5, 123)  # y should be str

    def test_kwargs_validated(self) -> None:
        """Keyword arguments are also validated."""

        @validate_args(name=lambda n: len(n) > 0)
        def greet(name: str) -> str:
            return f"Hello, {name}"

        with pytest.raises(ValueError):
            greet(name="")

    def test_preserves_function_name(self) -> None:
        """Decorator preserves __name__."""

        @validate_args(x=lambda x: x > 0)
        def my_func(x: int) -> int:
            return x

        assert my_func.__name__ == "my_func"


# ============================================================================
# Tests for safe_import
# ============================================================================


class TestSafeImport:
    """Tests for safe_import utility."""

    def test_available_module(self) -> None:
        """Returns module when available."""
        mod = safe_import("os")
        assert mod is not None
        assert hasattr(mod, "getcwd")

    def test_unavailable_module(self) -> None:
        """Returns None when module is not available."""
        mod = safe_import("nonexistent_module_xyz_123")
        assert mod is None

    def test_submodule_import(self) -> None:
        """Can import submodules."""
        mod = safe_import("os.path")
        assert mod is not None
        assert hasattr(mod, "join")


# ============================================================================
# Tests for check_module_available
# ============================================================================


class TestCheckModuleAvailable:
    """Tests for check_module_available utility."""

    def test_available_module(self) -> None:
        """Returns True for available module."""
        assert check_module_available("os") is True

    def test_unavailable_module(self) -> None:
        """Returns False for unavailable module."""
        assert check_module_available("nonexistent_module_xyz_123") is False

    def test_numpy_available(self) -> None:
        """numpy is available in our environment."""
        assert check_module_available("numpy") is True
