"""Tests for tracing - Request tracing and correlation ID utilities.

These tests verify the tracing system using Design by Contract
principles to ensure proper request tracking across the API.
"""

import re
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestTracedLog:
    """Functional tests for traced_log function."""

    def test_injects_request_id(self) -> None:
        """Test that traced_log injects request_id when available."""
        from src.api.utils.tracing import (
            _request_id_var,
            traced_log,
        )

        test_id = "req_logtest"
        token = _request_id_var.set(test_id)

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("info", "Test message")

            # Check that request_id was in extra
            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["request_id"] == test_id

        _request_id_var.reset(token)

    def test_injects_correlation_id_from_context(self) -> None:
        """Test that traced_log injects correlation_id from context."""
        from src.api.utils.tracing import (
            TraceContext,
            _trace_context_var,
            traced_log,
        )

        context = TraceContext(
            request_id="req_123",
            correlation_id="cor_logtest",
        )
        token = _trace_context_var.set(context)

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("info", "Test message")

            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["correlation_id"] == "cor_logtest"

        _trace_context_var.reset(token)

    def test_passes_kwargs_to_extra(self) -> None:
        """Test that kwargs are passed to extra."""
        from src.api.utils.tracing import traced_log

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("info", "Test", engine="mujoco", model="arm.urdf")

            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["engine"] == "mujoco"
            assert call_args[1]["extra"]["model"] == "arm.urdf"

    def test_supports_different_log_levels(self) -> None:
        """Test that different log levels are supported."""
        from src.api.utils.tracing import traced_log

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("debug", "Debug message")
            mock_logger.debug.assert_called_once()

            traced_log("warning", "Warning message")
            mock_logger.warning.assert_called_once()

            traced_log("error", "Error message")
            mock_logger.error.assert_called_once()
