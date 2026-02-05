"""Tests for tracing - Request tracing and correlation ID utilities.

These tests verify the tracing system using Design by Contract
principles to ensure proper request tracking across the API.
"""

import re
from unittest.mock import MagicMock, patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestGenerateRequestIdContract:
    """Design by Contract tests for generate_request_id function."""

    def test_returns_string(self):
        """Postcondition: Returns a string."""
        from src.api.utils.tracing import generate_request_id

        result = generate_request_id()
        assert isinstance(result, str)

    def test_returns_non_empty(self):
        """Postcondition: Returns non-empty string."""
        from src.api.utils.tracing import generate_request_id

        result = generate_request_id()
        assert len(result) > 0

    def test_starts_with_prefix(self):
        """Postcondition: Starts with 'req_' prefix."""
        from src.api.utils.tracing import generate_request_id

        result = generate_request_id()
        assert result.startswith("req_")


class TestGenerateRequestId:
    """Functional tests for generate_request_id."""

    def test_generates_unique_ids(self):
        """Test that each call generates a unique ID."""
        from src.api.utils.tracing import generate_request_id

        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_id_format(self):
        """Test that ID has expected format."""
        from src.api.utils.tracing import generate_request_id

        result = generate_request_id()
        # Format: req_ + 16 hex characters
        assert re.match(r"^req_[a-f0-9]{16}$", result)


class TestGenerateCorrelationIdContract:
    """Design by Contract tests for generate_correlation_id function."""

    def test_returns_string(self):
        """Postcondition: Returns a string."""
        from src.api.utils.tracing import generate_correlation_id

        result = generate_correlation_id()
        assert isinstance(result, str)

    def test_starts_with_prefix(self):
        """Postcondition: Starts with 'cor_' prefix."""
        from src.api.utils.tracing import generate_correlation_id

        result = generate_correlation_id()
        assert result.startswith("cor_")


class TestGenerateCorrelationId:
    """Functional tests for generate_correlation_id."""

    def test_generates_unique_ids(self):
        """Test that each call generates a unique ID."""
        from src.api.utils.tracing import generate_correlation_id

        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_id_format(self):
        """Test that ID has expected format."""
        from src.api.utils.tracing import generate_correlation_id

        result = generate_correlation_id()
        # Format: cor_ + 16 hex characters
        assert re.match(r"^cor_[a-f0-9]{16}$", result)


class TestRequestIdContextContract:
    """Design by Contract tests for request ID context management."""

    def test_get_returns_string(self):
        """Postcondition: get_request_id returns a string."""
        from src.api.utils.tracing import get_request_id

        result = get_request_id()
        assert isinstance(result, str)

    def test_set_returns_token(self):
        """Postcondition: set_request_id returns a context token."""
        from src.api.utils.tracing import set_request_id

        token = set_request_id("test_id")
        assert token is not None


class TestRequestIdContext:
    """Functional tests for request ID context management."""

    def test_get_returns_empty_when_not_set(self):
        """Test that get returns empty string when not set."""
        from src.api.utils.tracing import (
            _request_id_var,
            get_request_id,
        )

        # Reset to default state
        token = _request_id_var.set("")
        try:
            result = get_request_id()
            assert result == ""
        finally:
            _request_id_var.reset(token)

    def test_set_and_get_round_trip(self):
        """Test that set and get work together."""
        from src.api.utils.tracing import get_request_id, set_request_id

        test_id = "req_test123"
        token = set_request_id(test_id)
        try:
            assert get_request_id() == test_id
        finally:
            from src.api.utils.tracing import _request_id_var

            _request_id_var.reset(token)

    def test_token_can_reset_value(self):
        """Test that token can reset to previous value."""
        from src.api.utils.tracing import (
            _request_id_var,
            get_request_id,
            set_request_id,
        )

        original = "original_id"
        new_id = "new_id"

        original_token = set_request_id(original)
        try:
            assert get_request_id() == original

            new_token = set_request_id(new_id)
            assert get_request_id() == new_id

            _request_id_var.reset(new_token)
            assert get_request_id() == original
        finally:
            _request_id_var.reset(original_token)


class TestTraceContextContract:
    """Design by Contract tests for TraceContext dataclass."""

    def test_to_dict_returns_dict(self):
        """Postcondition: to_dict returns a dictionary."""
        from src.api.utils.tracing import TraceContext

        context = TraceContext(
            request_id="req_123",
            correlation_id="cor_456",
        )
        result = context.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_fields(self):
        """Postcondition: to_dict includes required fields."""
        from src.api.utils.tracing import TraceContext

        context = TraceContext(
            request_id="req_123",
            correlation_id="cor_456",
        )
        result = context.to_dict()

        assert "request_id" in result
        assert "correlation_id" in result
        assert "operation" in result
        assert "metadata" in result


class TestTraceContext:
    """Functional tests for TraceContext dataclass."""

    def test_default_values(self):
        """Test default values for optional fields."""
        from src.api.utils.tracing import TraceContext

        context = TraceContext(
            request_id="req_123",
            correlation_id="cor_456",
        )

        assert context.operation == ""
        assert context.start_time == 0.0
        assert context.metadata == {}

    def test_custom_operation(self):
        """Test setting custom operation."""
        from src.api.utils.tracing import TraceContext

        context = TraceContext(
            request_id="req_123",
            correlation_id="cor_456",
            operation="GET /api/health",
        )

        assert context.operation == "GET /api/health"

    def test_to_dict_values(self):
        """Test that to_dict returns correct values."""
        from src.api.utils.tracing import TraceContext

        context = TraceContext(
            request_id="req_abc",
            correlation_id="cor_xyz",
            operation="POST /api/simulate",
            metadata={"engine": "mujoco"},
        )
        result = context.to_dict()

        assert result["request_id"] == "req_abc"
        assert result["correlation_id"] == "cor_xyz"
        assert result["operation"] == "POST /api/simulate"
        assert result["metadata"] == {"engine": "mujoco"}


class TestTraceContextManagement:
    """Functional tests for trace context management."""

    def test_get_returns_none_when_not_set(self):
        """Test that get returns None when not set."""
        from src.api.utils.tracing import (
            _trace_context_var,
            get_trace_context,
        )

        # Reset to default state
        token = _trace_context_var.set(None)
        try:
            result = get_trace_context()
            assert result is None
        finally:
            _trace_context_var.reset(token)

    def test_set_and_get_round_trip(self):
        """Test that set and get work together."""
        from src.api.utils.tracing import (
            TraceContext,
            _trace_context_var,
            get_trace_context,
            set_trace_context,
        )

        context = TraceContext(
            request_id="req_test",
            correlation_id="cor_test",
        )
        token = set_trace_context(context)
        try:
            result = get_trace_context()
            assert result == context
        finally:
            _trace_context_var.reset(token)


class TestRequestTracerContract:
    """Design by Contract tests for RequestTracer middleware."""

    def test_instantiates(self):
        """Postcondition: RequestTracer can be instantiated."""
        from src.api.utils.tracing import RequestTracer

        tracer = RequestTracer()
        assert tracer is not None

    def test_has_trace_request_method(self):
        """Postcondition: RequestTracer has trace_request method."""
        from src.api.utils.tracing import RequestTracer

        tracer = RequestTracer()
        assert hasattr(tracer, "trace_request")
        assert callable(tracer.trace_request)


class TestRequestTracer:
    """Functional tests for RequestTracer middleware."""

    async def test_adds_headers_to_response(self):
        """Test that tracer adds tracing headers to response."""
        from src.api.utils.tracing import (
            CORRELATION_ID_HEADER,
            REQUEST_ID_HEADER,
            RequestTracer,
        )

        tracer = RequestTracer()

        # Mock request
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 200

        # Mock call_next
        async def mock_call_next(request):
            return mock_response

        with patch("src.api.utils.tracing.logger"):
            response = await tracer.trace_request(mock_request, mock_call_next)

        assert REQUEST_ID_HEADER in response.headers
        assert CORRELATION_ID_HEADER in response.headers
        assert "X-Response-Time-Ms" in response.headers

    async def test_preserves_incoming_correlation_id(self):
        """Test that tracer preserves incoming correlation ID."""
        from src.api.utils.tracing import (
            CORRELATION_ID_HEADER,
            RequestTracer,
        )

        tracer = RequestTracer()
        incoming_correlation = "cor_incoming123"

        mock_request = MagicMock()
        mock_request.headers = {CORRELATION_ID_HEADER: incoming_correlation}
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 200

        async def mock_call_next(request):
            return mock_response

        with patch("src.api.utils.tracing.logger"):
            response = await tracer.trace_request(mock_request, mock_call_next)

        assert response.headers[CORRELATION_ID_HEADER] == incoming_correlation

    async def test_handles_exception(self):
        """Test that tracer handles exceptions properly."""
        from src.api.utils.tracing import RequestTracer

        tracer = RequestTracer()

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.method = "POST"
        mock_request.url.path = "/api/error"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        async def mock_call_next_error(request):
            raise ValueError("Test error")

        with patch("src.api.utils.tracing.logger"):
            with pytest.raises(ValueError, match="Test error"):
                await tracer.trace_request(mock_request, mock_call_next_error)


class TestTracedLogContract:
    """Design by Contract tests for traced_log function."""

    def test_does_not_raise(self):
        """Postcondition: traced_log does not raise exceptions."""
        from src.api.utils.tracing import traced_log

        with patch("src.api.utils.tracing.logger"):
            # Should not raise
            traced_log("info", "Test message")
            traced_log("warning", "Warning message", extra_field="value")


class TestTracedLog:
    """Functional tests for traced_log function."""

    def test_injects_request_id(self):
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

    def test_injects_correlation_id_from_context(self):
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

    def test_passes_kwargs_to_extra(self):
        """Test that kwargs are passed to extra."""
        from src.api.utils.tracing import traced_log

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("info", "Test", engine="mujoco", model="arm.urdf")

            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["engine"] == "mujoco"
            assert call_args[1]["extra"]["model"] == "arm.urdf"

    def test_supports_different_log_levels(self):
        """Test that different log levels are supported."""
        from src.api.utils.tracing import traced_log

        with patch("src.api.utils.tracing.logger") as mock_logger:
            traced_log("debug", "Debug message")
            mock_logger.debug.assert_called_once()

            traced_log("warning", "Warning message")
            mock_logger.warning.assert_called_once()

            traced_log("error", "Error message")
            mock_logger.error.assert_called_once()


class TestAllExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test that all __all__ exports are importable."""
        import src.api.utils.tracing as tracing
        from src.api.utils.tracing import __all__

        for name in __all__:
            assert hasattr(tracing, name), f"Missing export: {name}"

    def test_expected_exports_present(self):
        """Test that expected exports are in __all__."""
        from src.api.utils.tracing import __all__

        expected = [
            "CORRELATION_ID_HEADER",
            "REQUEST_ID_HEADER",
            "TraceContext",
            "RequestTracer",
            "generate_request_id",
            "generate_correlation_id",
            "get_request_id",
            "set_request_id",
            "get_trace_context",
            "set_trace_context",
            "traced_log",
        ]

        for name in expected:
            assert name in __all__, f"Missing from __all__: {name}"
