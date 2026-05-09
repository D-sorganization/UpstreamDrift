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


class TestRequestTracer:
    """Functional tests for RequestTracer middleware."""

    async def test_adds_headers_to_response(self) -> None:
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
        async def mock_call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("src.api.utils.tracing.logger"):
            response = await tracer.trace_request(mock_request, mock_call_next)

        assert REQUEST_ID_HEADER in response.headers
        assert CORRELATION_ID_HEADER in response.headers
        assert "X-Response-Time-Ms" in response.headers

    async def test_preserves_incoming_correlation_id(self) -> None:
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

        async def mock_call_next(request: MagicMock) -> MagicMock:
            return mock_response

        with patch("src.api.utils.tracing.logger"):
            response = await tracer.trace_request(mock_request, mock_call_next)

        assert response.headers[CORRELATION_ID_HEADER] == incoming_correlation

    async def test_handles_exception(self) -> None:
        """Test that tracer handles exceptions properly."""
        from src.api.utils.tracing import RequestTracer

        tracer = RequestTracer()

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.method = "POST"
        mock_request.url.path = "/api/error"
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        async def mock_call_next_error(request) -> NoReturn:
            raise ValueError("Test error")

        with (
            patch("src.api.utils.tracing.logger"),
            pytest.raises(ValueError, match="Test error"),
        ):
            await tracer.trace_request(mock_request, mock_call_next_error)
