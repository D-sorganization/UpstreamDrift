"""Tests for middleware - Security and upload middleware.

These tests verify the middleware using Design by Contract principles.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response

# Configure async tests
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only."""
    return "asyncio"


@pytest.fixture
def mock_request():
    """Create a mock request."""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.scheme = "https"
    request.headers = {}
    return request


@pytest.fixture
def mock_response():
    """Create a mock response."""
    response = MagicMock(spec=Response)
    response.headers = {}
    return response


class TestAddSecurityHeadersContract:
    """Design by Contract tests for add_security_headers middleware."""

    async def test_returns_response(self, mock_request, mock_response):
        """Postcondition: Returns a Response."""
        from src.api.middleware.security_headers import add_security_headers

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert result is not None


class TestAddSecurityHeaders:
    """Functional tests for add_security_headers middleware."""

    async def test_adds_x_content_type_options(self, mock_request, mock_response):
        """Test adding X-Content-Type-Options header."""
        from src.api.middleware.security_headers import add_security_headers

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert result.headers.get("X-Content-Type-Options") == "nosniff"

    async def test_adds_x_frame_options(self, mock_request, mock_response):
        """Test adding X-Frame-Options header."""
        from src.api.middleware.security_headers import add_security_headers

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert result.headers.get("X-Frame-Options") == "DENY"

    async def test_adds_x_xss_protection(self, mock_request, mock_response):
        """Test adding X-XSS-Protection header."""
        from src.api.middleware.security_headers import add_security_headers

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert result.headers.get("X-XSS-Protection") == "1; mode=block"

    async def test_adds_referrer_policy(self, mock_request, mock_response):
        """Test adding Referrer-Policy header."""
        from src.api.middleware.security_headers import add_security_headers

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert (
            result.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        )

    async def test_adds_hsts_for_https(self, mock_request, mock_response):
        """Test adding HSTS header for HTTPS requests."""
        from src.api.middleware.security_headers import add_security_headers

        mock_request.url.scheme = "https"

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert "Strict-Transport-Security" in result.headers
        assert "max-age=" in result.headers["Strict-Transport-Security"]
        assert "includeSubDomains" in result.headers["Strict-Transport-Security"]

    async def test_no_hsts_for_http(self, mock_request, mock_response):
        """Test not adding HSTS header for HTTP requests."""
        from src.api.middleware.security_headers import add_security_headers

        mock_request.url.scheme = "http"

        async def call_next(request):
            return mock_response

        result = await add_security_headers(mock_request, call_next)
        assert "Strict-Transport-Security" not in result.headers


class TestAddSecurityHeadersToResponseContract:
    """Design by Contract tests for add_security_headers_to_response function."""

    def test_returns_response(self, mock_request, mock_response):
        """Postcondition: Returns the same response with headers added."""
        from src.api.middleware.security_headers import add_security_headers_to_response

        result = add_security_headers_to_response(mock_response, mock_request)
        assert result is mock_response


class TestAddSecurityHeadersToResponse:
    """Functional tests for add_security_headers_to_response function."""

    def test_adds_all_headers(self, mock_request, mock_response):
        """Test adding all security headers."""
        from src.api.middleware.security_headers import add_security_headers_to_response

        mock_request.url.scheme = "https"
        result = add_security_headers_to_response(mock_response, mock_request)

        assert result.headers.get("X-Content-Type-Options") == "nosniff"
        assert result.headers.get("X-Frame-Options") == "DENY"
        assert result.headers.get("X-XSS-Protection") == "1; mode=block"
        assert (
            result.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        )


class TestValidateUploadSizeContract:
    """Design by Contract tests for validate_upload_size middleware."""

    async def test_returns_response(self, mock_request, mock_response):
        """Postcondition: Returns a Response."""
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {}

        async def call_next(request):
            return mock_response

        result = await validate_upload_size(mock_request, call_next)
        assert result is not None


class TestValidateUploadSize:
    """Functional tests for validate_upload_size middleware."""

    async def test_allows_request_without_content_length(
        self, mock_request, mock_response
    ):
        """Test allowing requests without Content-Length header."""
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {}

        async def call_next(request):
            return mock_response

        result = await validate_upload_size(mock_request, call_next)
        assert result is mock_response

    async def test_allows_small_request(self, mock_request, mock_response):
        """Test allowing small requests."""
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {"content-length": "1000"}  # 1KB

        async def call_next(request):
            return mock_response

        result = await validate_upload_size(mock_request, call_next)
        assert result is mock_response

    async def test_rejects_oversized_request(self, mock_request):
        """Test rejecting oversized requests."""
        from src.api.config import MAX_UPLOAD_SIZE_BYTES
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {"content-length": str(MAX_UPLOAD_SIZE_BYTES + 1)}

        async def call_next(request):
            return MagicMock()

        result = await validate_upload_size(mock_request, call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 413

    async def test_rejects_invalid_content_length(self, mock_request):
        """Test rejecting invalid Content-Length header."""
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {"content-length": "not-a-number"}

        async def call_next(request):
            return MagicMock()

        result = await validate_upload_size(mock_request, call_next)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    async def test_adds_security_headers_on_rejection(self, mock_request):
        """Test that security headers are added to rejection responses."""
        from src.api.config import MAX_UPLOAD_SIZE_BYTES
        from src.api.middleware.upload_limits import validate_upload_size

        mock_request.headers = {"content-length": str(MAX_UPLOAD_SIZE_BYTES + 1)}
        mock_request.url = MagicMock()
        mock_request.url.scheme = "https"

        async def call_next(request):
            return MagicMock()

        result = await validate_upload_size(mock_request, call_next)

        # Security headers should be present
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers


class TestUploadLimitsConfig:
    """Tests for upload limits configuration."""

    def test_max_upload_size_bytes_defined(self):
        """Postcondition: MAX_UPLOAD_SIZE_BYTES is defined."""
        from src.api.config import MAX_UPLOAD_SIZE_BYTES

        assert isinstance(MAX_UPLOAD_SIZE_BYTES, int)
        assert MAX_UPLOAD_SIZE_BYTES > 0

    def test_max_upload_size_mb_defined(self):
        """Postcondition: MAX_UPLOAD_SIZE_MB is defined."""
        from src.api.config import MAX_UPLOAD_SIZE_MB

        assert isinstance(MAX_UPLOAD_SIZE_MB, int)
        assert MAX_UPLOAD_SIZE_MB > 0

    def test_consistency_between_mb_and_bytes(self):
        """Postcondition: MB and bytes values are consistent."""
        from src.api.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB

        # Should be approximately consistent (MB * 1024 * 1024 = bytes)
        expected_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        assert MAX_UPLOAD_SIZE_BYTES == expected_bytes
