"""Tests for error_codes - Structured API error handling.

These tests verify the error code system using Design by Contract
principles to ensure consistent error responses across the API.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException


class TestErrorCategoryContract:
    """Design by Contract tests for ErrorCategory enum."""

    def test_categories_are_string_enum(self):
        """Postcondition: All categories are string enum values."""
        from src.api.utils.error_codes import ErrorCategory

        for category in ErrorCategory:
            assert isinstance(category.value, str)
            assert len(category.value) == 3  # 3-letter category codes

    def test_all_categories_unique(self):
        """Postcondition: All category values must be unique."""
        from src.api.utils.error_codes import ErrorCategory

        values = [c.value for c in ErrorCategory]
        assert len(values) == len(set(values))


class TestErrorCodeContract:
    """Design by Contract tests for ErrorCode enum."""

    def test_codes_follow_format(self):
        """Postcondition: All codes follow GMS-XXX-NNN format."""
        import re

        from src.api.utils.error_codes import ErrorCode

        pattern = r"^GMS-[A-Z]{3}-\d{3}$"

        for code in ErrorCode:
            assert re.match(pattern, code.value), f"{code.value} doesn't match format"

    def test_all_codes_unique(self):
        """Postcondition: All error code values must be unique."""
        from src.api.utils.error_codes import ErrorCode

        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values))

    def test_all_codes_have_metadata(self):
        """Postcondition: Every error code must have metadata defined."""
        from src.api.utils.error_codes import ERROR_METADATA, ErrorCode

        for code in ErrorCode:
            assert code in ERROR_METADATA, f"Missing metadata for {code}"
            metadata = ERROR_METADATA[code]
            assert "status_code" in metadata
            assert "message" in metadata
            assert "category" in metadata


class TestErrorMetadata:
    """Tests for ERROR_METADATA configuration."""

    def test_status_codes_are_valid_http(self):
        """Test that all status codes are valid HTTP status codes."""
        from src.api.utils.error_codes import ERROR_METADATA

        valid_status_codes = {
            400,
            401,
            403,
            404,
            409,
            422,
            429,
            500,
            503,
            504,
        }

        for code, metadata in ERROR_METADATA.items():
            status = metadata.get("status_code")
            assert status in valid_status_codes, (
                f"Invalid status code {status} for {code}"
            )

    def test_messages_are_non_empty_strings(self):
        """Test that all messages are non-empty strings."""
        from src.api.utils.error_codes import ERROR_METADATA

        for metadata in ERROR_METADATA.values():
            message = metadata.get("message")
            assert isinstance(message, str)
            assert len(message) > 0

    def test_categories_match_code_prefix(self):
        """Test that category matches the code prefix."""
        from src.api.utils.error_codes import ERROR_METADATA, ErrorCategory

        code_to_category = {
            "GEN": ErrorCategory.GENERAL,
            "ENG": ErrorCategory.ENGINE,
            "SIM": ErrorCategory.SIMULATION,
            "VID": ErrorCategory.VIDEO,
            "ANL": ErrorCategory.ANALYSIS,
            "AUT": ErrorCategory.AUTH,
            "VAL": ErrorCategory.VALIDATION,
            "RES": ErrorCategory.RESOURCE,
            "SYS": ErrorCategory.SYSTEM,
        }

        for code, metadata in ERROR_METADATA.items():
            # Extract category from code (e.g., "GMS-ENG-001" -> "ENG")
            code_prefix = code.value.split("-")[1]
            expected_category = code_to_category.get(code_prefix)

            if expected_category:
                assert metadata.get("category") == expected_category, (
                    f"Category mismatch for {code}"
                )


class TestAPIErrorContract:
    """Design by Contract tests for APIError dataclass."""

    def test_from_code_returns_api_error(self):
        """Postcondition: from_code returns an APIError instance."""
        from src.api.utils.error_codes import APIError, ErrorCode

        result = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        assert isinstance(result, APIError)

    def test_to_dict_returns_dict(self):
        """Postcondition: to_dict returns a dictionary."""
        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        result = error.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_fields(self):
        """Postcondition: to_dict includes error code and message."""
        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError.from_code(ErrorCode.INVALID_REQUEST)
        result = error.to_dict()

        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]


class TestAPIError:
    """Functional tests for APIError dataclass."""

    def test_from_code_uses_default_message(self):
        """Test that from_code uses default message when none provided."""
        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError.from_code(ErrorCode.ENGINE_NOT_FOUND)

        assert "engine" in error.message.lower() or "not found" in error.message.lower()

    def test_from_code_accepts_custom_message(self):
        """Test that from_code accepts custom message."""
        from src.api.utils.error_codes import APIError, ErrorCode

        custom_message = "Custom error message"
        error = APIError.from_code(ErrorCode.INTERNAL_ERROR, message=custom_message)

        assert error.message == custom_message

    def test_from_code_accepts_details(self):
        """Test that from_code accepts additional details."""
        from src.api.utils.error_codes import APIError, ErrorCode

        details = {"engine": "mujoco", "reason": "timeout"}
        error = APIError.from_code(ErrorCode.ENGINE_LOAD_FAILED, details=details)

        assert error.details == details

    def test_to_dict_includes_details_when_present(self):
        """Test that to_dict includes details when present."""
        from src.api.utils.error_codes import APIError, ErrorCode

        details = {"field": "username", "reason": "required"}
        error = APIError.from_code(ErrorCode.VALIDATION_MISSING_FIELD, details=details)
        result = error.to_dict()

        assert result["error"]["details"] == details

    def test_to_dict_omits_details_when_empty(self):
        """Test that to_dict omits details when empty."""
        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        result = error.to_dict()

        assert "details" not in result["error"]

    def test_to_dict_includes_request_id_when_set(self):
        """Test that to_dict includes request_id when present."""
        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Test error",
            request_id="req_abc123",
        )
        result = error.to_dict()

        assert result["error"]["request_id"] == "req_abc123"

    def test_to_dict_omits_request_id_when_empty(self):
        """Test that to_dict omits request_id when empty."""
        from src.api.utils.error_codes import APIError, ErrorCode

        with (
            patch("src.api.utils.error_codes.get_request_id", return_value=""),
            patch("src.api.utils.error_codes.get_trace_context", return_value=None),
        ):
            error = APIError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Test",
                request_id="",
            )
            result = error.to_dict()
            assert "request_id" not in result["error"]

    def test_to_response_returns_json_response(self):
        """Test that to_response returns a JSONResponse."""
        from fastapi.responses import JSONResponse

        from src.api.utils.error_codes import APIError, ErrorCode

        error = APIError.from_code(ErrorCode.VALIDATION_FAILED)
        response = error.to_response()

        assert isinstance(response, JSONResponse)

    def test_to_response_uses_correct_status_code(self):
        """Test that to_response uses the correct status code."""
        from src.api.utils.error_codes import APIError, ErrorCode

        test_cases = [
            (ErrorCode.INTERNAL_ERROR, 500),
            (ErrorCode.INVALID_REQUEST, 400),
            (ErrorCode.RATE_LIMITED, 429),
            (ErrorCode.RESOURCE_NOT_FOUND, 404),
            (ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS, 403),
            (ErrorCode.VALIDATION_FAILED, 422),
        ]

        for code, expected_status in test_cases:
            error = APIError.from_code(code)
            response = error.to_response()
            assert response.status_code == expected_status

    def test_post_init_injects_trace_context(self):
        """Test that __post_init__ injects trace context."""
        from src.api.utils.error_codes import APIError, ErrorCode
        from src.api.utils.tracing import TraceContext

        mock_context = TraceContext(
            request_id="req_test123",
            correlation_id="cor_test456",
        )

        with (
            patch(
                "src.api.utils.error_codes.get_request_id", return_value="req_test123"
            ),
            patch(
                "src.api.utils.error_codes.get_trace_context", return_value=mock_context
            ),
        ):
            error = APIError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Test",
            )
            assert error.request_id == "req_test123"
            assert error.correlation_id == "cor_test456"


class TestAPIExceptionContract:
    """Design by Contract tests for APIException."""

    def test_inherits_from_http_exception(self):
        """Postcondition: APIException inherits from HTTPException."""
        from src.api.utils.error_codes import APIException, ErrorCode

        exc = APIException(ErrorCode.INTERNAL_ERROR)
        assert isinstance(exc, HTTPException)

    def test_has_error_attribute(self):
        """Postcondition: APIException has error attribute."""
        from src.api.utils.error_codes import APIException, ErrorCode

        exc = APIException(ErrorCode.INTERNAL_ERROR)
        assert hasattr(exc, "error")


class TestAPIException:
    """Functional tests for APIException."""

    def test_status_code_from_metadata(self):
        """Test that status code comes from error metadata."""
        from src.api.utils.error_codes import APIException, ErrorCode

        test_cases = [
            (ErrorCode.INTERNAL_ERROR, 500),
            (ErrorCode.ENGINE_NOT_FOUND, 404),
            (ErrorCode.VALIDATION_FAILED, 422),
            (ErrorCode.AUTH_TOKEN_INVALID, 401),
        ]

        for code, expected_status in test_cases:
            exc = APIException(code)
            assert exc.status_code == expected_status

    def test_detail_is_structured(self):
        """Test that detail is a structured error dict."""
        from src.api.utils.error_codes import APIException, ErrorCode

        exc = APIException(ErrorCode.ENGINE_LOAD_FAILED)

        assert isinstance(exc.detail, dict)
        assert "error" in exc.detail
        assert "code" in exc.detail["error"]

    def test_custom_message_used(self):
        """Test that custom message is used when provided."""
        from src.api.utils.error_codes import APIException, ErrorCode

        custom = "Custom error message"
        exc = APIException(ErrorCode.INTERNAL_ERROR, message=custom)

        assert exc.error.message == custom

    def test_details_included(self):
        """Test that details are included in error."""
        from src.api.utils.error_codes import APIException, ErrorCode

        details = {"engine": "drake", "model": "arm.urdf"}
        exc = APIException(ErrorCode.SIMULATION_FAILED, details=details)

        assert exc.error.details == details


class TestRaiseApiErrorContract:
    """Design by Contract tests for raise_api_error function."""

    def test_always_raises(self):
        """Postcondition: raise_api_error always raises APIException."""
        from src.api.utils.error_codes import APIException, ErrorCode, raise_api_error

        with pytest.raises(APIException):
            raise_api_error(ErrorCode.INTERNAL_ERROR)


class TestRaiseApiError:
    """Functional tests for raise_api_error function."""

    def test_raises_with_correct_code(self):
        """Test that raise_api_error raises with correct code."""
        from src.api.utils.error_codes import APIException, ErrorCode, raise_api_error

        with pytest.raises(APIException) as exc_info:
            raise_api_error(ErrorCode.ENGINE_NOT_AVAILABLE)

        assert exc_info.value.error.code == ErrorCode.ENGINE_NOT_AVAILABLE

    def test_raises_with_custom_message(self):
        """Test that raise_api_error uses custom message."""
        from src.api.utils.error_codes import APIException, ErrorCode, raise_api_error

        custom = "MuJoCo engine failed to initialize"
        with pytest.raises(APIException) as exc_info:
            raise_api_error(ErrorCode.ENGINE_INITIALIZATION_FAILED, message=custom)

        assert exc_info.value.error.message == custom

    def test_raises_with_kwargs_as_details(self):
        """Test that raise_api_error passes kwargs as details."""
        from src.api.utils.error_codes import APIException, ErrorCode, raise_api_error

        with pytest.raises(APIException) as exc_info:
            raise_api_error(
                ErrorCode.SIMULATION_INVALID_PARAMS,
                timestep=-0.001,
                field="timestep",
                reason="must be positive",
            )

        details = exc_info.value.error.details
        assert details["timestep"] == -0.001
        assert details["field"] == "timestep"
        assert details["reason"] == "must be positive"

    def test_no_details_when_no_kwargs(self):
        """Test that no details when no kwargs provided."""
        from src.api.utils.error_codes import APIException, ErrorCode, raise_api_error

        with pytest.raises(APIException) as exc_info:
            raise_api_error(ErrorCode.INTERNAL_ERROR)

        assert exc_info.value.error.details == {}


class TestAllExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test that all __all__ exports are importable."""
        import src.api.utils.error_codes as ec
        from src.api.utils.error_codes import __all__

        for name in __all__:
            assert hasattr(ec, name), f"Missing export: {name}"

    def test_expected_exports_present(self):
        """Test that expected exports are in __all__."""
        from src.api.utils.error_codes import __all__

        expected = [
            "ErrorCategory",
            "ErrorCode",
            "ERROR_METADATA",
            "APIError",
            "APIException",
            "raise_api_error",
        ]

        for name in expected:
            assert name in __all__, f"Missing from __all__: {name}"
