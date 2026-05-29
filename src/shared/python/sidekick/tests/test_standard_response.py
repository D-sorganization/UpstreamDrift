"""Tests for StandardResponse, ErrorDetail, and related API structures.

These tests verify that the standardized response wrapper correctly handles
success and error responses, metadata tracking, and serialization.

Related to issue #2411 (API Standardization).
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from sidekick.api import (
    ErrorCode,
    ErrorDetail,
    ResponseMetadata,
    StandardResponse,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_error_codes_defined(self) -> None:
        """Verify that all required error codes exist."""
        required_codes = {
            "INVALID_INPUT",
            "NOT_FOUND",
            "SERVER_ERROR",
            "UNSUPPORTED_OPERATION",
            "TIMEOUT",
            "CONSTRAINT_VIOLATION",
        }
        actual_codes = {code.value for code in ErrorCode}
        assert required_codes.issubset(actual_codes)

    def test_error_code_string_value(self) -> None:
        """Verify error codes can be used as strings."""
        code = ErrorCode.INVALID_INPUT
        assert code.value == "INVALID_INPUT"
        assert str(code.value) == "INVALID_INPUT"


class TestErrorDetail:
    """Tests for ErrorDetail class."""

    def test_error_detail_creation(self) -> None:
        """Verify ErrorDetail can be created with required fields."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Test error message",
        )
        assert error.code == ErrorCode.INVALID_INPUT
        assert error.message == "Test error message"
        assert error.details is None
        assert error.request_id is None

    def test_error_detail_with_details(self) -> None:
        """Verify ErrorDetail can include nested error details."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Validation failed",
            details={
                "field": "pipe_diameter_m",
                "constraint": "must be > 0",
                "received": -1.5,
            },
        )
        assert error.details is not None
        assert error.details["field"] == "pipe_diameter_m"
        assert error.details["received"] == -1.5

    def test_error_detail_with_request_id(self) -> None:
        """Verify ErrorDetail can include request ID."""
        request_id = "test-request-123"
        error = ErrorDetail(
            code=ErrorCode.SERVER_ERROR,
            message="Server error",
            request_id=request_id,
        )
        assert error.request_id == request_id

    def test_error_detail_to_dict(self) -> None:
        """Verify ErrorDetail.to_dict() produces correct structure."""
        error = ErrorDetail(
            code=ErrorCode.NOT_FOUND,
            message="Resource not found",
            details={"resource_id": "12345"},
            request_id="req-999",
        )
        error_dict = error.to_dict()
        assert error_dict["code"] == ErrorCode.NOT_FOUND
        assert error_dict["message"] == "Resource not found"
        assert error_dict["details"] == {"resource_id": "12345"}
        assert error_dict["request_id"] == "req-999"


class TestResponseMetadata:
    """Tests for ResponseMetadata class."""

    def test_metadata_creation(self) -> None:
        """Verify ResponseMetadata can be created."""
        metadata = ResponseMetadata(
            request_id="test-req-123",
            processing_time_ms=125.5,
        )
        assert metadata.request_id == "test-req-123"
        assert metadata.processing_time_ms == 125.5
        assert metadata.api_version == "1.0.0"

    def test_metadata_custom_api_version(self) -> None:
        """Verify ResponseMetadata can have custom api_version."""
        metadata = ResponseMetadata(
            request_id="test-req-123",
            processing_time_ms=50.0,
            api_version="2.0.0",
        )
        assert metadata.api_version == "2.0.0"

    def test_metadata_to_dict(self) -> None:
        """Verify ResponseMetadata.to_dict() produces correct structure."""
        metadata = ResponseMetadata(
            request_id="req-001",
            processing_time_ms=200.0,
            api_version="1.0.0",
        )
        metadata_dict = metadata.to_dict()
        assert metadata_dict["request_id"] == "req-001"
        assert metadata_dict["processing_time_ms"] == 200.0
        assert metadata_dict["api_version"] == "1.0.0"


class TestStandardResponseSuccess:
    """Tests for StandardResponse success responses."""

    def test_success_response_creation(self) -> None:
        """Verify success response can be created."""
        data = {"pressure_drop": 1023.4, "velocity": 45.2}
        response = StandardResponse.success(data=data)
        assert response.status == "success"
        assert response.data == data
        assert response.error is None
        assert response.metadata is not None

    def test_success_response_auto_generates_request_id(self) -> None:
        """Verify success response auto-generates request_id if not provided."""
        response = StandardResponse.success(data={"result": 123})
        assert response.metadata.request_id is not None
        # Should be a valid UUID format
        try:
            UUID(response.metadata.request_id)
        except ValueError:
            pytest.fail(f"Invalid UUID: {response.metadata.request_id}")

    def test_success_response_custom_request_id(self) -> None:
        """Verify success response can use custom request_id."""
        custom_id = "custom-request-xyz"
        response = StandardResponse.success(
            data={"result": 123},
            request_id=custom_id,
        )
        assert response.metadata.request_id == custom_id

    def test_success_response_processing_time(self) -> None:
        """Verify processing_time_ms is stored correctly."""
        response = StandardResponse.success(
            data={"result": 456},
            processing_time_ms=125.5,
        )
        assert response.metadata.processing_time_ms == 125.5

    def test_success_response_to_dict(self) -> None:
        """Verify success response.to_dict() produces correct structure."""
        data = {"pressure_drop": 1023.4}
        response = StandardResponse.success(
            data=data,
            processing_time_ms=100.0,
            request_id="req-success-001",
        )
        result = response.to_dict()
        assert result["status"] == "success"
        assert result["data"] == data
        assert result["error"] is None
        assert result["metadata"]["request_id"] == "req-success-001"
        assert result["metadata"]["processing_time_ms"] == 100.0


class TestStandardResponseError:
    """Tests for StandardResponse error responses."""

    def test_error_response_creation(self) -> None:
        """Verify error response can be created."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid input",
        )
        response = StandardResponse.error(error=error)
        assert response.status == "error"
        assert response.data is None
        assert response.error == error
        assert response.metadata is not None

    def test_error_response_auto_generates_request_id(self) -> None:
        """Verify error response auto-generates request_id if error lacks one."""
        error = ErrorDetail(
            code=ErrorCode.SERVER_ERROR,
            message="Server error",
        )
        response = StandardResponse.error(error=error)
        assert error.request_id is not None
        assert response.metadata.request_id == error.request_id

    def test_error_response_preserves_error_request_id(self) -> None:
        """Verify error response uses error's request_id if already set."""
        custom_id = "error-req-123"
        error = ErrorDetail(
            code=ErrorCode.NOT_FOUND,
            message="Not found",
            request_id=custom_id,
        )
        response = StandardResponse.error(error=error)
        assert response.metadata.request_id == custom_id

    def test_error_response_processing_time(self) -> None:
        """Verify processing_time_ms is stored in error response."""
        error = ErrorDetail(
            code=ErrorCode.TIMEOUT,
            message="Request timed out",
        )
        response = StandardResponse.error(
            error=error,
            processing_time_ms=5000.0,
        )
        assert response.metadata.processing_time_ms == 5000.0

    def test_error_response_to_dict(self) -> None:
        """Verify error response.to_dict() produces correct structure."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Validation failed",
            details={"field": "temperature_k", "reason": "must be > 0"},
            request_id="req-error-001",
        )
        response = StandardResponse.error(error=error, processing_time_ms=50.0)
        result = response.to_dict()
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == ErrorCode.INVALID_INPUT
        assert result["error"]["message"] == "Validation failed"
        assert result["error"]["details"]["field"] == "temperature_k"
        assert result["metadata"]["request_id"] == "req-error-001"
        assert result["metadata"]["processing_time_ms"] == 50.0


class TestStandardResponseInit:
    """Tests for StandardResponse.__init__()."""

    def test_init_with_success_status(self) -> None:
        """Verify StandardResponse can be initialized with status='success'."""
        response = StandardResponse(
            status="success",
            data={"result": 123},
        )
        assert response.status == "success"

    def test_init_with_error_status(self) -> None:
        """Verify StandardResponse can be initialized with status='error'."""
        error = ErrorDetail(
            code=ErrorCode.SERVER_ERROR,
            message="Error",
        )
        response = StandardResponse(
            status="error",
            error=error,
        )
        assert response.status == "error"

    def test_init_with_invalid_status(self) -> None:
        """Verify StandardResponse raises ValueError for invalid status."""
        with pytest.raises(ValueError, match='status must be "success" or "error"'):
            StandardResponse(status="invalid", data={})

    def test_init_auto_generates_metadata(self) -> None:
        """Verify StandardResponse auto-generates metadata if not provided."""
        response = StandardResponse(status="success", data={})
        assert response.metadata is not None
        assert response.metadata.request_id is not None
        assert response.metadata.api_version == "1.0.0"

    def test_init_custom_metadata(self) -> None:
        """Verify StandardResponse can use custom metadata."""
        custom_metadata = ResponseMetadata(
            request_id="custom-123",
            processing_time_ms=75.0,
            api_version="2.0.0",
        )
        response = StandardResponse(
            status="success",
            data={},
            metadata=custom_metadata,
        )
        assert response.metadata == custom_metadata


class TestStandardResponseSerialization:
    """Tests for JSON serialization of StandardResponse."""

    def test_success_response_json_serializable(self) -> None:
        """Verify success response can be JSON serialized."""
        response = StandardResponse.success(
            data={"pressure_drop": 1023.4, "velocity": 45.2},
            processing_time_ms=125.0,
            request_id="req-001",
        )
        response_dict = response.to_dict()
        # Should not raise
        json_str = json.dumps(response_dict)
        assert "success" in json_str
        assert "1023.4" in json_str

    def test_error_response_json_serializable(self) -> None:
        """Verify error response can be JSON serialized."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Invalid pipe diameter",
            details={"field": "pipe_diameter_m", "value": -1.0},
            request_id="req-error-001",
        )
        response = StandardResponse.error(error=error, processing_time_ms=50.0)
        response_dict = response.to_dict()
        # Should not raise
        json_str = json.dumps(response_dict)
        assert "error" in json_str
        assert "INVALID_INPUT" in json_str

    def test_roundtrip_success_response(self) -> None:
        """Verify success response can be serialized and deserialized."""
        original_data = {"result": 123.45, "status": "ok"}
        response = StandardResponse.success(
            data=original_data,
            processing_time_ms=75.0,
            request_id="req-roundtrip-001",
        )
        response_dict = response.to_dict()
        json_str = json.dumps(response_dict)
        deserialized = json.loads(json_str)

        assert deserialized["status"] == "success"
        assert deserialized["data"] == original_data
        assert deserialized["error"] is None


class TestStandardResponseRepresentation:
    """Tests for string representation of StandardResponse."""

    def test_repr_success(self) -> None:
        """Verify __repr__() for success response."""
        response = StandardResponse.success(data={"result": 123})
        repr_str = repr(response)
        assert "StandardResponse" in repr_str
        assert "success" in repr_str

    def test_repr_error(self) -> None:
        """Verify __repr__() for error response."""
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Test error",
        )
        response = StandardResponse.error(error=error)
        repr_str = repr(response)
        assert "StandardResponse" in repr_str
        assert "error" in repr_str


class TestStandardResponseIntegration:
    """Integration tests for complete request/response flows."""

    def test_typical_success_flow(self) -> None:
        """Test typical successful API response flow."""
        # Simulate pressure_drop API endpoint
        request_data = {
            "pipe_diameter_m": 0.1,
            "pipe_length_m": 100.0,
            "flow_rate_kg_s": 5.0,
        }

        # Simulate calculation
        calculation_result = {
            "pressure_drop": 1023.4,
            "velocity": 45.2,
            "reynolds_number": 50000.0,
            "friction_factor": 0.025,
        }

        # Create response
        response = StandardResponse.success(
            data=calculation_result,
            processing_time_ms=125.0,
            request_id=f"req-{hash(str(request_data))}",
        )

        # Verify structure
        response_dict = response.to_dict()
        assert response_dict["status"] == "success"
        assert response_dict["data"]["pressure_drop"] == 1023.4
        assert response_dict["metadata"]["processing_time_ms"] == 125.0

    def test_typical_validation_error_flow(self) -> None:
        """Test typical validation error response flow."""
        # Simulate validation failure in API endpoint
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Validation failed for pressure_drop request",
            details={
                "violations": [
                    {
                        "field": "pipe_diameter_m",
                        "constraint": "greater than 0",
                        "value": -1.5,
                    },
                    {
                        "field": "pipe_length_m",
                        "constraint": "greater than 0",
                        "value": 0.0,
                    },
                ]
            },
        )

        # Create error response
        response = StandardResponse.error(
            error=error,
            processing_time_ms=15.0,
        )

        # Verify structure
        response_dict = response.to_dict()
        assert response_dict["status"] == "error"
        assert response_dict["error"]["code"] == ErrorCode.INVALID_INPUT
        assert len(response_dict["error"]["details"]["violations"]) == 2

    def test_typical_server_error_flow(self) -> None:
        """Test typical unexpected server error response flow."""
        # Simulate unexpected error during calculation
        error = ErrorDetail(
            code=ErrorCode.SERVER_ERROR,
            message="Unexpected error during pressure drop calculation",
            details={
                "exception_type": "ValueError",
                "traceback_context": "Stack trace would go here in production",
            },
        )

        response = StandardResponse.error(
            error=error,
            processing_time_ms=200.0,
        )

        response_dict = response.to_dict()
        assert response_dict["status"] == "error"
        assert response_dict["error"]["code"] == ErrorCode.SERVER_ERROR
        assert response_dict["error"]["request_id"] is not None
