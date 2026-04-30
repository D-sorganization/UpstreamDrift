"""Tests for src.api.utils.error_codes module."""

from __future__ import annotations

import pytest

from src.api.utils.error_codes import (
    ERROR_METADATA,
    APIError,
    APIException,
    ErrorCategory,
    ErrorCode,
    raise_api_error,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_general_category(self) -> None:
        assert ErrorCategory.GENERAL == "GEN"

    def test_engine_category(self) -> None:
        assert ErrorCategory.ENGINE == "ENG"

    def test_all_categories_are_strings(self) -> None:
        for cat in ErrorCategory:
            assert isinstance(cat.value, str)
            assert len(cat.value) == 3


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_internal_error_format(self) -> None:
        assert ErrorCode.INTERNAL_ERROR.value == "GMS-GEN-001"

    def test_all_codes_follow_format(self) -> None:
        """Every error code must follow GMS-XXX-NNN pattern."""
        for code in ErrorCode:
            parts = code.value.split("-")
            assert len(parts) == 3, f"{code.name} doesn't follow GMS-XXX-NNN"
            assert parts[0] == "GMS"

    def test_error_codes_are_unique(self) -> None:
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate error code values found"


class TestErrorMetadata:
    """Tests for ERROR_METADATA mapping."""

    def test_all_codes_have_metadata(self) -> None:
        for code in ErrorCode:
            assert code in ERROR_METADATA, f"{code.name} missing from ERROR_METADATA"

    def test_metadata_has_status_code(self) -> None:
        for code, meta in ERROR_METADATA.items():
            assert "status_code" in meta, f"{code.name} metadata missing status_code"
            assert isinstance(meta["status_code"], int)

    def test_metadata_has_message(self) -> None:
        for code, meta in ERROR_METADATA.items():
            assert "message" in meta, f"{code.name} metadata missing message"
            assert isinstance(meta["message"], str)


class TestAPIError:
    """Tests for APIError dataclass."""

    def test_from_code_defaults(self) -> None:
        error = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        assert error.code == ErrorCode.INTERNAL_ERROR
        assert error.message is not None
        assert len(error.message) > 0

    def test_from_code_custom_message(self) -> None:
        error = APIError.from_code(ErrorCode.INTERNAL_ERROR, message="Custom error")
        assert error.message == "Custom error"

    def test_from_code_with_details(self) -> None:
        error = APIError.from_code(ErrorCode.INVALID_REQUEST, details={"field": "name"})
        assert error.details == {"field": "name"}

    def test_to_dict_structure(self) -> None:
        error = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        d = error.to_dict()
        assert "error" in d
        assert "code" in d["error"]
        assert "message" in d["error"]
        assert d["error"]["code"] == "GMS-GEN-001"

    def test_to_response_returns_json(self) -> None:
        error = APIError.from_code(ErrorCode.INTERNAL_ERROR)
        resp = error.to_response()
        assert resp.status_code == 500

    def test_to_response_404(self) -> None:
        error = APIError.from_code(ErrorCode.RESOURCE_NOT_FOUND)
        resp = error.to_response()
        assert resp.status_code == 404


class TestAPIException:
    """Tests for APIException."""

    def test_inherits_http_exception(self) -> None:
        from fastapi import HTTPException

        exc = APIException(ErrorCode.INTERNAL_ERROR)
        assert isinstance(exc, HTTPException)

    def test_status_code_from_metadata(self) -> None:
        exc = APIException(ErrorCode.INTERNAL_ERROR)
        assert exc.status_code == 500

    def test_detail_is_dict(self) -> None:
        exc = APIException(ErrorCode.INVALID_REQUEST)
        assert isinstance(exc.detail, dict)

    def test_custom_message(self) -> None:
        exc = APIException(ErrorCode.INTERNAL_ERROR, message="boom")
        assert exc.error.message == "boom"


class TestRaiseAPIError:
    """Tests for raise_api_error helper."""

    def test_raises_api_exception(self) -> None:
        with pytest.raises(APIException):
            raise_api_error(ErrorCode.INTERNAL_ERROR)

    def test_raises_with_message(self) -> None:
        with pytest.raises(APIException) as exc_info:
            raise_api_error(ErrorCode.INVALID_REQUEST, message="bad input")
        assert exc_info.value.error.message == "bad input"

    def test_raises_with_details(self) -> None:
        with pytest.raises(APIException) as exc_info:
            raise_api_error(ErrorCode.ENGINE_NOT_FOUND, field="engine_id")
        assert "field" in exc_info.value.error.details
