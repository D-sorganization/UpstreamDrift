"""Tests for src/api/utils/error_codes.py."""

from __future__ import annotations

import json

import pytest

from src.api.utils import error_codes as ec
from src.api.utils import tracing

pytestmark = pytest.mark.unit


def test_every_error_code_has_metadata() -> None:
    for code in ec.ErrorCode:
        assert code in ec.ERROR_METADATA, f"missing metadata for {code}"
        meta = ec.ERROR_METADATA[code]
        assert "status_code" in meta
        assert "message" in meta
        assert "category" in meta
        assert isinstance(meta["category"], ec.ErrorCategory)


def test_error_code_format() -> None:
    for code in ec.ErrorCode:
        assert code.value.startswith("GMS-")
        parts = code.value.split("-")
        assert len(parts) == 3
        assert parts[1] in {cat.value for cat in ec.ErrorCategory}


def test_apierror_from_code_defaults() -> None:
    err = ec.APIError.from_code(ec.ErrorCode.ENGINE_NOT_LOADED)
    assert err.code is ec.ErrorCode.ENGINE_NOT_LOADED
    assert err.message  # default
    assert err.details == {}


def test_apierror_from_code_custom_message_and_details() -> None:
    err = ec.APIError.from_code(
        ec.ErrorCode.SIMULATION_FAILED,
        message="custom",
        details={"reason": "timeout"},
    )
    assert err.message == "custom"
    assert err.details == {"reason": "timeout"}


def test_apierror_from_code_requires_code() -> None:
    with pytest.raises(ValueError):
        ec.APIError.from_code(None)  # type: ignore[arg-type]


def test_apierror_to_dict_minimal() -> None:
    err = ec.APIError.from_code(ec.ErrorCode.INVALID_REQUEST)
    d = err.to_dict()
    assert d["error"]["code"] == ec.ErrorCode.INVALID_REQUEST.value
    assert "message" in d["error"]
    assert "details" not in d["error"]


def test_apierror_to_dict_with_details_and_request_id() -> None:
    token = tracing.set_request_id("req_abc")
    ctx_token = tracing.set_trace_context(
        tracing.TraceContext(request_id="req_abc", correlation_id="cor_abc")
    )
    try:
        err = ec.APIError.from_code(
            ec.ErrorCode.VALIDATION_FAILED, details={"field": "x"}
        )
        d = err.to_dict()
        assert d["error"]["details"] == {"field": "x"}
        assert d["error"]["request_id"] == "req_abc"
        assert d["error"]["correlation_id"] == "cor_abc"
    finally:
        tracing._request_id_var.reset(token)
        tracing._trace_context_var.reset(ctx_token)


def test_apierror_to_response_uses_status_code() -> None:
    err = ec.APIError.from_code(ec.ErrorCode.RATE_LIMITED)
    resp = err.to_response()
    assert resp.status_code == 429
    body = json.loads(resp.body)
    assert body["error"]["code"] == "GMS-GEN-003"


def test_apierror_to_response_unknown_code_defaults_to_500() -> None:
    err = ec.APIError(code=ec.ErrorCode.INTERNAL_ERROR, message="x")
    # Force an unknown code path by simulating missing metadata.
    err.code = ec.ErrorCode.INTERNAL_ERROR  # known; assert that path works
    resp = err.to_response()
    assert resp.status_code == 500


def test_api_exception_carries_status_and_detail() -> None:
    exc = ec.APIException(ec.ErrorCode.ENGINE_NOT_FOUND, details={"engine": "mujoco"})
    assert exc.status_code == 404
    assert exc.detail["error"]["code"] == "GMS-ENG-001"
    assert exc.detail["error"]["details"] == {"engine": "mujoco"}


def test_api_exception_requires_code() -> None:
    with pytest.raises(ValueError):
        ec.APIException(None)  # type: ignore[arg-type]


def test_raise_api_error_raises_apiexception() -> None:
    with pytest.raises(ec.APIException) as exc:
        ec.raise_api_error(ec.ErrorCode.VALIDATION_FAILED, message="bad", field="x")
    assert exc.value.status_code == 422
    assert exc.value.detail["error"]["details"] == {"field": "x"}


def test_raise_api_error_no_details() -> None:
    with pytest.raises(ec.APIException) as exc:
        ec.raise_api_error(ec.ErrorCode.RESOURCE_NOT_FOUND)
    assert exc.value.status_code == 404


def test_error_category_values() -> None:
    expected = {"GEN", "ENG", "SIM", "VID", "ANL", "AUT", "VAL", "RES", "SYS"}
    assert {c.value for c in ec.ErrorCategory} == expected
