"""Tests for centralized API error handling decorator (handle_api_errors).

Verifies that handle_api_errors does not leak absolute server paths, exception reprs,
or internals in 4xx bodies, reuses structured error codes from error_codes.py,
and logs original exceptions with tracebacks.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.middleware.error_handler import ClientInputError, handle_api_errors
from src.api.utils.error_codes import ErrorCode

pytestmark = pytest.mark.unit

_SECRET_POSIX_PATH = "/abs/secret/path/config.env"
_SECRET_WIN_PATH = r"C:\Users\secret\proj\.env"
_SECRET_ETC_PATH = "/etc/shadow"


@pytest.fixture
def app_and_client():
    app = FastAPI()

    @app.get("/sync-not-found")
    @handle_api_errors
    def sync_not_found() -> None:
        raise FileNotFoundError(_SECRET_POSIX_PATH)

    @app.get("/async-not-found")
    @handle_api_errors
    async def async_not_found() -> None:
        raise FileNotFoundError(_SECRET_POSIX_PATH)

    @app.get("/sync-permission-denied")
    @handle_api_errors
    def sync_permission_denied() -> None:
        raise PermissionError(_SECRET_ETC_PATH)

    @app.get("/async-permission-denied")
    @handle_api_errors
    async def async_permission_denied() -> None:
        raise PermissionError(_SECRET_ETC_PATH)

    @app.get("/sync-value-error-leak")
    @handle_api_errors
    def sync_value_error_leak() -> None:
        raise ValueError(f"Failed to load {_SECRET_WIN_PATH}: invalid header")

    @app.get("/async-value-error-leak")
    @handle_api_errors
    async def async_value_error_leak() -> None:
        raise ValueError(f"Failed to load {_SECRET_WIN_PATH}: invalid header")

    @app.get("/sync-unicode-error")
    @handle_api_errors
    def sync_unicode_error() -> None:
        b"\xff".decode("utf-8")

    @app.get("/sync-client-input-error")
    @handle_api_errors
    def sync_client_input_error() -> None:
        raise ClientInputError("Metric 'launch_angle' not found")

    @app.get("/sync-safe-value-error")
    @handle_api_errors
    def sync_safe_value_error() -> None:
        raise ValueError("Invalid metric name specified")

    @app.get("/sync-timeout-error")
    @handle_api_errors
    def sync_timeout_error() -> None:
        raise TimeoutError("Simulation engine hung")

    @app.get("/sync-unexpected-error")
    @handle_api_errors
    def sync_unexpected_error() -> None:
        class ArbitraryDomainError(Exception):
            pass

        raise ArbitraryDomainError("unexpected failure")

    @app.get("/sync-http-exception")
    @handle_api_errors
    def sync_http_exception() -> None:
        raise HTTPException(status_code=418, detail="I'm a teapot")

    client = TestClient(app, raise_server_exceptions=False)
    return app, client


@pytest.mark.parametrize(
    ("path", "expected_status", "forbidden_string", "expected_code"),
    [
        (
            "/sync-not-found",
            404,
            _SECRET_POSIX_PATH,
            ErrorCode.RESOURCE_NOT_FOUND.value,
        ),
        (
            "/async-not-found",
            404,
            _SECRET_POSIX_PATH,
            ErrorCode.RESOURCE_NOT_FOUND.value,
        ),
        (
            "/sync-permission-denied",
            403,
            _SECRET_ETC_PATH,
            ErrorCode.RESOURCE_ACCESS_DENIED.value,
        ),
        (
            "/async-permission-denied",
            403,
            _SECRET_ETC_PATH,
            ErrorCode.RESOURCE_ACCESS_DENIED.value,
        ),
        (
            "/sync-value-error-leak",
            400,
            _SECRET_WIN_PATH,
            ErrorCode.INVALID_REQUEST.value,
        ),
        (
            "/async-value-error-leak",
            400,
            _SECRET_WIN_PATH,
            ErrorCode.INVALID_REQUEST.value,
        ),
        (
            "/sync-unicode-error",
            400,
            "codec can't decode",
            ErrorCode.INVALID_REQUEST.value,
        ),
    ],
)
def test_handle_api_errors_sanitizes_4xx_details(
    app_and_client: tuple[FastAPI, TestClient],
    caplog: pytest.LogCaptureFixture,
    path: str,
    expected_status: int,
    forbidden_string: str,
    expected_code: str,
) -> None:
    """Non-client-safe exceptions must not leak absolute paths or exception reprs in 4xx bodies."""
    _, client = app_and_client
    with caplog.at_level(logging.WARNING):
        response = client.get(path)

    assert response.status_code == expected_status
    body_text = response.text
    assert forbidden_string not in body_text

    body_json = response.json()
    assert body_json["detail"]["error"]["code"] == expected_code

    # Verify that original exception was logged with traceback (exc_info=True)
    records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.exc_info is not None
    ]
    assert len(records) >= 1, (
        "Original exception must be logged at WARNING or higher with exc_info"
    )


def test_client_input_error_preserves_safe_message(
    app_and_client: tuple[FastAPI, TestClient],
) -> None:
    """ClientInputError message is retained in 400 detail."""
    _, client = app_and_client
    response = client.get("/sync-client-input-error")
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert "launch_angle" in body["detail"]["error"]["message"]


def test_safe_value_error_preserves_message(
    app_and_client: tuple[FastAPI, TestClient],
) -> None:
    """Safe ValueError without paths or codec errors preserves message."""
    _, client = app_and_client
    response = client.get("/sync-safe-value-error")
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert "Invalid metric name specified" in body["detail"]["error"]["message"]


def test_timeout_error_maps_to_504(
    app_and_client: tuple[FastAPI, TestClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TimeoutError maps to 504 Gateway Timeout rather than 500 Internal Server Error."""
    _, client = app_and_client
    with caplog.at_level(logging.WARNING):
        response = client.get("/sync-timeout-error")

    assert response.status_code == 504
    body = response.json()
    assert body["detail"]["error"]["code"] == ErrorCode.SIMULATION_TIMEOUT.value

    records = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.exc_info is not None
    ]
    assert len(records) >= 1


def test_unexpected_error_fallback_maps_to_500(
    app_and_client: tuple[FastAPI, TestClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected exception types outside common tuple reach exhaustive 500 fallback."""
    _, client = app_and_client
    with caplog.at_level(logging.ERROR):
        response = client.get("/sync-unexpected-error")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"]["error"]["code"] == ErrorCode.INTERNAL_ERROR.value
    assert "Internal server error" in body["detail"]["error"]["message"]


def test_http_exception_reraises_verbatim(
    app_and_client: tuple[FastAPI, TestClient],
) -> None:
    """HTTPException passes through unchanged."""
    _, client = app_and_client
    response = client.get("/sync-http-exception")
    assert response.status_code == 418
    assert response.json()["detail"] == "I'm a teapot"
