"""Tests for launcher-side Sidekick API readiness gating."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.launchers import sidekick_readiness as readiness


class _FakeHttpResponse:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self._body = body

    def read(self, _limit: int) -> bytes:
        return self._body


class _FakeHttpConnection:
    status = 200
    body = b'{"status":"ready"}'
    requests: list[tuple[str, str]] = []
    closed = False

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, method: str, path: str) -> None:
        self.requests.append((method, path))

    def getresponse(self) -> _FakeHttpResponse:
        return _FakeHttpResponse(self.status, self.body)

    def close(self) -> None:
        self.closed = True


def test_check_sidekick_api_readiness_reports_ready(monkeypatch) -> None:
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8123")
    monkeypatch.setattr(readiness, "HTTPConnection", _FakeHttpConnection)
    _FakeHttpConnection.status = 200
    _FakeHttpConnection.requests = []

    result = readiness.check_sidekick_api_readiness()

    assert result.ready is True
    assert result.url == "http://127.0.0.1:8123/readyz"
    assert _FakeHttpConnection.requests == [("GET", "/readyz")]


def test_check_sidekick_api_readiness_reports_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8123")
    monkeypatch.setattr(readiness, "HTTPConnection", _FakeHttpConnection)
    _FakeHttpConnection.status = 503
    _FakeHttpConnection.body = b'{"status":"not_ready"}'

    result = readiness.check_sidekick_api_readiness()

    assert result.ready is False
    assert result.status_code == 503
    assert "not_ready" in result.detail


def test_check_sidekick_api_readiness_reports_invalid_port(monkeypatch) -> None:
    monkeypatch.setenv("API_PORT", "not-a-port")

    result = readiness.check_sidekick_api_readiness()

    assert result.ready is False
    assert "Invalid API_PORT" in result.detail


def test_readiness_detail_for_log_truncates_body() -> None:
    result = readiness.SidekickApiReadiness(
        ready=False,
        url="http://127.0.0.1:8000/readyz",
        status_code=503,
        detail="x" * 400,
    )

    detail: dict[str, Any] = readiness.readiness_detail_for_log(result)

    assert detail == {
        "ready": False,
        "url": "http://127.0.0.1:8000/readyz",
        "status_code": 503,
        "detail": "x" * 300,
    }
