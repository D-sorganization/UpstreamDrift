"""The local desktop API only serves callers that hold the launcher capability.

The launcher UI reads the capability from ``GET /api/launcher/manifest`` and
sends it back in the ``X-Launcher-CSRF-Token`` header. Health probes, the
manifest itself, tile logos and the API docs pages stay readable without it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
local_server = pytest.importorskip("src.api.local_server")

from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.unit

HEADER = "X-Launcher-CSRF-Token"
OPT_OUT_ENV = "UPSTREAMDRIFT_LOCAL_API_ALLOW_UNAUTHENTICATED"
BASE_URL = "http://127.0.0.1:8000"
GUARDED_PATHS = ("/api/engines", "/api/v1/engines", "/api/launcher/processes")
OPEN_PATHS = ("/api/health", "/api/v1/health", "/api/launcher/manifest")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[object]:
    monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "ui" / "dist"))
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)
    yield local_server.create_local_app()


def _token(app: object) -> str:
    return app.state.launcher_csrf_token  # type: ignore[attr-defined]


@pytest.mark.parametrize("path", GUARDED_PATHS)
def test_request_without_capability_is_refused(app: object, path: str) -> None:
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get(path)
    assert response.status_code == 403
    assert "capability" in response.json()["detail"].lower()


@pytest.mark.parametrize("path", GUARDED_PATHS)
def test_request_with_capability_is_served(app: object, path: str) -> None:
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get(path, headers={HEADER: _token(app)})
    assert response.status_code != 403


@pytest.mark.parametrize("path", OPEN_PATHS)
def test_bootstrap_routes_stay_open(app: object, path: str) -> None:
    with TestClient(app, base_url=BASE_URL) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/api/docs", "/api/launcher/logos/missing.png"])
def test_header_less_browser_routes_stay_open(app: object, path: str) -> None:
    """Page navigations and ``<img>`` requests cannot carry a custom header."""
    with TestClient(app, base_url=BASE_URL) as client:
        assert client.get(path).status_code != 403


def test_wrong_capability_is_refused(app: object) -> None:
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/engines", headers={HEADER: "not-the-token"})
    assert response.status_code == 403


def test_blank_expected_capability_fails_closed(app: object) -> None:
    app.state.launcher_csrf_token = ""  # type: ignore[attr-defined]
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/engines", headers={HEADER: ""})
    assert response.status_code == 403


@pytest.mark.parametrize("header_name", ["origin", "referer"])
def test_capability_with_foreign_origin_is_refused(
    app: object, header_name: str
) -> None:
    headers = {HEADER: _token(app), header_name: "https://example.invalid/page"}
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/engines", headers=headers)
    assert response.status_code == 403


def test_capability_with_loopback_origin_is_served(app: object) -> None:
    headers = {HEADER: _token(app), "origin": "http://127.0.0.1:8000"}
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/engines", headers=headers)
    assert response.status_code != 403


def test_refusal_is_readable_by_an_allowed_browser_origin(app: object) -> None:
    """The guard sits inside CORS, so the UI sees the 403 rather than a
    blocked response."""
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get(
            "/api/engines", headers={"origin": "http://localhost:5173"}
        )
    assert response.status_code == 403
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


class TestOptOut:
    @pytest.fixture
    def opted_out_app(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        caplog: pytest.LogCaptureFixture,
    ) -> Iterator[object]:
        monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "ui" / "dist"))
        monkeypatch.setenv(OPT_OUT_ENV, "1")
        with caplog.at_level(logging.WARNING):
            yield local_server.create_local_app()

    def test_loopback_caller_is_served_without_capability(
        self, opted_out_app: object
    ) -> None:
        with TestClient(opted_out_app, base_url=BASE_URL) as client:
            assert client.get("/api/engines").status_code != 403

    def test_opt_out_is_announced_at_startup(
        self, opted_out_app: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        messages = [
            record.getMessage()
            for record in caplog.get_records("setup")
            if record.levelno >= logging.WARNING
        ]
        assert any(OPT_OUT_ENV in message for message in messages)

    def test_non_loopback_caller_is_still_refused(self, opted_out_app: object) -> None:
        with TestClient(
            opted_out_app, base_url=BASE_URL, client=("203.0.113.5", 50000)
        ) as client:
            assert client.get("/api/engines").status_code == 403

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_other_values_do_not_opt_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, value: str
    ) -> None:
        monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "ui" / "dist"))
        monkeypatch.setenv(OPT_OUT_ENV, value)
        app = local_server.create_local_app()
        with TestClient(app, base_url=BASE_URL) as client:
            assert client.get("/api/engines").status_code == 403
