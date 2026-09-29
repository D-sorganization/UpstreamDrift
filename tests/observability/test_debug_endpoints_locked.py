"""Production guard tests for local diagnostic/debug endpoints."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

local_server = pytest.importorskip("src.api.local_server")


DEBUG_ENDPOINTS = (
    "/api/diagnostics",
    "/api/diagnostics/html",
    "/api/debug/routes",
    "/api/debug/static",
)


@pytest.fixture(autouse=True)
def reset_local_server_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset env and startup metrics around local app construction."""
    monkeypatch.delenv("UPSTREAM_DRIFT_ENV", raising=False)
    monkeypatch.delenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", raising=False)
    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )
    yield


def test_debug_endpoints_are_not_registered_in_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Production mode must not expose diagnostic/debug endpoints by default."""
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "missing-ui"))

    app = local_server.create_local_app()

    with TestClient(app, base_url="http://localhost") as client:
        for endpoint in DEBUG_ENDPOINTS:
            response = client.get(endpoint)
            assert response.status_code == 404


def test_debug_endpoints_can_be_explicitly_enabled_in_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Operators can opt in to local diagnostic routes for controlled debugging."""
    monkeypatch.setenv("UPSTREAM_DRIFT_ENV", "production")
    monkeypatch.setenv("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", "true")
    monkeypatch.setenv("GOLF_UI_DIST", str(tmp_path / "missing-ui"))

    app = local_server.create_local_app()

    with TestClient(app, base_url="http://localhost") as client:
        response = client.get("/api/debug/routes")
        assert response.status_code == 200
