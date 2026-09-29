"""Production gating contracts for local diagnostics endpoints."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

DIAGNOSTIC_PATHS = (
    "/api/diagnostics",
    "/api/diagnostics/html",
    "/api/debug/routes",
    "/api/debug/static",
)


@pytest.fixture
def production_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create the local app while production mode is active."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()

    from src.api.local_server import create_local_app

    with TestClient(create_local_app()) as test_client:
        yield test_client

    environment.get_environment.cache_clear()


@pytest.mark.parametrize("path", DIAGNOSTIC_PATHS)
def test_local_diagnostics_are_not_exposed_in_production(
    production_client: TestClient, path: str
) -> None:
    """Local diagnostics and debug routes are disabled in production."""
    response = production_client.get(path)

    assert response.status_code == 404


def test_core_diagnostics_is_not_exposed_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main API browser diagnostics route is disabled in production."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()

    from src.api.routes.core import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as test_client:
        response = test_client.get("/api/diagnostics")

    assert response.status_code == 404
    environment.get_environment.cache_clear()
