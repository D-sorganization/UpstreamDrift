"""Unit tests for the core API route."""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.core import router
from src.api.dependencies import get_engine_manager

pytestmark = pytest.mark.unit


class MockEngineManager:
    def get_available_engines(self):
        return ["mujoco", "drake"]


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the core router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_root(client: TestClient) -> None:
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["status"] == "running"


def test_routes_core_health_check(client: TestClient) -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engines_available"] == 2
    assert "timestamp" in data


@patch("src.api.routes.core.is_production", return_value=False)
def test_get_diagnostics_unauthenticated_remote_rejected(
    mock_is_prod, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In remote mode and development, unauthenticated requests return 401."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "remote")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()
    response = client.get("/api/diagnostics")
    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()


@patch("src.api.routes.core.is_production", return_value=False)
def test_get_diagnostics_local_mode(
    mock_is_prod, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In local mode, diagnostics endpoint returns measured runtime probes."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()
    response = client.get("/api/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"]["running"] is True
    assert isinstance(data["backend"]["pid"], int)
    assert isinstance(data["backend"]["port"], int)
    assert data["backend"]["port"] == 8000
    assert data["python_found"] is True
    assert data["local_server_found"] is True
    assert data["repo_root"] == "."


@patch("src.api.routes.core.is_production", return_value=True)
def test_get_diagnostics_production(mock_is_prod, client: TestClient) -> None:
    """Test the diagnostics endpoint in production."""
    response = client.get("/api/diagnostics")
    assert response.status_code == 404
