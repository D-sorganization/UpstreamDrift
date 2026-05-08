"""Unit tests for the core API route."""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.core import router
from src.api.dependencies import get_engine_manager


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


def test_health_check(client: TestClient) -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engines_available"] == 2
    assert "timestamp" in data


@patch("src.api.routes.core.is_production", return_value=False)
def test_get_diagnostics_development(mock_is_prod, client: TestClient) -> None:
    """Test the diagnostics endpoint in development."""
    response = client.get("/api/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"]["running"] is True
    assert data["python_found"] is True


@patch("src.api.routes.core.is_production", return_value=True)
def test_get_diagnostics_production(mock_is_prod, client: TestClient) -> None:
    """Test the diagnostics endpoint in production."""
    response = client.get("/api/diagnostics")
    assert response.status_code == 404
