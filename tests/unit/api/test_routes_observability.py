"""Unit tests for the observability API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.observability import router


class MockEngineManager:
    def get_available_engines(self):
        return ["mujoco", "drake"]


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the observability router."""
    test_app = FastAPI()
    test_app.include_router(router)

    # Setup required state
    test_app.state.engine_manager = MockEngineManager()
    test_app.state.simulation_service = True
    test_app.state.analysis_service = True
    test_app.state.task_manager = True
    test_app.state.api_started_at = 1000.0

    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    """Test healthz endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readyz_success(client: TestClient) -> None:
    """Test readyz endpoint when ready."""
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["engines_available"] == 2


def test_readyz_not_ready(app: FastAPI) -> None:
    """Test readyz endpoint when missing dependencies."""
    delattr(app.state, "task_manager")
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "task_manager" in data["missing"]


def test_metrics(client: TestClient) -> None:
    """Test prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "upstreamdrift_api_info" in text
    assert "upstreamdrift_api_ready 1" in text
    assert "upstreamdrift_api_engines_available 2" in text
