"""Unit tests for the dataset API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.dataset import router
from src.api.dependencies import get_engine_manager


class MockEngine:
    def __init__(self):
        self.engine_type = "mock_engine"


class MockEngineManager:
    def __init__(self, has_engine=True):
        self.has_engine = has_engine
        self.engine = MockEngine() if has_engine else None

    def get_active_physics_engine(self):
        return self.engine


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the dataset router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_plot_types(client: TestClient) -> None:
    """Test getting plot types.

    The whole body used to sit in `try: ... except Exception: pass`, so this
    test could not fail (#8035). The endpoint is served by the app fixture in
    this module with its dependencies overridden, so there is no optional-import
    hazard to tolerate -- it either responds correctly or the test fails.
    """
    response = client.get("/dataset/plots/types")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, list)


def test_get_export_formats(client: TestClient) -> None:
    """Test getting export formats."""
    response = client.get("/dataset/export/formats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(fmt["format"] == "hdf5" for fmt in data)


def test_generate_no_engine(app: FastAPI) -> None:
    """Test generate dataset fails without active engine."""
    app.dependency_overrides[get_engine_manager] = lambda: MockEngineManager(
        has_engine=False
    )
    client = TestClient(app)
    response = client.post(
        "/dataset/generate", json={"num_samples": 10, "duration": 2.0}
    )
    assert response.status_code == 409
