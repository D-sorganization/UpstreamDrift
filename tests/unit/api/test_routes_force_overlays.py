"""Unit tests for the force overlays API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.force_overlays import router
from src.api.dependencies import get_engine_manager


class MockEngine:
    def __init__(self):
        self.time = 0.5
        self.joint_names = ["base", "arm"]

    def get_state(self):
        return {"time": 0.5, "torques": [10.0, -5.0]}


class MockEngineManager:
    def __init__(self, has_engine=True):
        self.engine = MockEngine() if has_engine else None

    def get_active_engine(self):
        return self.engine


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the force overlays router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_force_overlays(client: TestClient) -> None:
    """Test getting force overlays."""
    response = client.get("/simulation/forces?force_types=applied,gravity")
    assert response.status_code == 200
    data = response.json()
    assert "vectors" in data
    assert data["sim_time"] == 0.5

    # Check that vectors have correct format
    vectors = data["vectors"]
    assert len(vectors) > 0
    assert "body_name" in vectors[0]
    assert "force_type" in vectors[0]


def test_update_force_overlay_config(client: TestClient) -> None:
    """Test updating force overlay config."""
    payload = {
        "enabled": True,
        "force_types": ["applied"],
        "color_by_magnitude": True,
        "show_labels": True,
        "scale_factor": 0.05,
    }
    response = client.post("/simulation/forces/config", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overlay_config"]["scale_factor"] == 0.05
    assert data["overlay_config"]["show_labels"] is True


def test_demo_vectors_without_engine(app: FastAPI) -> None:
    """Test fallback to demo vectors when no engine is loaded."""
    app.dependency_overrides[get_engine_manager] = lambda: MockEngineManager(
        has_engine=False
    )
    client = TestClient(app)
    response = client.get("/simulation/forces?force_types=all")
    assert response.status_code == 200
    data = response.json()
    assert len(data["vectors"]) > 0
