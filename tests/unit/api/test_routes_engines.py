"""Unit tests for the engines API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.engines import router
from src.api.dependencies import get_engine_manager
from src.shared.python.engine_core.engine_registry import EngineType


class MockEngineCapabilities:
    def to_dict(self):
        return {
            "engine_name": "MuJoCo",
            "spatial_jacobian_order": "linear",
            "physics": "full",
            "contacts": "partial",
            "muscles": "none",
        }


class MockEngine:
    def get_capabilities(self):
        return MockEngineCapabilities()

    def load_from_path(self, path):
        pass

    def get_state(self):
        return {"time": 0.0}


class MockEngineManager:
    def get_available_engines(self):
        return [EngineType.MUJOCO, EngineType.DRAKE, EngineType.JAXSIM]

    def get_current_engine(self):
        return EngineType.MUJOCO

    def get_engine_status(self, engine_type):
        from enum import Enum

        class Status(Enum):
            LOADED = "loaded"
            AVAILABLE = "available"
            UNAVAILABLE = "unavailable"

        if engine_type == EngineType.MUJOCO:
            return Status.LOADED
        return Status.AVAILABLE

    def switch_engine(self, engine_type):
        return True

    def get_active_physics_engine(self):
        return MockEngine()


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the engines router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_routes_engines_get_engines(client: TestClient) -> None:
    """Test getting all engines."""
    response = client.get("/engines")
    assert response.status_code == 200
    data = response.json()
    assert "engines" in data
    assert "mode" in data
    assert len(data["engines"]) > 0


def test_routes_engines_surfaces_jaxsim_capabilities(client: TestClient) -> None:
    """JaxSim appears in /engines with differentiable-analysis capability tags."""
    response = client.get("/engines")
    assert response.status_code == 200
    engines = response.json()["engines"]

    jaxsim = next(engine for engine in engines if engine["name"] == "jaxsim")
    assert jaxsim["available"] is True
    assert jaxsim["capabilities"] == [
        "rigid_body",
        "differentiable",
        "gradients",
        "parameter_sensitivity",
    ]


def test_load_engine(client: TestClient) -> None:
    """Test loading an engine."""
    response = client.post("/engines/mujoco/load")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loaded"
    assert data["engine"] == "mujoco"


def test_load_unknown_engine(client: TestClient) -> None:
    """Test loading an unknown engine."""
    response = client.post("/engines/unknown/load")
    assert response.status_code == 400


def test_get_engine_capabilities(client: TestClient) -> None:
    """Test getting capabilities for a specific engine."""
    response = client.get("/engines/mujoco/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_type"] == "mujoco"
    assert "capabilities" in data
    assert "summary" in data

    # Assert filtering works correctly
    cap_names = [c["name"] for c in data["capabilities"]]
    assert "physics" in cap_names
    assert "contacts" in cap_names
    assert "spatial_jacobian_order" not in cap_names
