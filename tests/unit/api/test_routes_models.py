"""Unit tests for the models API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.models import router


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the models router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_list_models(client: TestClient) -> None:
    """Test listing models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    # simple_pendulum should be in the list
    model_names = [m["name"] for m in data["models"]]
    assert "simple_pendulum" in model_names


def test_get_model_urdf(client: TestClient) -> None:
    """Test getting parsed URDF data."""
    response = client.get("/models/simple_pendulum/urdf")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "simple_pendulum"
    assert "links" in data
    assert "joints" in data
    assert "root_link" in data
    assert "urdf_raw" in data


def test_get_model_urdf_not_found(client: TestClient) -> None:
    """Test getting parsed URDF data for non-existent model."""
    response = client.get("/models/unknown_model/urdf")
    assert response.status_code == 404
