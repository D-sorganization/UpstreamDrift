"""Unit tests for the model explorer API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.model_explorer import router


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the model explorer router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_inspect_model_success(client: TestClient) -> None:
    """Test inspecting a valid URDF model."""
    payload = {"model_path": "simple_pendulum.urdf"}
    response = client.post("/tools/model-explorer/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_format"] == "urdf"
    assert "tree" in data
    assert len(data["tree"]) > 0
    # There should be links and joints in the tree
    node_types = [node["node_type"] for node in data["tree"]]
    assert "root" in node_types or "link" in node_types


def test_inspect_model_not_found(client: TestClient) -> None:
    """Test inspecting a non-existent model."""
    payload = {"model_path": "non_existent_model.urdf"}
    response = client.post("/tools/model-explorer/inspect", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()


def test_compare_models_success(client: TestClient) -> None:
    """Test comparing two models (Frankenstein mode)."""
    payload = {
        "model_a_path": "simple_pendulum.urdf",
        "model_b_path": "double_pendulum.urdf",
    }
    response = client.post("/tools/model-explorer/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model_a" in data
    assert "model_b" in data
    assert "shared_joints" in data
    assert "unique_to_a" in data
    assert "unique_to_b" in data
    assert isinstance(data["shared_joints"], list)
