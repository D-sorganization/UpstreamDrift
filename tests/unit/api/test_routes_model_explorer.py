"""Unit tests for the model explorer API route."""

from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.routes import model_explorer
from src.api.routes.model_explorer import router

pytestmark = pytest.mark.unit


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


def test_resolve_model_path_rejects_existing_absolute_path_outside_allowed_dirs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing files outside model roots must not bypass containment."""
    repo_root = tmp_path / "repo"
    allowed = repo_root / "src" / "shared" / "urdf"
    allowed.mkdir(parents=True)
    outside = tmp_path / "outside.urdf"
    outside.write_text("<robot name='outside'><link name='base'/></robot>")

    monkeypatch.setattr(model_explorer, "_find_project_root", lambda: repo_root)
    monkeypatch.setattr(model_explorer, "_MODEL_DIRS", [Path("src/shared/urdf")])

    with pytest.raises(HTTPException) as excinfo:
        model_explorer._resolve_model_path(str(outside))

    assert excinfo.value.status_code == 400


def test_resolve_model_path_rejects_traversal_to_existing_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent traversal must be rejected before filesystem existence checks."""
    repo_root = tmp_path / "repo"
    allowed = repo_root / "src" / "shared" / "urdf"
    allowed.mkdir(parents=True)
    (tmp_path / "outside.urdf").write_text(
        "<robot name='outside'><link name='base'/></robot>"
    )

    monkeypatch.setattr(model_explorer, "_find_project_root", lambda: repo_root)
    monkeypatch.setattr(model_explorer, "_MODEL_DIRS", [Path("src/shared/urdf")])

    with pytest.raises(HTTPException) as excinfo:
        model_explorer._resolve_model_path("../outside.urdf")

    assert excinfo.value.status_code == 400


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
