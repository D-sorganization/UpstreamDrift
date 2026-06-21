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


@pytest.mark.unit
def test_get_model_urdf_basename_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disambiguated 'dir/name' entry resolves by exact basename."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "sub/widget", "format": "urdf", "path": "missing/widget.urdf"},
        ],
    )
    # Exact basename "widget" matches the single "sub/widget" entry. The file
    # does not exist, so resolution gets past the name lookup and 404s on the
    # missing file (deterministic basename match, not a substring guess).
    response = client.get("/models/widget/urdf")
    assert response.status_code == 404
    assert "file not found" in response.json()["detail"].lower()


@pytest.mark.unit
def test_get_model_urdf_ambiguous_basename(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two entries sharing a basename yield a 404 listing the candidates."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "a/widget", "format": "urdf", "path": "a/widget.urdf"},
            {"name": "b/widget", "format": "urdf", "path": "b/widget.urdf"},
        ],
    )
    response = client.get("/models/widget/urdf")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "ambiguous" in detail.lower()
    assert "a/widget" in detail and "b/widget" in detail


@pytest.mark.unit
def test_get_model_urdf_no_substring_match(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A substring of a model name no longer resolves (deterministic match)."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "simple_pendulum", "format": "urdf", "path": "x/sp.urdf"},
        ],
    )
    # "pendulum" used to match via the old substring fallback; now it 404s.
    response = client.get("/models/pendulum/urdf")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
