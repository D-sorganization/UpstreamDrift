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


# ----------------------------------------------------------------------
# URDF parse caching (issue #8943)
# ----------------------------------------------------------------------


def test_parse_urdf_tree_cached_reuses_parse_until_file_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cached parse reuses one parse per file identity; mtime change re-parses."""
    import os

    urdf_path = (
        model_explorer._find_project_root()
        / "tests"
        / "fixtures"
        / "models"
        / "simple_pendulum.urdf"
    )
    content = urdf_path.read_text(encoding="utf-8")

    calls: list[str] = []
    real = model_explorer._parse_urdf_tree

    def spy(urdf_content: str, file_path: str) -> object:
        calls.append(file_path)
        return real(urdf_content, file_path)

    monkeypatch.setattr(model_explorer, "_parse_urdf_tree", spy)

    target = tmp_path / "probe.urdf"
    target.write_text(content, encoding="utf-8")
    from src.api.routes._route_utils import urdf_file_key

    first_key = urdf_file_key(target)
    first = model_explorer._parse_urdf_tree_cached(first_key, "probe.urdf")
    second = model_explorer._parse_urdf_tree_cached(first_key, "probe.urdf")
    assert calls == ["probe.urdf"]
    assert first.model_name == second.model_name

    # Changing the file's mtime must invalidate the cache entry.
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    model_explorer._parse_urdf_tree_cached(urdf_file_key(target), "probe.urdf")
    assert len(calls) == 2
    model_explorer._parse_urdf_tree_cached.cache_clear()


def test_get_model_explorer_parses_once_across_repeated_requests(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """The tree endpoint must not re-read/re-parse the URDF per request."""
    calls: list[str] = []
    real = model_explorer._parse_urdf_tree

    def spy(urdf_content: str, file_path: str) -> object:
        calls.append(file_path)
        return real(urdf_content, file_path)

    monkeypatch.setattr(model_explorer, "_parse_urdf_tree", spy)
    model_explorer._parse_urdf_tree_cached.cache_clear()

    first = client.get("/tools/model-explorer/simple_pendulum")
    assert first.status_code == 200
    assert len(calls) == 1
    second = client.get("/tools/model-explorer/simple_pendulum")
    assert second.status_code == 200
    assert len(calls) == 1
