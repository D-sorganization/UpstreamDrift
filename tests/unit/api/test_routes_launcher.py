"""Unit tests for the launcher API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import launcher as launcher_routes
from src.api.routes.launcher import router
from src.config.launcher_manifest_loader import LauncherManifest, LauncherTile

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the launcher router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def hidden_tile_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a manifest with one visible tile and one hidden alias."""
    manifest = LauncherManifest(
        version="test",
        description="visibility fixture",
        tiles=(
            LauncherTile(
                id="visible_tool",
                name="Visible Tool",
                description="Shown in public launcher catalog endpoints",
                category="tool",
                type="special_app",
                path="src/tools/visible_tool/__main__.py",
                logo="visible.svg",
                status="ready",
                order=1,
            ),
            LauncherTile(
                id="hidden_alias",
                name="Hidden Alias",
                description="Legacy alias hidden from public launcher catalog endpoints",
                category="tool",
                type="special_app",
                path="src/tools/hidden_alias/__main__.py",
                logo="hidden.svg",
                status="ready",
                order=2,
                hidden=True,
                hidden_reason="Legacy alias",
                hidden_owner="launcher-team",
            ),
        ),
    )
    monkeypatch.setitem(launcher_routes._launcher_state, "manifest", manifest)


def test_get_manifest(client: TestClient) -> None:
    """Test getting the launcher manifest."""
    response = client.get("/launcher/manifest")
    assert response.status_code == 200
    data = response.json()
    assert "tiles" in data
    assert len(data["tiles"]) > 0


def test_get_tiles(client: TestClient) -> None:
    """Test getting all tiles."""
    response = client.get("/launcher/tiles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_tiles_excludes_hidden_entries_by_default(
    client: TestClient, hidden_tile_manifest: None
) -> None:
    """Public tile listing uses the same visible-tile contract as the manifest."""
    response = client.get("/launcher/tiles")
    assert response.status_code == 200

    tile_ids = {tile["id"] for tile in response.json()}
    assert tile_ids == {"visible_tool"}


def test_get_tile_found(client: TestClient) -> None:
    """Test getting a specific tile."""
    # Assuming "mujoco" or something similar exists
    # we first get all tiles to find a valid id
    tiles_resp = client.get("/launcher/tiles")
    valid_id = tiles_resp.json()[0]["id"]

    response = client.get(f"/launcher/tiles/{valid_id}")
    assert response.status_code == 200
    assert response.json()["id"] == valid_id


def test_get_tile_not_found(client: TestClient) -> None:
    """Test getting a non-existent tile."""
    response = client.get("/launcher/tiles/unknown_tile_id")
    assert response.status_code == 404


def test_get_tile_returns_not_found_for_hidden_entries_by_default(
    client: TestClient, hidden_tile_manifest: None
) -> None:
    """Hidden aliases are not addressable through the public tile detail route."""
    response = client.get("/launcher/tiles/hidden_alias")
    assert response.status_code == 404


def test_get_engines_and_tools(client: TestClient) -> None:
    """Test getting filtered tiles."""
    engines_resp = client.get("/launcher/engines")
    assert engines_resp.status_code == 200
    assert isinstance(engines_resp.json(), list)

    tools_resp = client.get("/launcher/tools")
    assert tools_resp.status_code == 200
    assert isinstance(tools_resp.json(), list)


def test_validate_logos(client: TestClient) -> None:
    """Test validating logos."""
    response = client.get("/launcher/logos/validate")
    assert response.status_code == 200
    data = response.json()
    assert "all_valid" in data
    assert "missing_count" in data


def test_get_logo_invalid_path(client: TestClient) -> None:
    """Test path traversal protection for logos."""
    response = client.get("/launcher/logos/..%2Fsecret.txt")
    assert response.status_code in (400, 404)


def test_get_engine_capabilities(client: TestClient) -> None:
    """Test getting all engine capabilities."""
    response = client.get("/launcher/engines/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "mujoco" in data
    assert "drake" in data
    assert data["jaxsim"]["engine_name"] == "JaxSim"
    assert data["jaxsim"]["contact_forces"] == "partial"
    assert data["jaxsim"]["inverse_dynamics"] == "full"


def test_get_single_engine_capabilities(client: TestClient) -> None:
    """Test getting capabilities for a specific engine."""
    response = client.get("/launcher/engines/mujoco/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_name"] == "MuJoCo"


def test_get_jaxsim_capabilities(client: TestClient) -> None:
    """JaxSim capability profile supports selector grey-out decisions."""
    response = client.get("/launcher/engines/jaxsim/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_name"] == "JaxSim"
    assert data["jacobian"] == "full"
    assert data["force_visualization"] == "none"


def test_get_single_engine_capabilities_not_found(client: TestClient) -> None:
    """Test getting capabilities for an unknown engine."""
    response = client.get("/launcher/engines/unknown/capabilities")
    assert response.status_code == 404
