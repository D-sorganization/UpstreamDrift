"""TDD Tests for Launcher API Parity.

Tests that the /api/launcher/* endpoints serve the manifest correctly,
ensuring the Tauri/React frontend can consume the same tile definitions
as the PyQt launcher.

Tests:
    1. GET /api/launcher/manifest returns full manifest
    2. GET /api/launcher/tiles returns all tiles in order
    3. GET /api/launcher/engines returns only physics_engine tiles
    4. GET /api/launcher/tools returns only tool tiles
    5. GET /api/launcher/tiles/{id} returns specific tile
    6. GET /api/launcher/tiles/{id} returns 404 for unknown tile
    7. Response format matches what React frontend expects
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from src.api.server import app
except ImportError:
    pytest.skip("API server deps not available", allow_module_level=True)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create test client with proper application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


class TestLauncherManifestEndpoints:
    """Test /api/launcher/* endpoints for launcher parity."""

    def test_get_manifest(self, client: TestClient) -> None:
        """GET /api/launcher/manifest returns full manifest."""
        response = client.get("/api/launcher/manifest")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert "tiles" in data
        assert isinstance(data["tiles"], list)
        assert len(data["tiles"]) > 0

    def test_manifest_tiles_have_required_fields(self, client: TestClient) -> None:
        """All tiles in the manifest have required fields."""
        response = client.get("/api/launcher/manifest")
        data = response.json()

        for tile in data["tiles"]:
            assert "id" in tile, f"Tile missing id: {tile}"
            assert "name" in tile, f"Tile missing name: {tile.get('id')}"
            assert "description" in tile, f"Tile missing description: {tile.get('id')}"
            assert "category" in tile, f"Tile missing category: {tile.get('id')}"
            assert "type" in tile, f"Tile missing type: {tile.get('id')}"
            assert "logo" in tile, f"Tile missing logo: {tile.get('id')}"
            assert "status" in tile, f"Tile missing status: {tile.get('id')}"
            assert "capabilities" in tile, (
                f"Tile missing capabilities: {tile.get('id')}"
            )
            assert "order" in tile, f"Tile missing order: {tile.get('id')}"

    def test_manifest_tiles_sorted_by_order(self, client: TestClient) -> None:
        """Tiles are sorted by their order field."""
        response = client.get("/api/launcher/manifest")
        tiles = response.json()["tiles"]
        orders = [t["order"] for t in tiles]
        assert orders == sorted(orders)

    def test_model_explorer_is_first(self, client: TestClient) -> None:
        """Model Explorer must be the first tile."""
        response = client.get("/api/launcher/manifest")
        tiles = response.json()["tiles"]
        assert tiles[0]["id"] == "model_explorer"

    def test_get_tiles(self, client: TestClient) -> None:
        """GET /api/launcher/tiles returns all tiles."""
        response = client.get("/api/launcher/tiles")
        assert response.status_code == 200

        tiles = response.json()
        assert isinstance(tiles, list)
        assert len(tiles) > 0

    def test_get_tile_by_id(self, client: TestClient) -> None:
        """GET /api/launcher/tiles/{id} returns specific tile."""
        response = client.get("/api/launcher/tiles/mujoco_unified")
        assert response.status_code == 200

        tile = response.json()
        assert tile["id"] == "mujoco_unified"
        assert tile["name"] == "MuJoCo"
        assert tile["category"] == "physics_engine"

    def test_get_tile_not_found(self, client: TestClient) -> None:
        """GET /api/launcher/tiles/{id} returns 404 for unknown tile."""
        response = client.get("/api/launcher/tiles/nonexistent")
        assert response.status_code == 404

    def test_get_engines(self, client: TestClient) -> None:
        """GET /api/launcher/engines returns only physics engine tiles."""
        response = client.get("/api/launcher/engines")
        assert response.status_code == 200

        engines = response.json()
        assert isinstance(engines, list)
        assert len(engines) > 0
        for eng in engines:
            assert eng["category"] == "physics_engine"
            assert "engine_type" in eng

    def test_get_tools(self, client: TestClient) -> None:
        """GET /api/launcher/tools returns only tool tiles."""
        response = client.get("/api/launcher/tools")
        assert response.status_code == 200

        tools = response.json()
        assert isinstance(tools, list)
        assert len(tools) > 0
        for tool in tools:
            assert tool["category"] == "tool"


class TestLauncherParityRequirements:
    """Test that the API serves data matching PyQt launcher expectations.

    These tests ensure the Tauri frontend can reproduce the PyQt launcher's
    tile grid layout using only data from the API.
    """

    REQUIRED_TILE_IDS = {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "putting_green",
        "matlab_unified",
        "motion_capture",
        "model_explorer",
        "video_analyzer",
        "data_explorer",
    }

    def test_all_required_tiles_present(self, client: TestClient) -> None:
        """All tiles from the PyQt launcher must be available."""
        response = client.get("/api/launcher/tiles")
        tile_ids = {t["id"] for t in response.json()}
        missing = self.REQUIRED_TILE_IDS - tile_ids
        assert not missing, f"Missing required tiles: {missing}"

    def test_status_chips_not_unknown(self, client: TestClient) -> None:
        """No tile should have 'unknown' status (fixes #1168)."""
        response = client.get("/api/launcher/tiles")
        for tile in response.json():
            assert tile["status"] != "unknown", (
                f"Tile '{tile['id']}' has unknown status"
            )

    def test_putting_green_has_valid_status(self, client: TestClient) -> None:
        """Putting Green tile has a valid (non-unknown) status chip."""
        response = client.get("/api/launcher/tiles/putting_green")
        assert response.status_code == 200
        tile = response.json()
        assert tile["status"] == "simulator"

    def test_special_app_tiles_have_valid_status(self, client: TestClient) -> None:
        """All special_app tiles have valid (non-unknown) status chips."""
        response = client.get("/api/launcher/tiles")
        special_apps = [t for t in response.json() if t["type"] == "special_app"]
        for tile in special_apps:
            assert tile["status"] in {
                "utility",
                "external",
            }, f"special_app tile '{tile['id']}' has status '{tile['status']}'"

    def test_motion_capture_has_all_capabilities(self, client: TestClient) -> None:
        """Motion Capture tile declares C3D, OpenPose, and MediaPipe capabilities."""
        response = client.get("/api/launcher/tiles/motion_capture")
        assert response.status_code == 200
        caps = response.json()["capabilities"]
        assert "c3d_viewer" in caps
        assert "openpose" in caps
        assert "mediapipe" in caps


class TestLogoEndpoints:
    """Test /api/launcher/logos/* endpoints (Phase 3, #1164)."""

    def test_get_logo_svg(self, client: TestClient) -> None:
        """GET /api/launcher/logos/{file} returns SVG logo."""
        response = client.get("/api/launcher/logos/mujoco_humanoid.svg")
        assert response.status_code == 200
        assert "image/svg+xml" in response.headers["content-type"]
        assert b"<svg" in response.content

    def test_get_logo_not_found(self, client: TestClient) -> None:
        """GET /api/launcher/logos/{file} returns 404 for missing logo."""
        response = client.get("/api/launcher/logos/nonexistent.svg")
        assert response.status_code == 404

    def test_get_logo_path_traversal_blocked(self, client: TestClient) -> None:
        """Path traversal attempts are rejected (400 or 404, never 200)."""
        # Test with dot-dot in filename
        response = client.get("/api/launcher/logos/..%2Fetc%2Fpasswd")
        assert response.status_code in {400, 404}
        # Direct dot-dot attempt
        response2 = client.get("/api/launcher/logos/..secret.svg")
        assert response2.status_code in {400, 404}

    def test_all_manifest_logos_servable(self, client: TestClient) -> None:
        """Every logo referenced in the manifest is served by the API."""
        tiles_resp = client.get("/api/launcher/tiles")
        for tile in tiles_resp.json():
            logo = tile["logo"]
            resp = client.get(f"/api/launcher/logos/{logo}")
            assert resp.status_code == 200, (
                f"Logo '{logo}' for tile '{tile['id']}' not served (HTTP {resp.status_code})"
            )

    def test_validate_logos_all_present(self, client: TestClient) -> None:
        """Logo validation reports all logos present (Phase 3 complete)."""
        response = client.get("/api/launcher/logos/validate")
        assert response.status_code == 200
        data = response.json()
        assert data["all_valid"] is True
        assert data["missing_count"] == 0
        assert data["present"] == data["total"]

    def test_logo_filenames_are_svg(self, client: TestClient) -> None:
        """All logos should now use SVG format."""
        response = client.get("/api/launcher/tiles")
        for tile in response.json():
            assert tile["logo"].endswith(".svg"), (
                f"Tile '{tile['id']}' logo '{tile['logo']}' is not SVG"
            )


class TestNewTiles:
    """Test newly added tiles (Phase 4, #1167, #1177)."""

    def test_video_analyzer_tile_exists(self, client: TestClient) -> None:
        """Video Analyzer tile is present (closes #1167)."""
        response = client.get("/api/launcher/tiles/video_analyzer")
        assert response.status_code == 200
        tile = response.json()
        assert tile["name"] == "Video Analyzer"
        assert tile["category"] == "tool"
        assert tile["status"] == "utility"

    def test_video_analyzer_has_capabilities(self, client: TestClient) -> None:
        """Video Analyzer declares video/pose capabilities."""
        response = client.get("/api/launcher/tiles/video_analyzer")
        caps = response.json()["capabilities"]
        assert "video_processing" in caps
        assert "pose_estimation" in caps

    def test_video_analyzer_has_logo(self, client: TestClient) -> None:
        """Video Analyzer logo is servable."""
        response = client.get("/api/launcher/logos/video_analyzer.svg")
        assert response.status_code == 200

    def test_data_explorer_tile_exists(self, client: TestClient) -> None:
        """Data Explorer tile is present (closes #1177, #1178)."""
        response = client.get("/api/launcher/tiles/data_explorer")
        assert response.status_code == 200
        tile = response.json()
        assert tile["name"] == "Data Explorer"
        assert tile["category"] == "tool"
        assert "data_import" in tile["capabilities"]

    def test_total_tile_count(self, client: TestClient) -> None:
        """Manifest now has 12 tiles (10 original + video_analyzer + project_map)."""
        response = client.get("/api/launcher/tiles")
        tiles = response.json()
        assert len(tiles) == 12

    def test_tool_tiles_count(self, client: TestClient) -> None:
        """There should be 4 tool tiles now."""
        response = client.get("/api/launcher/tools")
        tools = response.json()
        tool_ids = {t["id"] for t in tools}
        assert "model_explorer" in tool_ids
        assert "motion_capture" in tool_ids
        assert "video_analyzer" in tool_ids
        assert "data_explorer" in tool_ids


class TestEngineCapabilitiesAPI:
    """Tests for GET /api/launcher/engines/capabilities endpoints."""

    def test_all_engine_capabilities(self, client: TestClient) -> None:
        """GET /engines/capabilities should return all engine profiles."""
        response = client.get("/api/launcher/engines/capabilities")
        assert response.status_code == 200
        data = response.json()
        # Must include all known engines
        expected_engines = {
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
            "myosuite",
            "pendulum",
            "putting_green",
        }
        assert expected_engines.issubset(set(data.keys()))

    def test_mujoco_capabilities(self, client: TestClient) -> None:
        """MuJoCo should have FULL support for all capabilities."""
        response = client.get("/api/launcher/engines/mujoco/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["engine_name"] == "MuJoCo"
        assert data["mass_matrix"] == "full"
        assert data["jacobian"] == "full"
        assert data["video_export"] == "full"
        assert data["dataset_export"] == "full"
        assert data["force_visualization"] == "full"

    def test_drake_capabilities(self, client: TestClient) -> None:
        """Drake should have PARTIAL contact forces."""
        response = client.get("/api/launcher/engines/drake/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["engine_name"] == "Drake"
        assert data["contact_forces"] == "partial"
        assert data["mass_matrix"] == "full"

    def test_unknown_engine_404(self, client: TestClient) -> None:
        """Unknown engine should return 404."""
        response = client.get("/api/launcher/engines/nonexistent/capabilities")
        assert response.status_code == 404

    def test_capabilities_have_all_fields(self, client: TestClient) -> None:
        """Every capability profile must have all required fields."""
        response = client.get("/api/launcher/engines/capabilities")
        data = response.json()
        required_fields = {
            "engine_name",
            "mass_matrix",
            "jacobian",
            "contact_forces",
            "inverse_dynamics",
            "drift_acceleration",
            "video_export",
            "dataset_export",
            "force_visualization",
            "model_positioning",
            "measurements",
        }
        for engine_id, profile in data.items():
            missing = required_fields - set(profile.keys())
            assert not missing, f"{engine_id} missing fields: {missing}"
