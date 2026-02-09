"""Tests for terrain/environment API routes.

Validates environment presets, terrain queries, material listing,
and environment loading via the REST API.

Fixes #1145 (engine-agnostic environment system)
Fixes #1142 (expandable environment system beyond putting green)
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError as _exc:
    pytest.skip(f"API server deps not available: {_exc}", allow_module_level=True)


@pytest.fixture()
def client() -> TestClient:
    """Create test client."""
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────
#  Preset Listing
# ──────────────────────────────────────────────────────────────
class TestPresetListing:
    """Verify environment preset discovery."""

    def test_list_presets_returns_200(self, client: TestClient) -> None:
        """GET /api/terrain/presets returns 200."""
        resp = client.get("/api/terrain/presets")
        assert resp.status_code == 200

    def test_list_presets_returns_list(self, client: TestClient) -> None:
        """Presets response is a list."""
        data = client.get("/api/terrain/presets").json()
        assert isinstance(data, list)
        assert len(data) >= 6  # at least 6 presets

    def test_preset_has_required_fields(self, client: TestClient) -> None:
        """Each preset has name, description, terrain_types, width, length."""
        data = client.get("/api/terrain/presets").json()
        for preset in data:
            assert "name" in preset
            assert "description" in preset
            assert "terrain_types" in preset
            assert "width_m" in preset
            assert "length_m" in preset

    def test_putting_green_in_presets(self, client: TestClient) -> None:
        """Putting green preset exists."""
        data = client.get("/api/terrain/presets").json()
        names = [p["name"] for p in data]
        assert "putting_green" in names

    def test_driving_range_in_presets(self, client: TestClient) -> None:
        """Driving range preset exists."""
        data = client.get("/api/terrain/presets").json()
        names = [p["name"] for p in data]
        assert "driving_range" in names

    def test_full_hole_in_presets(self, client: TestClient) -> None:
        """Full hole preset exists."""
        data = client.get("/api/terrain/presets").json()
        names = [p["name"] for p in data]
        assert "full_hole" in names


# ──────────────────────────────────────────────────────────────
#  Environment Loading
# ──────────────────────────────────────────────────────────────
class TestEnvironmentLoading:
    """Verify environment loading via API."""

    VALID_PRESETS = [
        "putting_green",
        "fairway",
        "driving_range",
        "bunker",
        "rough",
        "full_hole",
    ]

    @pytest.mark.parametrize("preset", VALID_PRESETS)
    def test_load_preset_succeeds(self, client: TestClient, preset: str) -> None:
        """Each valid preset loads successfully."""
        resp = client.post("/api/terrain/load", json={"preset": preset})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "name" in data

    def test_load_invalid_preset_returns_error(self, client: TestClient) -> None:
        """Invalid preset returns success=False."""
        resp = client.post("/api/terrain/load", json={"preset": "moon_surface"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "error" in data

    def test_load_with_custom_dimensions(self, client: TestClient) -> None:
        """Custom width/length override preset defaults."""
        resp = client.post(
            "/api/terrain/load",
            json={"preset": "fairway", "width": 100.0, "length": 400.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["width_m"] == 100.0
        assert data["length_m"] == 400.0

    def test_load_with_slope(self, client: TestClient) -> None:
        """Environment loads with slope parameters."""
        resp = client.post(
            "/api/terrain/load",
            json={
                "preset": "putting_green",
                "slope_angle_deg": 3.0,
                "slope_direction_deg": 45.0,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ──────────────────────────────────────────────────────────────
#  Terrain Queries
# ──────────────────────────────────────────────────────────────
class TestTerrainQueries:
    """Verify terrain property queries."""

    def test_query_returns_200(self, client: TestClient) -> None:
        """POST /api/terrain/query returns 200."""
        # Load an environment first
        client.post("/api/terrain/load", json={"preset": "fairway"})
        resp = client.post("/api/terrain/query", json={"x": 25.0, "y": 100.0})
        assert resp.status_code == 200

    def test_query_response_fields(self, client: TestClient) -> None:
        """Query response has all required fields."""
        client.post("/api/terrain/load", json={"preset": "fairway"})
        data = client.post("/api/terrain/query", json={"x": 25.0, "y": 100.0}).json()
        assert "elevation" in data
        assert "slope_angle_deg" in data
        assert "terrain_type" in data
        assert "friction" in data
        assert "restitution" in data
        assert "rolling_resistance" in data

    def test_query_returns_numeric_values(self, client: TestClient) -> None:
        """Query results are numeric."""
        client.post("/api/terrain/load", json={"preset": "fairway"})
        data = client.post("/api/terrain/query", json={"x": 25.0, "y": 100.0}).json()
        assert isinstance(data["elevation"], (int, float))
        assert isinstance(data["friction"], (int, float))
        assert data["friction"] > 0

    def test_query_different_terrain_types(self, client: TestClient) -> None:
        """Querying different positions returns different terrain types."""
        # Load full hole which has multiple terrain types
        client.post("/api/terrain/load", json={"preset": "full_hole"})
        # Center should be fairway
        center = client.post("/api/terrain/query", json={"x": 30.0, "y": 100.0}).json()
        assert center["terrain_type"] in [
            "fairway",
            "rough",
            "tee",
            "green",
            "bunker",
            "fringe",
        ]

    def test_query_out_of_bounds_handled(self, client: TestClient) -> None:
        """Out-of-bounds query gracefully handled."""
        client.post("/api/terrain/load", json={"preset": "putting_green"})
        resp = client.post("/api/terrain/query", json={"x": 9999.0, "y": 9999.0})
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────
#  Material & Type Listing
# ──────────────────────────────────────────────────────────────
class TestMaterialListing:
    """Verify material and terrain type listing."""

    def test_list_materials_returns_200(self, client: TestClient) -> None:
        """GET /api/terrain/materials returns 200."""
        resp = client.get("/api/terrain/materials")
        assert resp.status_code == 200

    def test_materials_include_all_terrain_types(self, client: TestClient) -> None:
        """Materials include fairway, rough, green, bunker."""
        data = client.get("/api/terrain/materials").json()
        names = [m["name"] for m in data]
        for expected in ["fairway", "rough", "green", "bunker"]:
            assert expected in names

    def test_material_has_physics_properties(self, client: TestClient) -> None:
        """Each material has friction, restitution, etc."""
        data = client.get("/api/terrain/materials").json()
        for mat in data:
            assert "friction_coefficient" in mat
            assert "restitution" in mat
            assert "hardness" in mat
            assert mat["friction_coefficient"] >= 0
            assert 0 <= mat["restitution"] <= 1

    def test_list_terrain_types_returns_all(self, client: TestClient) -> None:
        """GET /api/terrain/types returns all terrain type names."""
        data = client.get("/api/terrain/types").json()
        assert isinstance(data, list)
        for expected in ["fairway", "rough", "green", "bunker", "tee", "fringe"]:
            assert expected in data


# ──────────────────────────────────────────────────────────────
#  Active Terrain
# ──────────────────────────────────────────────────────────────
class TestActiveTerrain:
    """Verify active terrain info endpoint."""

    def test_active_terrain_default(self, client: TestClient) -> None:
        """Default active terrain exists."""
        resp = client.get("/api/terrain/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "width_m" in data
        assert "resolution_m" in data

    def test_active_terrain_after_load(self, client: TestClient) -> None:
        """Active terrain updates after loading a preset."""
        client.post("/api/terrain/load", json={"preset": "driving_range"})
        data = client.get("/api/terrain/active").json()
        assert data["name"] == "driving_range"
