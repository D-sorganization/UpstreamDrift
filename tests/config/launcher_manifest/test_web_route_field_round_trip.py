"""TDD Tests for Launcher Manifest Loader.

Tests the shared launcher manifest system that ensures parity between
PyQt and Tauri/React launchers.

Test Categories:
    1. Manifest Loading — validate JSON parsing and DBC contracts
    2. Tile Properties — verify all tiles have required fields
    3. Logo Validation — check logo files exist on disk
    4. Ordering — verify Model Explorer is first tile
    5. Parity — verify all tiles can be consumed by both launchers
    6. Categories — verify physics_engine, tool, external groupings
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import (
    ASSETS_DIR,
    MANIFEST_PATH,
    LauncherManifest,
    LauncherTile,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manifest() -> LauncherManifest:
    """Load the production manifest."""
    return LauncherManifest.load()


@pytest.fixture
def sample_tile_dict() -> dict:
    """A minimal valid tile dictionary."""
    return {
        "id": "test_tile",
        "name": "Test Tile",
        "description": "A test tile",
        "category": "tool",
        "type": "special_app",
        "path": "src/test.py",
        "logo": "test.png",
        "status": "utility",
        "capabilities": ["test_cap"],
        "order": 1,
    }


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """A minimal local registry file for provider-manifest tests."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    return config_path


# =============================================================================
# 1. Manifest Loading
# =============================================================================


# =============================================================================
# 2. Tile Properties
# =============================================================================


# =============================================================================
# 3. Logo Validation
# =============================================================================


# =============================================================================
# 4. Ordering
# =============================================================================


# =============================================================================
# 5. Parity (PyQt ↔ Tauri)
# =============================================================================


# =============================================================================
# 6. Category Queries
# =============================================================================


class TestWebRouteFieldRoundTrip:
    """Tests for web_route round-trip preservation (issue #2494)."""

    def test_from_dict_preserves_web_route(self) -> None:
        """from_dict() must read web_route from the manifest dict."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "tool",
            "type": "web",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "gui_ready",
            "web_route": "/tools/test",
        }
        tile = LauncherTile.from_dict(data)
        assert tile.web_route == "/tools/test", (
            "Assertion failed: tile.web_route == /tools/test"
        )

    def test_to_dict_includes_web_route(self) -> None:
        """to_dict() must serialize web_route so it survives a round-trip."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "tool",
            "type": "web",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "gui_ready",
            "web_route": "/tools/test",
        }
        tile = LauncherTile.from_dict(data)
        serialized = tile.to_dict()
        assert "web_route" in serialized, "Assertion failed: web_route in serialized"
        assert serialized["web_route"] == "/tools/test", (
            "Assertion failed: serialized[web_route] == /tools/test"
        )

    def test_web_route_none_by_default(self) -> None:
        """web_route defaults to None when absent from the manifest dict."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "physics_engine",
            "type": "mujoco",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "engine_ready",
        }
        tile = LauncherTile.from_dict(data)
        assert tile.web_route is None, "Assertion failed: tile.web_route is None"

    def test_to_dict_omits_web_route_when_none(self) -> None:
        """to_dict() must not include web_route key when it is None."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "physics_engine",
            "type": "mujoco",
            "path": "/some/path",
            "logo": "logo.png",
        }
        tile = LauncherTile.from_dict(data)
        serialized = tile.to_dict()
        assert "web_route" not in serialized, (
            "Assertion failed: web_route not in serialized"
        )
