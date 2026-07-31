"""Tests verifying tile coverage after the post-#5556 audit (PR: fix/6c-remaining-tiles).

These tests assert that every tile added in the audit branch is present in
the manifest, that no duplicate IDs exist, and that the manifest is valid
JSON.  They serve as a permanent regression guard so future manifest edits
cannot silently drop a tile that was deliberately added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST_PATH = (
    Path(__file__).parent.parent.parent / "src" / "config" / "launcher_manifest.json"
)

# ---------------------------------------------------------------------------
# Tiles added by the post-#5556 audit (PR fix/6c-remaining-tiles-post-5556-audit)
# ---------------------------------------------------------------------------
TILES_ADDED_IN_AUDIT = [
    # Engine-specific dashboards (#5515)
    "drake_dashboard",
    "mujoco_dashboard",
    "pinocchio_dashboard",
    # Analysis Tools API (#5521)
    "analysis_tools_api",
    # Motion Pipeline (#5523)
    "motion_pipeline",
    # Capability tiles (#5524-#5530)
    "perturbation_analysis",
    "force_overlays",
    "realtime_ws",
    "aip",
    "actuator_controls",
    # New feature tiles (#5532-#5535)
    "unreal_integration",
    "robotics_module",
    "tools_calculator_hub",
    "pid_generator",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest_data() -> dict:
    """Load the raw launcher manifest JSON once per test module."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tile_ids(manifest_data: dict) -> set[str]:
    """Return the set of all tile IDs present in the manifest."""
    return {tile["id"] for tile in manifest_data["tiles"]}


# ---------------------------------------------------------------------------
# Manifest validity
# ---------------------------------------------------------------------------


class TestManifestValidity:
    """Basic structural validity tests for the manifest file."""

    def test_manifest_is_valid_json(self) -> None:
        """The manifest file must parse as valid JSON without raising."""
        raw = MANIFEST_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, dict)
        assert "tiles" in data

    def test_manifest_has_tiles_list(self, manifest_data: dict) -> None:
        """tiles must be a non-empty list."""
        assert isinstance(manifest_data["tiles"], list)
        assert len(manifest_data["tiles"]) > 0

    def test_no_duplicate_ids(self, manifest_data: dict) -> None:
        """No two tiles may share the same ID."""
        ids = [tile["id"] for tile in manifest_data["tiles"]]
        seen: set[str] = set()
        duplicates: list[str] = []
        for tid in ids:
            if tid in seen:
                duplicates.append(tid)
            seen.add(tid)
        assert not duplicates, f"Duplicate tile IDs found: {duplicates}"

    def test_no_duplicate_orders(self, manifest_data: dict) -> None:
        """No two tiles may share the same order value."""
        orders = [tile["order"] for tile in manifest_data["tiles"]]
        seen: set[int] = set()
        duplicates: list[int] = []
        for order in orders:
            if order in seen:
                duplicates.append(order)
            seen.add(order)
        assert not duplicates, f"Duplicate order values: {duplicates}"


# ---------------------------------------------------------------------------
# Audit tile presence
# ---------------------------------------------------------------------------


class TestAuditTilesPresent:
    """Verify every tile added in the post-#5556 audit is still in the manifest."""

    @pytest.mark.parametrize("tile_id", TILES_ADDED_IN_AUDIT)
    def test_audit_tile_is_present(self, tile_ids: set[str], tile_id: str) -> None:
        """Each tile introduced in the audit must be present in the manifest."""
        assert tile_id in tile_ids, (
            f"Expected tile '{tile_id}' in manifest but it was not found. "
            "This tile was added intentionally — restore it or update this test."
        )


# ---------------------------------------------------------------------------
# Audit tile field completeness
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "id",
    "name",
    "description",
    "category",
    "type",
    "logo",
    "status",
    "capabilities",
    "order",
]


class TestAuditTileFields:
    """Verify every audit tile has all required fields and a path or web_route."""

    def _get_audit_tiles(self, manifest_data: dict) -> list[dict]:
        audit_set = set(TILES_ADDED_IN_AUDIT)
        return [t for t in manifest_data["tiles"] if t["id"] in audit_set]

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_audit_tiles_have_required_field(
        self, manifest_data: dict, field: str
    ) -> None:
        """Every audit tile must declare all required fields with a non-None value."""
        missing: list[str] = []
        for tile in self._get_audit_tiles(manifest_data):
            if field not in tile or tile[field] is None:
                missing.append(tile.get("id", "<unknown>"))
        assert not missing, f"Audit tiles missing required field '{field}': {missing}"

    def test_audit_tiles_have_path_or_web_route(self, manifest_data: dict) -> None:
        """Every audit tile must have either a path or a web_route."""
        missing: list[str] = []
        for tile in self._get_audit_tiles(manifest_data):
            if not tile.get("path") and not tile.get("web_route"):
                missing.append(tile["id"])
        assert (
            not missing
        ), f"Audit tiles with neither 'path' nor 'web_route': {missing}"
