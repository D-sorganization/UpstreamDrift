"""Test manifest category completeness, tile field validation, and handler coverage.

Ensures every sidebar category with tiles has at least one non-hidden tile, all
tiles have required fields, IDs and orders are unique, hidden tiles explain why,
and all tile types that have a path have a corresponding handler in
ModelHandlerRegistry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.launcher_manifest_loader import (
    LAUNCHER_CATEGORIES,
    LAUNCHER_CATEGORY_LABELS,
    MANIFEST_PATH,
    LauncherManifest,
    LauncherTile,
)
from src.launchers.launcher_model_handlers import ModelHandlerRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manifest() -> LauncherManifest:
    """Load the production manifest."""
    return LauncherManifest.load()


@pytest.fixture
def manifest_json() -> dict:
    """Load the raw manifest JSON."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def registry() -> ModelHandlerRegistry:
    """Create a fresh handler registry."""
    return ModelHandlerRegistry()


# =============================================================================
# Category Completeness
# =============================================================================


class TestCategoryCompleteness:
    """Verify categories that have tiles are populated correctly."""

    SIDEBAR_CATEGORIES = LAUNCHER_CATEGORIES

    def test_categories_with_tiles_have_visible_entries(
        self, manifest: LauncherManifest
    ) -> None:
        """Any category that has tiles must have at least one visible tile."""
        empty_visible: list[str] = []
        for category in self.SIDEBAR_CATEGORIES:
            all_tiles = manifest.get_tiles_by_category(category, include_hidden=True)
            visible_tiles = manifest.get_tiles_by_category(category)
            if all_tiles and not visible_tiles:
                empty_visible.append(category)
        assert not empty_visible, (
            f"Categories with only hidden tiles: {', '.join(empty_visible)}"
        )

    def test_no_category_is_completely_empty(self, manifest: LauncherManifest) -> None:
        """Physics engine and tool categories must have at least one tile."""
        essential = {"physics_engine", "tool"}
        for cat in essential:
            tiles = manifest.get_tiles_by_category(cat, include_hidden=True)
            assert tiles, f"Essential category {cat!r} has no tiles at all"


# =============================================================================
# Tile Field Validation
# =============================================================================


class TestTileFieldValidation:
    """Verify every tile has all required fields."""

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

    def test_all_tiles_have_required_fields(self, manifest_json: dict) -> None:
        """Every tile must declare all required fields."""
        missing_fields: list[str] = []
        for tile in manifest_json["tiles"]:
            for field in self.REQUIRED_FIELDS:
                if field not in tile or tile[field] is None:
                    missing_fields.append(
                        f"{tile.get('id', '<unknown>')!r} missing {field!r}"
                    )
        assert not missing_fields, (
            f"Tiles missing required fields:\n" + "\n".join(missing_fields)
        )

    def test_every_tile_has_path_or_web_route(self, manifest_json: dict) -> None:
        """Each tile must have either a path or a web_route."""
        missing: list[str] = []
        for tile in manifest_json["tiles"]:
            has_path = bool(tile.get("path"))
            has_web_route = bool(tile.get("web_route"))
            if not has_path and not has_web_route:
                missing.append(
                    f"{tile['id']!r} has neither 'path' nor 'web_route'"
                )
        assert not missing, "\n".join(missing)

    def test_all_tiles_have_valid_category(
        self, manifest: LauncherManifest
    ) -> None:
        """Every tile category must be one of the canonical categories."""
        for tile in manifest.tiles:
            assert tile.category in LAUNCHER_CATEGORIES, (
                f"Tile {tile.id!r} has invalid category: {tile.category!r}"
            )


# =============================================================================
# Uniqueness Constraints
# =============================================================================


class TestUniquenessConstraints:
    """Verify tile IDs and orders are unique."""

    def test_all_tile_ids_are_unique(self, manifest: LauncherManifest) -> None:
        """No two tiles may share the same ID."""
        ids = [t.id for t in manifest.tiles]
        seen: set[str] = set()
        dupes: list[str] = []
        for tid in ids:
            if tid in seen:
                dupes.append(tid)
            seen.add(tid)
        assert not dupes, f"Duplicate tile IDs: {dupes}"

    def test_all_tile_orders_are_unique(self, manifest_json: dict) -> None:
        """No two tiles may share the same order value."""
        orders = [t["order"] for t in manifest_json["tiles"]]
        seen: set[int] = set()
        dupes: list[int] = []
        for order in orders:
            if order in seen:
                dupes.append(order)
            seen.add(order)
        assert not dupes, f"Duplicate order values: {dupes}"


# =============================================================================
# Hidden Tile Validation
# =============================================================================


class TestHiddenTileValidation:
    """Verify hidden tiles have a hidden_reason field."""

    def test_hidden_tiles_have_reason(self, manifest_json: dict) -> None:
        """Every hidden tile must explain why it is hidden."""
        hidden_without_reason: list[str] = []
        for tile in manifest_json["tiles"]:
            if tile.get("hidden", False):
                if not tile.get("hidden_reason"):
                    hidden_without_reason.append(tile["id"])
        assert not hidden_without_reason, (
            f"Hidden tiles missing 'hidden_reason': "
            f"{', '.join(hidden_without_reason)}"
        )


# =============================================================================
# Handler Coverage
# =============================================================================


class TestHandlerCoverage:
    """Verify tile types have a handler in ModelHandlerRegistry."""

    def test_all_manifest_tile_types_have_handlers(
        self, manifest: LauncherManifest, registry: ModelHandlerRegistry
    ) -> None:
        """Every tile type in the manifest must have a handler that can_handle()
        it."""
        missing: list[str] = []
        for tile in manifest.tiles:
            handler = registry.get_handler(tile.type)
            if handler is None:
                missing.append(f"{tile.id!r} (type={tile.type!r})")
        assert not missing, (
            f"Tiles with no handler in ModelHandlerRegistry: {', '.join(missing)}"
        )

    def test_handler_can_handle_returns_true_for_manifest_types(
        self, manifest: LauncherManifest, registry: ModelHandlerRegistry
    ) -> None:
        """Each handler's can_handle() must return True for the declared type."""
        for tile in manifest.tiles:
            handler = registry.get_handler(tile.type)
            assert handler is not None, f"No handler for tile {tile.id!r}"
            assert handler.can_handle(tile.type) is True, (
                f"Handler {type(handler).__name__}.can_handle({tile.type!r}) "
                f"returned False for tile {tile.id!r}"
            )
