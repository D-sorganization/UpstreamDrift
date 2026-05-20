"""Unit tests for LauncherManifest.load and queries (isolated, no real manifest)."""

from __future__ import annotations

import json

import pytest

from src.config.launcher_manifest_loader import (
    LAUNCHER_CATEGORIES,
    LAUNCHER_CATEGORY_LABELS,
    TOOL_LIKE_CATEGORIES,
    LauncherManifest,
)


class TestLoad:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError, match="Launcher manifest not found"):
            LauncherManifest.load(missing, include_provider_tiles=False)

    def test_missing_tiles_key_raises(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'tiles' array"):
            LauncherManifest.load(p, include_provider_tiles=False)

    def test_tiles_not_a_list_raises(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"tiles": {"foo": 1}}), encoding="utf-8")
        with pytest.raises(ValueError, match="'tiles' must be a list"):
            LauncherManifest.load(p, include_provider_tiles=False)

    def test_loads_empty_manifest(self, write_manifest):
        path = write_manifest()
        manifest = LauncherManifest.load(path, include_provider_tiles=False)
        assert manifest.version == "1.0.0"
        assert manifest.description == "test manifest"
        assert manifest.tiles == ()

    def test_default_version_when_missing(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"tiles": []}), encoding="utf-8")
        manifest = LauncherManifest.load(p, include_provider_tiles=False)
        assert manifest.version == "0.0.0"
        assert manifest.description == ""

    def test_tiles_sorted_by_order_then_id(self, write_manifest, make_tile):
        tiles_raw = [
            make_tile(id="z_late", order=3),
            make_tile(id="b_first", order=1),
            make_tile(id="a_first", order=1),
            make_tile(id="m_mid", order=2),
        ]
        path = write_manifest(tiles_raw)
        manifest = LauncherManifest.load(path, include_provider_tiles=False)
        assert [t.id for t in manifest.tiles] == [
            "a_first",
            "b_first",
            "m_mid",
            "z_late",
        ]

    def test_duplicate_ids_raises(self, write_manifest, make_tile):
        tiles_raw = [make_tile(id="dup"), make_tile(id="dup", name="Dup2")]
        path = write_manifest(tiles_raw)
        with pytest.raises(ValueError, match="Duplicate tile IDs"):
            LauncherManifest.load(path, include_provider_tiles=False)

    def test_provider_registry_missing_returns_empty(
        self, tmp_path, write_manifest, make_tile
    ):
        path = write_manifest([make_tile(id="alpha")])
        bogus_registry = tmp_path / "does_not_exist.yaml"
        manifest = LauncherManifest.load(
            path, include_provider_tiles=True, registry_path=bogus_registry
        )
        assert [t.id for t in manifest.tiles] == ["alpha"]


class TestQueries:
    @pytest.fixture
    def manifest(self, write_manifest, make_tile):
        tiles_raw = [
            make_tile(id="engine_a", category="physics_engine", order=1),
            make_tile(id="engine_b", category="physics_engine", order=2),
            make_tile(id="tool_a", category="tool", order=3),
            make_tile(id="ext_a", category="external", order=4),
            make_tile(
                id="hidden_alias",
                category="tool",
                order=5,
                hidden=True,
                hidden_reason="legacy alias",
                hidden_owner="team-x",
            ),
        ]
        path = write_manifest(tiles_raw)
        return LauncherManifest.load(path, include_provider_tiles=False)

    def test_get_tile_found(self, manifest):
        tile = manifest.get_tile("engine_a")
        assert tile is not None
        assert tile.id == "engine_a"

    def test_get_tile_not_found_returns_none(self, manifest):
        assert manifest.get_tile("nonexistent") is None

    def test_get_tile_none_raises(self, manifest):
        with pytest.raises(ValueError, match="tile_id must be provided"):
            manifest.get_tile(None)  # type: ignore[arg-type]

    def test_get_tiles_by_category_filters_hidden(self, manifest):
        tools = manifest.get_tiles_by_category("tool")
        assert [t.id for t in tools] == ["tool_a"]

    def test_get_tiles_by_category_include_hidden(self, manifest):
        tools = manifest.get_tiles_by_category("tool", include_hidden=True)
        assert {t.id for t in tools} == {"tool_a", "hidden_alias"}

    def test_get_tiles_by_category_invalid(self, manifest):
        with pytest.raises(ValueError, match="Unknown launcher category"):
            manifest.get_tiles_by_category("bogus")

    def test_physics_engines_property(self, manifest):
        ids = [t.id for t in manifest.physics_engines]
        assert ids == ["engine_a", "engine_b"]

    def test_tools_property_excludes_hidden(self, manifest):
        assert [t.id for t in manifest.tools] == ["tool_a"]

    def test_categories_property(self, manifest):
        cats = manifest.categories
        assert set(cats.keys()) == set(LAUNCHER_CATEGORY_LABELS.keys())
        assert {t.id for t in cats["physics_engine"]} == {"engine_a", "engine_b"}
        # hidden tile excluded
        assert all(not t.hidden for tiles in cats.values() for t in tiles)

    def test_visible_tiles_excludes_hidden(self, manifest):
        ids = [t.id for t in manifest.visible_tiles]
        assert "hidden_alias" not in ids
        assert "tool_a" in ids

    def test_tile_ids_and_ordered_ids(self, manifest):
        assert manifest.tile_ids == manifest.ordered_ids
        assert manifest.tile_ids == [
            "engine_a",
            "engine_b",
            "tool_a",
            "ext_a",
            "hidden_alias",
        ]

    def test_to_dict_default_excludes_hidden(self, manifest):
        d = manifest.to_dict()
        ids = [t["id"] for t in d["tiles"]]
        assert "hidden_alias" not in ids
        assert d["version"] == "1.0.0"
        assert d["category_labels"] == dict(LAUNCHER_CATEGORY_LABELS)

    def test_to_dict_include_hidden(self, manifest):
        d = manifest.to_dict(include_hidden=True)
        ids = [t["id"] for t in d["tiles"]]
        assert "hidden_alias" in ids

    def test_validate_logos_lists_missing(self, manifest):
        missing = manifest.validate_logos()
        # All synthetic logos are nonexistent
        assert "engine_a" in missing
        assert len(missing) == len(manifest.tiles)


class TestModuleConstants:
    def test_categories_frozenset_matches_labels(self):
        assert frozenset(LAUNCHER_CATEGORY_LABELS) == LAUNCHER_CATEGORIES

    def test_tool_like_subset_of_categories(self):
        assert TOOL_LIKE_CATEGORIES <= LAUNCHER_CATEGORIES

    def test_physics_engine_not_in_tools(self):
        assert "physics_engine" not in TOOL_LIKE_CATEGORIES
