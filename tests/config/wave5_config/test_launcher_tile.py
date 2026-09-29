"""Unit tests for LauncherTile dataclass."""

from __future__ import annotations

import pytest

from src.config.launcher_manifest_loader import (
    ASSETS_DIR,
    TOOL_LIKE_CATEGORIES,
    LauncherTile,
)


class TestFromDict:
    def test_minimal_valid_entry(self, minimal_tile_dict):
        tile = LauncherTile.from_dict(minimal_tile_dict)
        assert tile.id == "alpha"
        assert tile.name == "Alpha"
        assert tile.category == "tool"
        assert tile.status == "utility"
        assert tile.order == 1
        assert tile.capabilities == ()
        assert tile.tags == ()
        assert tile.hidden is False

    def test_missing_required_fields_raises(self, minimal_tile_dict):
        del minimal_tile_dict["name"]
        with pytest.raises(ValueError, match="missing required fields"):
            LauncherTile.from_dict(minimal_tile_dict)

    def test_missing_all_required_fields_raises(self):
        with pytest.raises(ValueError, match="missing required fields"):
            LauncherTile.from_dict({})

    def test_defaults_when_optional_missing(self, minimal_tile_dict):
        minimal_tile_dict.pop("status")
        minimal_tile_dict.pop("order")
        tile = LauncherTile.from_dict(minimal_tile_dict)
        assert tile.status == "unknown"
        assert tile.order == 99

    def test_capabilities_and_tags_become_tuples(self, make_tile):
        data = make_tile(capabilities=["a", "b"], tags=["x"])
        tile = LauncherTile.from_dict(data)
        assert tile.capabilities == ("a", "b")
        assert tile.tags == ("x",)

    def test_python_paths_becomes_tuple(self, make_tile):
        data = make_tile(python_paths=["src/", "lib/"])
        tile = LauncherTile.from_dict(data)
        assert tile.python_paths == ("src/", "lib/")

    def test_hidden_requires_reason_and_owner(self, make_tile):
        data = make_tile(hidden=True)
        with pytest.raises(ValueError, match="hidden_reason"):
            LauncherTile.from_dict(data)

    def test_hidden_blank_reason_rejected(self, make_tile):
        data = make_tile(hidden=True, hidden_reason="   ", hidden_owner="team")
        with pytest.raises(ValueError, match="hidden_reason"):
            LauncherTile.from_dict(data)

    def test_hidden_non_string_reason_rejected(self, make_tile):
        data = make_tile(hidden=True, hidden_reason=42, hidden_owner="team")
        with pytest.raises(ValueError, match="hidden_reason"):
            LauncherTile.from_dict(data)

    def test_hidden_missing_owner_raises(self, make_tile):
        data = make_tile(hidden=True, hidden_reason="legacy alias")
        with pytest.raises(ValueError, match="hidden_owner"):
            LauncherTile.from_dict(data)

    def test_hidden_blank_owner_raises(self, make_tile):
        data = make_tile(hidden=True, hidden_reason="alias", hidden_owner="")
        with pytest.raises(ValueError, match="hidden_owner"):
            LauncherTile.from_dict(data)

    def test_hidden_valid_strips_whitespace(self, make_tile):
        data = make_tile(
            hidden=True,
            hidden_reason="  legacy  ",
            hidden_owner="  team-x  ",
        )
        tile = LauncherTile.from_dict(data)
        assert tile.hidden is True
        assert tile.hidden_reason == "legacy"
        assert tile.hidden_owner == "team-x"


class TestToDict:
    def test_minimal_round_trip_excludes_optional(self, minimal_tile_dict):
        tile = LauncherTile.from_dict(minimal_tile_dict)
        d = tile.to_dict()
        assert d["id"] == "alpha"
        assert d["capabilities"] == []
        assert "engine_type" not in d
        assert "provider" not in d
        assert "hidden" not in d

    def test_includes_engine_type(self, make_tile):
        tile = LauncherTile.from_dict(make_tile(engine_type="drake"))
        assert tile.to_dict()["engine_type"] == "drake"

    def test_includes_provider_fields(self, make_tile):
        data = make_tile(
            provider="acme",
            source_root="/srv/acme",
            working_dir="/srv/acme/work",
            python_paths=["/srv/acme/src"],
            web_route="/acme",
            tags=["beta"],
        )
        tile = LauncherTile.from_dict(data)
        d = tile.to_dict()
        assert d["provider"] == "acme"
        assert d["source_root"] == "/srv/acme"
        assert d["working_dir"] == "/srv/acme/work"
        assert d["python_paths"] == ["/srv/acme/src"]
        assert d["web_route"] == "/acme"
        assert d["tags"] == ["beta"]

    def test_includes_hidden_block(self, make_tile):
        data = make_tile(
            hidden=True,
            hidden_reason="legacy alias",
            hidden_owner="team",
        )
        tile = LauncherTile.from_dict(data)
        d = tile.to_dict()
        assert d["hidden"] is True
        assert d["hidden_reason"] == "legacy alias"
        assert d["hidden_owner"] == "team"


class TestProperties:
    def test_logo_path_uses_assets_dir(self, minimal_tile_dict):
        tile = LauncherTile.from_dict(minimal_tile_dict)
        assert tile.logo_path == ASSETS_DIR / "alpha.svg"

    def test_logo_exists_false_for_unknown(self, make_tile):
        tile = LauncherTile.from_dict(make_tile(logo="does-not-exist.svg"))
        assert tile.logo_exists is False

    def test_is_physics_engine_true(self, make_tile):
        tile = LauncherTile.from_dict(make_tile(category="physics_engine"))
        assert tile.is_physics_engine is True
        assert tile.is_tool is False

    def test_is_tool_true_for_tool_like(self, make_tile):
        for cat in TOOL_LIKE_CATEGORIES:
            tile = LauncherTile.from_dict(make_tile(category=cat))
            assert tile.is_tool is True, cat

    def test_external_is_not_tool_or_engine(self, make_tile):
        tile = LauncherTile.from_dict(make_tile(category="external"))
        assert tile.is_tool is False
        assert tile.is_physics_engine is False

    def test_frozen_dataclass(self, minimal_tile_dict):
        from dataclasses import FrozenInstanceError

        tile = LauncherTile.from_dict(minimal_tile_dict)
        with pytest.raises(FrozenInstanceError):
            tile.id = "other"  # type: ignore[misc]
