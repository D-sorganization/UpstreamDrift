"""Tests for the UpstreamDrift tool catalog (#5314).

Verifies that all UD tools are discoverable, categorized, and have
the required metadata fields.  These tests are the inventory-vs-launcher
coverage tests described in the acceptance criteria.
"""

from __future__ import annotations

import pytest

from src.shared.python.gui_launcher.ud_tool_catalog import (
    VALID_CATEGORIES,
    UDToolEntry,
    UDToolCatalog,
    get_ud_tool_catalog,
)

# ── UDToolEntry unit tests ────────────────────────────────────────────


class TestUDToolEntry:
    def test_all_required_fields_present(self) -> None:
        entry = UDToolEntry(
            tool_id="pose_studio",
            title="Pose Studio",
            category="Biomechanics",
            description="Interactive cross-engine pose editor",
            command="src.tools.pose_studio.main",
            is_hidden=False,
        )
        assert entry.tool_id == "pose_studio"
        assert entry.title == "Pose Studio"
        assert entry.category == "Biomechanics"
        assert entry.description
        assert entry.command

    def test_requires_non_empty_tool_id(self) -> None:
        with pytest.raises(ValueError, match="tool_id"):
            UDToolEntry(
                tool_id="",
                title="T",
                category="Biomechanics",
                description="D",
                command="c",
            )

    def test_requires_valid_category(self) -> None:
        with pytest.raises(ValueError, match="category"):
            UDToolEntry(
                tool_id="t",
                title="T",
                category="INVALID_CAT_XYZ",
                description="D",
                command="c",
            )

    def test_hidden_entries_require_reason(self) -> None:
        with pytest.raises(ValueError, match="hidden_reason"):
            UDToolEntry(
                tool_id="t",
                title="T",
                category="Biomechanics",
                description="D",
                command="c",
                is_hidden=True,
                hidden_reason=None,
            )

    def test_hidden_entry_with_reason_is_valid(self) -> None:
        entry = UDToolEntry(
            tool_id="dev_only",
            title="Dev Tool",
            category="Developer Tools",
            description="Only for devs",
            command="src.dev.main",
            is_hidden=True,
            hidden_reason="Requires unreleased hardware (owner: engineering, unblock: #9999)",
        )
        assert entry.is_hidden is True
        assert entry.hidden_reason


# ── UDToolCatalog unit tests ──────────────────────────────────────────


class TestUDToolCatalog:
    def test_catalog_is_not_empty(self) -> None:
        cat = UDToolCatalog()
        assert len(cat.all_tools()) > 0

    def test_all_tools_have_non_empty_title(self) -> None:
        for tool in UDToolCatalog().all_tools():
            assert tool.title, f"Tool '{tool.tool_id}' has empty title"

    def test_all_tools_have_non_empty_description(self) -> None:
        for tool in UDToolCatalog().all_tools():
            assert tool.description, f"Tool '{tool.tool_id}' missing description"

    def test_all_tools_have_valid_category(self) -> None:
        for tool in UDToolCatalog().all_tools():
            assert tool.category in VALID_CATEGORIES, (
                f"Tool '{tool.tool_id}' has unknown category '{tool.category}'"
            )

    def test_all_tools_have_non_empty_command(self) -> None:
        for tool in UDToolCatalog().all_tools():
            assert tool.command, f"Tool '{tool.tool_id}' missing command"

    def test_hidden_tools_have_documented_reason(self) -> None:
        for tool in UDToolCatalog().all_tools():
            if tool.is_hidden:
                assert tool.hidden_reason, (
                    f"Hidden tool '{tool.tool_id}' must document reason"
                )

    def test_no_duplicate_tool_ids(self) -> None:
        tools = UDToolCatalog().all_tools()
        ids = [t.tool_id for t in tools]
        assert len(ids) == len(set(ids)), "Duplicate tool_id found"

    def test_list_by_category_returns_subset(self) -> None:
        cat = UDToolCatalog()
        physics_tools = cat.by_category("Physics Engines")
        assert len(physics_tools) > 0
        for tool in physics_tools:
            assert tool.category == "Physics Engines"

    def test_list_by_category_unknown_returns_empty(self) -> None:
        cat = UDToolCatalog()
        assert cat.by_category("NonExistentCategory_XYZ") == []

    def test_visible_tools_are_subset_of_all(self) -> None:
        cat = UDToolCatalog()
        visible = cat.visible_tools()
        all_tools = cat.all_tools()
        assert len(visible) <= len(all_tools)
        for tool in visible:
            assert not tool.is_hidden

    def test_physics_engine_tools_are_registered(self) -> None:
        """Physics engine tiles must be in the catalog."""
        cat = UDToolCatalog()
        # At least one engine-related tool must be present
        engine_tools = {
            t.tool_id for t in cat.all_tools() if t.category == "Physics Engines"
        }
        assert engine_tools, "No Physics Engine tools registered"

    def test_biomechanics_tools_are_registered(self) -> None:
        cat = UDToolCatalog()
        bio_tools = cat.by_category("Biomechanics")
        assert len(bio_tools) > 0, "No Biomechanics tools registered"

    def test_get_tool_by_id(self) -> None:
        cat = UDToolCatalog()
        all_tools = cat.all_tools()
        first = all_tools[0]
        found = cat.get(first.tool_id)
        assert found is not None
        assert found.tool_id == first.tool_id

    def test_get_unknown_id_returns_none(self) -> None:
        cat = UDToolCatalog()
        assert cat.get("nonexistent_xyz_abc") is None

    def test_list_categories_returns_all_valid(self) -> None:
        cat = UDToolCatalog()
        cats = cat.list_categories()
        for c in cats:
            assert c in VALID_CATEGORIES

    def test_category_counts_sum_to_all_visible(self) -> None:
        cat = UDToolCatalog()
        visible = cat.visible_tools()
        category_sum = sum(len(cat.by_category(c)) for c in cat.list_categories())
        # Hidden tools may appear in by_category; total by-cat >= visible count
        assert category_sum >= len(visible)


# ── get_ud_tool_catalog singleton ─────────────────────────────────────


class TestGetUDToolCatalog:
    def test_returns_catalog_instance(self) -> None:
        c = get_ud_tool_catalog()
        assert isinstance(c, UDToolCatalog)

    def test_singleton_returns_same_instance(self) -> None:
        c1 = get_ud_tool_catalog()
        c2 = get_ud_tool_catalog()
        assert c1 is c2


# ── VALID_CATEGORIES contract ─────────────────────────────────────────


class TestValidCategories:
    def test_expected_categories_present(self) -> None:
        required = {
            "Physics Engines",
            "Biomechanics",
            "Simulation",
            "Motion Capture",
            "Analysis",
            "Visualization",
            "Developer Tools",
        }
        assert required.issubset(VALID_CATEGORIES), (
            f"Missing categories: {required - VALID_CATEGORIES}"
        )

    def test_categories_are_non_empty_strings(self) -> None:
        for cat in VALID_CATEGORIES:
            assert isinstance(cat, str) and cat
