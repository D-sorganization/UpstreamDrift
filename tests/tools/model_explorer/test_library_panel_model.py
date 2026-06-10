"""Headless tests for the unified model-explorer library panel model."""

from __future__ import annotations

from typing import Any

import pytest

from src.tools.model_explorer.library_panel_model import (
    LibraryPanelModel,
    format_badge_for_model,
)

pytestmark = [pytest.mark.unit]


class FakeLibrary:
    def __init__(self) -> None:
        self._info: dict[tuple[str, str], dict[str, Any]] = {
            (
                "human",
                "simple_humanoid",
            ): {
                "name": "Simple Humanoid",
                "description": "Bundled humanoid model with declared interfaces.",
                "type": "local",
            },
            ("golf_clubs", "driver"): {
                "name": "Driver",
                "description": "Generated golf club URDF.",
            },
            ("pendulum", "double"): {
                "name": "Double Pendulum",
                "description": "MJCF pendulum fixture.",
                "type": "mjcf",
                "path": "src/engines/pendulum.xml",
            },
        }

    def list_available_models(self) -> dict[str, Any]:
        return {
            "human": ["simple_humanoid"],
            "golf_clubs": ["driver"],
            "pendulum": ["double"],
            "discovered": [
                {
                    "name": "repo_arm.urdf",
                    "description": "Repo URDF arm",
                    "path": "src/models/repo_arm.urdf",
                    "type": "urdf",
                    "config_key": "repo_arm",
                }
            ],
            "sibling": [
                {
                    "name": "simple_arm.osim",
                    "description": "OpenSim sibling arm",
                    "path": "../OpenSim_Models/simple_arm.osim",
                    "type": "osim",
                    "repo": "OpenSim_Models",
                    "config_key": "sibling_arm",
                }
            ],
            "embedded": {
                "full_body_golf_swing": {
                    "name": "Full Body Golf Swing",
                    "description": "Embedded MuJoCo XML.",
                    "content": "<mujoco/>",
                }
            },
            "robot_descriptions": [],
            "imported": [],
        }

    def get_model_info(self, category: str, key: str) -> dict[str, Any] | None:
        return self._info.get((category, key))


def test_from_library_flattens_all_categories_with_sibling_entries() -> None:
    model = LibraryPanelModel.from_library(FakeLibrary())

    rows = {(entry.category, entry.key): entry for entry in model.entries}

    assert ("human", "simple_humanoid") in rows
    assert ("sibling", "sibling_arm") in rows
    assert rows[("sibling", "sibling_arm")].category_label == "Sibling Repositories"
    assert rows[("sibling", "sibling_arm")].format_badge == "OSIM"


def test_format_badge_for_model_uses_category_defaults_and_explicit_types() -> None:
    assert format_badge_for_model("golf_clubs", {"name": "Driver"}) == "URDF"
    assert format_badge_for_model("human", {"type": "embedded"}) == "MJCF"
    assert format_badge_for_model("discovered", {"type": "sdf"}) == "SDF"
    assert format_badge_for_model("component", {"type": "component"}) == "COMPONENT"


def test_filter_entries_matches_name_category_format_repo_and_path_tokens() -> None:
    model = LibraryPanelModel.from_library(FakeLibrary())

    assert [entry.key for entry in model.filter_entries("sibling osim")] == [
        "sibling_arm"
    ]
    assert [entry.key for entry in model.filter_entries("OpenSim simple_arm")] == [
        "sibling_arm"
    ]
    assert [entry.key for entry in model.filter_entries("mjcf pendulum")] == ["double"]


def test_grouped_entries_preserves_category_order_and_drops_empty_groups() -> None:
    model = LibraryPanelModel.from_library(FakeLibrary())

    groups = model.grouped_entries("arm")

    assert [group.category for group in groups] == ["discovered", "sibling"]
    assert [entry.key for group in groups for entry in group.entries] == [
        "repo_arm",
        "sibling_arm",
    ]


def test_invalid_inputs_raise_clear_contract_errors() -> None:
    with pytest.raises(ValueError, match="listing"):
        LibraryPanelModel.from_listing(None, lambda _category, _key: None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="category"):
        format_badge_for_model("", {"type": "urdf"})
