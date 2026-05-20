"""Tests for the launcher UX polish PR.

Covers:
- Status-chip mapping reads launcher.status from YAML before falling back
  to type-based detection (no more "Unknown" chip on special_app, etc.).
- Category dispatch maps the new launcher.category values to the new
  taxonomy (Physics Engines / Simulation / Motion Matching /
  Motion Capture / Tools & Data / Documentation).
- Per-tile quick-launch button is wired to the launcher's launch path
  and is hidden by default (revealed on hover).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---- _get_status_info -----------------------------------------------------


@pytest.fixture
def make_card(qapp):
    """Factory: build a real DraggableModelCard with a mock model + launcher.

    Depends on the session-scoped ``qapp`` fixture from
    ``tests/launchers/conftest.py`` so a QApplication exists.
    """
    from src.launchers.model_card import DraggableModelCard

    def _make(
        *, type_: str, status: str | None = None, launcher_category: str | None = None
    ) -> DraggableModelCard:
        model = MagicMock()
        model.id = "test_model"
        model.name = "Test"
        model.description = ""
        model.engine_type = ""
        model.type = type_
        if status is None and launcher_category is None:
            model.launcher = None
        else:
            launcher = {}
            if status is not None:
                launcher["status"] = status
            if launcher_category is not None:
                launcher["category"] = launcher_category
            model.launcher = launcher
        parent_launcher = MagicMock()
        parent_launcher.layout_edit_mode = False
        return DraggableModelCard(model, parent_launcher)

    return _make


@pytest.mark.parametrize(
    "yaml_status,expected_text,expected_class",
    [
        ("ready", "Ready", "success"),
        ("Ready", "Ready", "success"),
        ("READY", "Ready", "success"),
        ("available", "Ready", "success"),
        ("stable", "Ready", "success"),
        ("beta", "Beta", "info"),
        ("experimental", "Experimental", "info"),
        ("alpha", "Alpha", "warning"),
        ("broken", "Broken", "error"),
        ("deprecated", "Deprecated", "warning"),
        ("external", "External", "external"),
    ],
)
def test_status_chip_reads_yaml_first(
    make_card, yaml_status, expected_text, expected_class
):
    card = make_card(type_="special_app", status=yaml_status)
    text, css_class = card._get_status_info()
    assert text == expected_text
    assert css_class == expected_class


def test_status_chip_no_unknown_for_special_app(make_card):
    """Regression: special_app without launcher block must not say 'Unknown'."""
    card = make_card(type_="special_app", status=None)
    text, _ = card._get_status_info()
    assert text != "Unknown"
    assert text == "Ready"


def test_status_chip_no_unknown_for_putting_green(make_card):
    card = make_card(type_="putting_green", status=None)
    text, _ = card._get_status_info()
    assert text != "Unknown"


def test_status_chip_no_unknown_for_matlab_suite(make_card):
    card = make_card(type_="matlab_suite", status=None)
    text, css = card._get_status_info()
    assert text != "Unknown"
    assert css == "external"


def test_status_chip_document_renders_as_reference(make_card):
    card = make_card(type_="document", status=None)
    text, _ = card._get_status_info()
    assert text == "Reference"


def test_status_chip_unknown_yaml_value_falls_back(make_card):
    """An unrecognised launcher.status string falls through to type detection."""
    card = make_card(type_="custom_humanoid", status="totally-unknown-value")
    text, _ = card._get_status_info()
    assert text == "GUI Ready"  # type-based fallback


# ---- _get_model_category --------------------------------------------------


@pytest.fixture
def layout_manager(tmp_path):
    """A bare LayoutManager just for category dispatch tests."""
    from src.launchers.launcher_layout_manager import LayoutManager

    return LayoutManager(
        config_file=tmp_path / "layout.json",
        available_models={},
        get_model_func=lambda mid: None,
        create_card_func=lambda model: None,
    )


@pytest.mark.parametrize(
    "category,expected",
    [
        ("physics_engine", "Physics Engines"),
        ("simulation", "Simulation"),
        ("motion_matching", "Motion Matching"),
        ("motion_capture", "Motion Capture"),
        ("tool", "Tools & Data"),
        ("documentation", "Documentation"),
        ("external", "Tools & Data"),
        # Case-insensitive
        ("Physics_Engine", "Physics Engines"),
        ("MOTION_MATCHING", "Motion Matching"),
    ],
)
def test_category_dispatch_from_launcher_yaml(layout_manager, category, expected):
    model = MagicMock()
    model.launcher = {"category": category}
    model.type = ""
    assert layout_manager._get_model_category(model) == expected


def test_category_fallback_for_putting_green_type(layout_manager):
    model = MagicMock()
    model.launcher = None
    model.type = "putting_green"
    assert layout_manager._get_model_category(model) == "Simulation"


def test_category_fallback_for_engine_types(layout_manager):
    for t in ("custom_humanoid", "drake", "pinocchio", "opensim", "myosim"):
        model = MagicMock()
        model.launcher = None
        model.type = t
        assert (
            layout_manager._get_model_category(model) == "Physics Engines"
        ), f"type {t!r} did not map to Physics Engines"


def test_category_fallback_for_document_type(layout_manager):
    model = MagicMock()
    model.launcher = None
    model.type = "document"
    assert layout_manager._get_model_category(model) == "Documentation"


# ---- Quick-launch button --------------------------------------------------


def test_quick_launch_button_built_and_hidden(make_card):
    card = make_card(type_="special_app", status="ready")
    btn = card._btn_quick_launch
    assert btn is not None
    assert btn.objectName() == "CardQuickLaunch"
    assert btn.isHidden(), "quick-launch button must start hidden"


def test_quick_launch_button_calls_launcher(make_card):
    card = make_card(type_="special_app", status="ready")
    card.parent_launcher.launch_model_direct = MagicMock()
    card._on_quick_launch_clicked()
    card.parent_launcher.launch_model_direct.assert_called_once_with(card.model.id)


def test_quick_launch_button_safe_when_launcher_missing(make_card):
    card = make_card(type_="special_app", status="ready")
    card.parent_launcher = None
    # Must not raise even without a parent launcher.
    card._on_quick_launch_clicked()
