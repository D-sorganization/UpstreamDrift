"""Tests for ScrewVisualizationTab UI component.

Validates that the shared Screw Theory visualization tab can be instantiated
and provides the expected interface: is_active() and get_target_body().
"""

from __future__ import annotations

import pytest
from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

pytestmark = pytest.mark.unit


@skip_if_unavailable("pyqt6")
class TestScrewVisualizationTab:
    """Tests for ScrewVisualizationTab widget."""

    @pytest.fixture(scope="class")
    def qapp(self):
        """Ensure QApplication exists for the test class."""
        from src.shared.python.gui_pkg.gui_utils import get_qapp

        return get_qapp()

    @pytest.fixture
    def tab(self, qapp):
        """Instantiate a ScrewVisualizationTab for testing."""
        from src.shared.python.screw_theory.ui import ScrewVisualizationTab

        return ScrewVisualizationTab()

    def test_ui_instantiation(self, tab) -> None:
        """Tab can be created without errors."""
        assert tab is not None

    def test_is_active_default_false(self, tab) -> None:
        """Screw axis visualization is off by default.

        Post: is_active() returns False before any user interaction.
        """
        assert tab.is_active() is False

    def test_get_target_body_default_empty(self, tab) -> None:
        """Target body field is empty by default.

        Post: get_target_body() returns empty string when no input given.
        """
        assert tab.get_target_body() == ""

    def test_show_screw_axis_cb_exists(self, tab) -> None:
        """The show_screw_axis_cb checkbox widget exists."""
        assert hasattr(tab, "show_screw_axis_cb")
        assert tab.show_screw_axis_cb is not None

    def test_target_body_input_exists(self, tab) -> None:
        """The target_body_input line edit widget exists."""
        assert hasattr(tab, "target_body_input")
        assert tab.target_body_input is not None

    def test_visualization_changed_signal_exists(self, tab) -> None:
        """The visualization_changed signal exists on the tab."""
        assert hasattr(tab, "visualization_changed")

    def test_target_body_changed_signal_exists(self, tab) -> None:
        """The target_body_changed signal exists on the tab."""
        assert hasattr(tab, "target_body_changed")

    def test_set_target_body_via_input(self, tab) -> None:
        """Setting text on target_body_input is reflected in get_target_body()."""
        tab.target_body_input.setText("club_head")
        assert tab.get_target_body() == "club_head"

    def test_toggle_active_via_checkbox(self, tab) -> None:
        """Checking the checkbox makes is_active() return True."""
        from PyQt6.QtCore import Qt

        tab.show_screw_axis_cb.setCheckState(Qt.CheckState.Checked)
        assert tab.is_active() is True

    def test_uncheck_checkbox_deactivates(self, tab) -> None:
        """Unchecking the checkbox makes is_active() return False."""
        from PyQt6.QtCore import Qt

        tab.show_screw_axis_cb.setCheckState(Qt.CheckState.Checked)
        tab.show_screw_axis_cb.setCheckState(Qt.CheckState.Unchecked)
        assert tab.is_active() is False
