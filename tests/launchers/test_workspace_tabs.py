from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QTabWidget, QWidget

from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher


@pytest.fixture
def launcher(qtbot):
    app = UpstreamDriftLauncher(loading=True)
    qtbot.addWidget(app)
    return app


def test_workspace_tabs_initialized(launcher):
    """Launcher should have a QTabWidget for workspace tabs, starting with Home."""
    assert hasattr(
        launcher, "workspace_tabs"
    ), "Launcher missing workspace_tabs attribute"
    assert isinstance(
        launcher.workspace_tabs, QTabWidget
    ), "workspace_tabs should be a QTabWidget"
    assert (
        launcher.workspace_tabs.count() == 1
    ), "There should be exactly one initial tab"
    assert (
        launcher.workspace_tabs.tabText(0) == "Home"
    ), "The initial tab should be named 'Home'"


def test_can_dock_engine_as_tab(launcher):
    """Launcher should expose a method to dock a widget as a new tab."""
    dummy_widget = QWidget()

    # This method needs to be implemented in UpstreamDriftLauncher
    launcher.dock_widget_as_tab(dummy_widget, "Test Engine")

    assert launcher.workspace_tabs.count() == 2, "Tab count should increase to 2"
    assert (
        launcher.workspace_tabs.tabText(1) == "Test Engine"
    ), "Tab should have the correct name"
    assert (
        launcher.workspace_tabs.widget(1) == dummy_widget
    ), "Tab should contain the correct widget"


def test_can_popout_engine(launcher):
    """Launcher should expose a method to pop out a widget into a new window."""
    dummy_widget = QWidget()

    launcher.popout_widget(dummy_widget, "Test Popped Out Engine")

    assert hasattr(
        launcher, "_popped_out_windows"
    ), "Launcher missing _popped_out_windows tracking list"
    assert len(launcher._popped_out_windows) == 1, "There should be 1 popped out window"
    win = launcher._popped_out_windows[0]
    assert (
        win.windowTitle() == "Test Popped Out Engine"
    ), "Popped out window should have the correct title"
