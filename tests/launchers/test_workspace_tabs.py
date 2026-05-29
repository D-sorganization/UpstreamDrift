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
    assert hasattr(launcher, "workspace_tabs"), (
        "Launcher missing workspace_tabs attribute"
    )
    assert isinstance(launcher.workspace_tabs, QTabWidget), (
        "workspace_tabs should be a QTabWidget"
    )
    assert launcher.workspace_tabs.count() == 1, (
        "There should be exactly one initial tab"
    )
    assert launcher.workspace_tabs.tabText(0) == "Home", (
        "The initial tab should be named 'Home'"
    )


def test_can_dock_engine_as_tab(launcher):
    """Launcher should expose a method to dock a widget as a new tab."""
    dummy_widget = QWidget()

    # This method needs to be implemented in UpstreamDriftLauncher
    launcher.dock_widget_as_tab(dummy_widget, "Test Engine")

    assert launcher.workspace_tabs.count() == 2, "Tab count should increase to 2"
    assert launcher.workspace_tabs.tabText(1) == "Test Engine", (
        "Tab should have the correct name"
    )
    assert launcher.workspace_tabs.widget(1) == dummy_widget, (
        "Tab should contain the correct widget"
    )


def test_can_popout_engine(launcher):
    """Launcher should expose a method to pop out a widget into a new window."""
    dummy_widget = QWidget()

    launcher.popout_widget(dummy_widget, "Test Popped Out Engine")

    assert hasattr(launcher, "_popped_out_windows"), (
        "Launcher missing _popped_out_windows tracking list"
    )
    assert len(launcher._popped_out_windows) == 1, "There should be 1 popped out window"
    win = launcher._popped_out_windows[0]
    assert win.windowTitle() == "Test Popped Out Engine", (
        "Popped out window should have the correct title"
    )


def test_tab_close_never_prompt(launcher, monkeypatch):
    """If confirm_close_tabs is 'never', closing a tab should not prompt and should delete the widget."""
    from unittest.mock import MagicMock
    from src.shared.python.ui.preferences_dialog import UserPreferences
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="never")),
    )

    msg_spy = MagicMock()
    monkeypatch.setattr(QMessageBox, "question", msg_spy)

    dummy_widget = QWidget()
    launcher.workspace_tabs.addTab(dummy_widget, "Temp Tab")
    idx = launcher.workspace_tabs.indexOf(dummy_widget)

    launcher.workspace_tabs.close_tab(idx)

    assert launcher.workspace_tabs.indexOf(dummy_widget) == -1
    msg_spy.assert_not_called()


def test_tab_close_unsaved_clean_no_prompt(launcher, monkeypatch):
    """If confirm_close_tabs is 'unsaved' and widget is clean, closing a tab should not prompt."""
    from unittest.mock import MagicMock
    from src.shared.python.ui.preferences_dialog import UserPreferences
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="unsaved")),
    )

    msg_spy = MagicMock()
    monkeypatch.setattr(QMessageBox, "question", msg_spy)

    class CleanWidget(QWidget):
        def is_dirty(self) -> bool:
            return False

    dummy_widget = CleanWidget()
    launcher.workspace_tabs.addTab(dummy_widget, "Temp Tab")
    idx = launcher.workspace_tabs.indexOf(dummy_widget)

    launcher.workspace_tabs.close_tab(idx)

    assert launcher.workspace_tabs.indexOf(dummy_widget) == -1
    msg_spy.assert_not_called()


def test_tab_close_unsaved_dirty_prompts(launcher, monkeypatch):
    """If confirm_close_tabs is 'unsaved' and widget is dirty, closing a tab should prompt and obey reply."""
    from src.shared.python.ui.preferences_dialog import UserPreferences
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="unsaved")),
    )

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )

    class DirtyWidget(QWidget):
        def is_dirty(self) -> bool:
            return True

    dummy_widget = DirtyWidget()
    launcher.workspace_tabs.addTab(dummy_widget, "Temp Tab")
    idx = launcher.workspace_tabs.indexOf(dummy_widget)

    launcher.workspace_tabs.close_tab(idx)

    assert launcher.workspace_tabs.indexOf(dummy_widget) != -1

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    launcher.workspace_tabs.close_tab(idx)

    assert launcher.workspace_tabs.indexOf(dummy_widget) == -1


def test_tab_close_always_prompts(launcher, monkeypatch):
    """If confirm_close_tabs is 'always', it should prompt even if the widget is clean."""
    from src.shared.python.ui.preferences_dialog import UserPreferences
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="always")),
    )

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )

    dummy_widget = QWidget()
    launcher.workspace_tabs.addTab(dummy_widget, "Temp Tab")
    idx = launcher.workspace_tabs.indexOf(dummy_widget)

    launcher.workspace_tabs.close_tab(idx)
    assert launcher.workspace_tabs.indexOf(dummy_widget) != -1

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    launcher.workspace_tabs.close_tab(idx)
    assert launcher.workspace_tabs.indexOf(dummy_widget) == -1
