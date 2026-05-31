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


def test_right_click_tab_shows_context_menu(launcher, qtbot) -> None:
    """Right-clicking a tab bar button should trigger the custom context menu with Undock option."""
    from unittest.mock import MagicMock, patch
    from PyQt6.QtCore import QPoint, QRect

    dummy = QWidget()
    launcher.workspace_tabs.addTab(dummy, "Test Tab")
    tab_bar = launcher.workspace_tabs.tabBar()
    assert tab_bar is not None

    # Mock tabBar() to return our custom tab_bar wrapper instance
    launcher.workspace_tabs.tabBar = MagicMock(return_value=tab_bar)

    # Mock tabRect to bypass offscreen/headless 0-size QRect issue
    tab_bar.tabRect = lambda idx: QRect(0, 0, 100, 30)
    center = QPoint(50, 15)

    with patch("PyQt6.QtWidgets.QMenu.exec") as mock_exec:
        launcher.workspace_tabs._show_tab_context_menu(center)
        mock_exec.assert_called_once()


def test_launch_c3d_viewer_embedded(launcher) -> None:
    """If the c3d_viewer embeddable tool is registered, _launch_c3d_viewer loads it as a tab."""
    from unittest.mock import MagicMock, patch
    from src.shared.python.launcher_embed import (
        EmbeddableTool,
        register_embeddable_tool,
        unregister_embeddable_tool,
        get_embeddable_tool,
    )

    # Mock tool adapter
    mock_tool = MagicMock(spec=EmbeddableTool)
    mock_tool.tool_id = "c3d_viewer"
    mock_tool.create_main_widget.return_value = QWidget()

    # Pre-register mock tool
    if get_embeddable_tool("c3d_viewer") is not None:
        unregister_embeddable_tool("c3d_viewer")
    register_embeddable_tool(mock_tool)

    launcher.dock_widget_as_tab = MagicMock()

    try:
        with patch.object(
            launcher.process_manager, "launch_script"
        ) as mock_launch_script:
            launcher._launch_c3d_viewer()
            # Verify it was docked as tab rather than spawned as a process
            launcher.dock_widget_as_tab.assert_called_once()
            assert launcher.dock_widget_as_tab.call_args[0][1] == "C3D Viewer"
            mock_launch_script.assert_not_called()
    finally:
        unregister_embeddable_tool("c3d_viewer")


def test_tab_right_click_undock_menu(launcher, qtbot):
    """Test that right-clicking a tab bar index triggers context menu and can undock it."""
    from unittest.mock import patch
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QTabBar

    dummy = QWidget()
    launcher.workspace_tabs.addTab(dummy, "Undock Test Tab")

    tab_bar = launcher.workspace_tabs.tabBar()
    assert tab_bar is not None

    from PyQt6.QtCore import QPointF

    event = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(10.0, 10.0),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )

    with (
        patch.object(QTabBar, "tabAt", return_value=1),
        patch("PyQt6.QtWidgets.QMenu.exec") as mock_exec,
    ):
        res = launcher.workspace_tabs.eventFilter(tab_bar, event)
        assert res is True
        mock_exec.assert_called_once()


def test_detached_tab_window_redock_via_dialog(launcher, qtbot):
    """Test DetachedTabWindow close event prompting redocking."""
    from PyQt6.QtWidgets import QMessageBox
    from unittest.mock import patch, MagicMock

    dummy = QWidget()
    launcher.workspace_tabs.addTab(dummy, "Detached Tab")

    from PyQt6.QtCore import QPoint

    launcher.workspace_tabs.detach_tab(1, QPoint(100, 100))

    assert len(launcher.workspace_tabs.detached_tabs) == 1
    detached_window = list(launcher.workspace_tabs.detached_tabs.keys())[0]

    captured_msg = None

    def mock_exec(self_msg):
        nonlocal captured_msg
        captured_msg = self_msg
        return 0

    with (
        patch.object(QMessageBox, "exec", mock_exec),
        patch.object(QMessageBox, "clickedButton") as mock_clicked,
    ):
        mock_clicked.side_effect = lambda: next(
            btn for btn in captured_msg.buttons() if btn.text() == "Redock"
        )

        close_event = MagicMock()
        detached_window.closeEvent(close_event)

        close_event.accept.assert_called_once()
        assert launcher.workspace_tabs.indexOf(dummy) != -1
        assert len(launcher.workspace_tabs.detached_tabs) == 0


def test_detached_tab_window_close_via_dialog(launcher, qtbot):
    """Test DetachedTabWindow close event prompting closing/deleting tab."""
    from PyQt6.QtWidgets import QMessageBox
    from unittest.mock import patch, MagicMock

    dummy = QWidget()
    launcher.workspace_tabs.addTab(dummy, "Detached Tab 2")

    from PyQt6.QtCore import QPoint

    launcher.workspace_tabs.detach_tab(1, QPoint(100, 100))

    assert len(launcher.workspace_tabs.detached_tabs) == 1
    detached_window = list(launcher.workspace_tabs.detached_tabs.keys())[0]

    captured_msg = None

    def mock_exec(self_msg):
        nonlocal captured_msg
        captured_msg = self_msg
        return 0

    with (
        patch.object(QMessageBox, "exec", mock_exec),
        patch.object(QMessageBox, "clickedButton") as mock_clicked,
    ):
        mock_clicked.side_effect = lambda: next(
            btn for btn in captured_msg.buttons() if btn.text() == "Close Tab"
        )

        close_event = MagicMock()
        detached_window.closeEvent(close_event)

        close_event.accept.assert_called_once()
        assert launcher.workspace_tabs.indexOf(dummy) == -1
        assert len(launcher.workspace_tabs.detached_tabs) == 0


def test_c3d_viewer_tabs_enabled_on_startup(qtbot):
    """Test that C3D viewer tabs are enabled by default on startup."""
    import importlib

    try:
        c3d_viewer_mod = importlib.import_module(
            "src.engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
        )
    except ImportError:
        c3d_viewer_mod = importlib.import_module(
            "engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
        )
    MainWidget = c3d_viewer_mod.MainWidget

    widget = MainWidget()
    qtbot.addWidget(widget)

    assert widget.tabs.isEnabled() is True


def test_docked_tab_menu_bar_preservation(launcher, qtbot):
    """Test that a QMainWindow's menu bar is preserved when docked and undocked/redocked."""
    from PyQt6.QtWidgets import QMainWindow
    from PyQt6.QtGui import QAction
    from src.shared.python.gui_pkg.draggable_tabs import DockedTabWrapper
    from PyQt6.QtCore import QPoint

    # 1. Create a dummy QMainWindow with a menu bar
    win = QMainWindow()
    menu_bar = win.menuBar()
    file_menu = menu_bar.addMenu("&File")
    test_action = QAction("Test Action", win)
    file_menu.addAction(test_action)

    # Custom attribute to verify attribute proxying
    win.my_custom_attribute = "hello_world"

    # 2. Add QMainWindow to the workspace tabs
    launcher.workspace_tabs.addTab(win, "Main Window Tab")

    # 3. Verify it gets auto-wrapped in DockedTabWrapper
    wrapped_widget = launcher.workspace_tabs.widget(1)
    assert isinstance(wrapped_widget, DockedTabWrapper)

    # 4. Verify attribute proxying works
    assert wrapped_widget.my_custom_attribute == "hello_world"
    wrapped_widget.my_custom_attribute = "modified_value"
    assert win.my_custom_attribute == "modified_value"

    # 5. Verify the menu bar layout
    assert wrapped_widget.layout() is not None
    assert wrapped_widget.layout().count() == 2  # menu_bar and win
    assert wrapped_widget.layout().itemAt(0).widget() == menu_bar

    # 6. Detach the tab
    launcher.workspace_tabs.detach_tab(1, QPoint(100, 100))
    assert len(launcher.workspace_tabs.detached_tabs) == 1
    detached_win = list(launcher.workspace_tabs.detached_tabs.keys())[0]

    # 7. Verify DetachedTabWindow set QMainWindow as central and set menu bar natively
    assert detached_win.centralWidget() == win
    assert detached_win.menuBar() == menu_bar

    # 8. Re-dock the tab
    launcher.workspace_tabs.reattach_tab(detached_win)
    assert len(launcher.workspace_tabs.detached_tabs) == 0

    # 9. Verify wrapped back inside DockedTabWrapper and layout restored
    re_wrapped_widget = launcher.workspace_tabs.widget(1)
    assert isinstance(re_wrapped_widget, DockedTabWrapper)
    assert re_wrapped_widget.layout().count() == 2
    assert re_wrapped_widget.layout().itemAt(0).widget() == menu_bar
    assert re_wrapped_widget.layout().itemAt(1).widget() == win


# ── Epic #6013: close-in-background (keep running hidden) ──────────────


def _add_background_tab(launcher, monkeypatch, title="BG Tool"):
    """Add a background-eligible tab and return (widget, index).

    Forces 'never' confirmation so close_tab does not prompt.
    """
    from src.shared.python.ui.preferences_dialog import UserPreferences

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="never")),
    )
    tabs = launcher.workspace_tabs
    widget = QWidget()
    idx = tabs.add_background_tab(widget, title)
    return widget, idx


def test_close_keeps_widget_alive_when_background(launcher, monkeypatch):
    """Closing a background-eligible tab hides it but keeps the widget alive."""
    widget, idx = _add_background_tab(launcher, monkeypatch)
    tabs = launcher.workspace_tabs
    assert tabs.indexOf(widget) == idx

    tabs.close_tab(idx)

    # Tab removed from the bar...
    assert tabs.indexOf(widget) == -1
    # ...but the widget itself is still a live, retained object.
    assert "BG Tool" in tabs.list_background_tabs()
    # Widget not deleted: attribute access must still work (no RuntimeError).
    assert widget.objectName() == ""


def test_restore_background_tab(launcher, monkeypatch):
    """A backgrounded tab can be restored back into the tab bar."""
    widget, idx = _add_background_tab(launcher, monkeypatch)
    tabs = launcher.workspace_tabs

    tabs.close_tab(idx)
    assert tabs.indexOf(widget) == -1

    tabs.restore_background_tab("BG Tool")

    restored_idx = tabs.indexOf(widget)
    assert restored_idx != -1
    assert tabs.tabText(restored_idx) == "BG Tool"
    assert "BG Tool" not in tabs.list_background_tabs()


def test_background_tab_state_preserved(launcher, monkeypatch):
    """State set on a background tab's widget survives close + restore."""
    from src.shared.python.ui.preferences_dialog import UserPreferences

    monkeypatch.setattr(
        UserPreferences,
        "load",
        classmethod(lambda cls: UserPreferences(confirm_close_tabs="never")),
    )

    class StatefulWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.counter = 0

    tabs = launcher.workspace_tabs
    widget = StatefulWidget()
    idx = tabs.add_background_tab(widget, "Stateful")
    widget.counter = 42

    tabs.close_tab(idx)
    assert widget.counter == 42  # retained while hidden

    tabs.restore_background_tab("Stateful")
    restored = tabs.widget(tabs.indexOf(widget))
    assert restored is widget
    assert restored.counter == 42
