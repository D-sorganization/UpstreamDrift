"""Tests for ``src.launchers.help_menu``.

The help menu is a thin wrapper around ``QMenuBar``/``QAction``.
The keyboard-shortcuts dialog scrapes :class:`QAction` and
:class:`QShortcut` instances registered on a parent window, so we
construct a minimal parent with a known set of shortcuts and assert the
table is populated correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import QMainWindow

from src.launchers import help_menu


@pytest.fixture
def parent_window(qapp) -> QMainWindow:
    win = QMainWindow()
    yield win
    win.deleteLater()


def test_add_action_sets_label_tooltip_status_and_handler(parent_window, qapp) -> None:
    handler = MagicMock()
    menu = parent_window.menuBar().addMenu("Test")
    action = help_menu._add_action(
        menu,
        parent_window,
        "&Hello",
        tooltip="Show greeting",
        status_tip="Greets the user",
        handler=handler,
        shortcut="Ctrl+G",
    )
    assert action.text() == "&Hello"
    assert action.toolTip() == "Show greeting"
    assert action.statusTip() == "Greets the user"
    assert action.shortcut().toString() == "Ctrl+G"
    # Trigger the action and confirm the handler ran:
    action.trigger()
    handler.assert_called_once()


def test_add_action_without_shortcut(parent_window, qapp) -> None:
    menu = parent_window.menuBar().addMenu("Test")
    action = help_menu._add_action(
        menu,
        parent_window,
        "&No accel",
        tooltip="t",
        status_tip="s",
        handler=lambda: None,
    )
    assert action.shortcut().isEmpty()


def test_build_help_menu_creates_expected_actions(parent_window, qapp) -> None:
    bar = parent_window.menuBar()
    menu = help_menu.build_help_menu(bar, parent_window)

    actions = menu.actions()
    # Filter separators
    labels = [a.text() for a in actions if not a.isSeparator()]
    assert "&User Guide" in labels
    assert "&Motion-Match Loaders" in labels
    assert "&Keyboard Shortcuts" in labels
    assert "&Report a Bug" in labels
    assert "&About" in labels


def test_build_help_menu_uses_custom_show_shortcuts(parent_window, qapp) -> None:
    bar = parent_window.menuBar()
    custom = MagicMock()
    menu = help_menu.build_help_menu(bar, parent_window, show_shortcuts=custom)
    # Locate the Keyboard Shortcuts action and trigger it.
    for action in menu.actions():
        if action.text() == "&Keyboard Shortcuts":
            action.trigger()
            break
    custom.assert_called_once()


def test_collect_shortcut_rows_returns_actions_and_shortcuts(
    parent_window, qapp
) -> None:
    a = QAction("Save", parent_window)
    a.setShortcut("Ctrl+S")
    parent_window.addAction(a)

    sc = QShortcut(QKeySequence("Ctrl+Shift+P"), parent_window)
    sc.setObjectName("Command Palette")

    rows = help_menu.collect_shortcut_rows(parent_window)

    keys = {r[0] for r in rows}
    assert "Ctrl+S" in keys
    assert "Ctrl+Shift+P" in keys


def test_collect_shortcut_rows_skips_actions_without_shortcuts(
    parent_window, qapp
) -> None:
    a = QAction("No Accel", parent_window)
    parent_window.addAction(a)
    rows = help_menu.collect_shortcut_rows(parent_window)
    # The action above has no shortcut so it should not show up.
    for key, _ in rows:
        assert key  # truthy; never empty


def test_collect_shortcut_rows_dedupes(parent_window, qapp) -> None:
    a1 = QAction("Save", parent_window)
    a1.setShortcut("Ctrl+S")
    a2 = QAction("Save", parent_window)
    a2.setShortcut("Ctrl+S")
    parent_window.addAction(a1)
    parent_window.addAction(a2)
    rows = help_menu.collect_shortcut_rows(parent_window)
    save_rows = [r for r in rows if r[0] == "Ctrl+S" and r[1] == "Save"]
    assert len(save_rows) == 1


def test_keyboard_shortcuts_dialog_builds_with_rows(parent_window, qapp) -> None:
    a = QAction("Open", parent_window)
    a.setShortcut("Ctrl+O")
    parent_window.addAction(a)

    dlg = help_menu.KeyboardShortcutsDialog(parent_window)
    assert dlg.windowTitle() == "Keyboard Shortcuts"
    dlg.deleteLater()


def test_show_keyboard_shortcuts_modal_invokes_exec(parent_window, qapp) -> None:
    with patch.object(help_menu.KeyboardShortcutsDialog, "exec", return_value=0) as ex:
        help_menu.show_keyboard_shortcuts_modal(parent_window)
        ex.assert_called_once()
