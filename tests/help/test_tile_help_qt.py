"""Qt affordance tests for the tile help system (#9413, #10868, #10869).

These tests verify the Qt-specific behavior of the tile help system:
* ``attach_tile_help`` installs an F1 shortcut on plain widgets,
* a Help menu is built on windows without conflicting F1 shortcuts (#10868),
* any pre-existing F1 shortcut in menu actions is cleared defensively (#10868),
* the help dock resolves a tile's page through the registry rather than a
  hand-maintained rule table,
* ``show_tile_help`` renders modeless dialogs properly.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6.QtWidgets")
if getattr(sys.modules.get("PyQt6"), "__ud_fake__", False):
    pytest.skip(
        "real PyQt6 is unavailable; skipping Qt help tests",
        allow_module_level=True,
    )

from PyQt6.QtGui import QAction, QShortcut  # noqa: E402
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextBrowser,
    QWidget,
)  # noqa: E402

from src.shared.python.ui import tile_help  # noqa: E402
from src.shared.python.ui.tile_help import (
    HELP_SHORTCUT,
    REPO_ROOT,
    load_tile_registry,
)  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry():
    return load_tile_registry()


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _first_ready_tile_with_help(registry):
    for tile in registry.tiles:
        if tile.help and tile.maturity == "ready":
            return tile
    pytest.skip("no ready tile declares a help page")


def test_attach_tile_help_installs_f1_on_a_plain_widget(qapp, registry) -> None:
    tile = _first_ready_tile_with_help(registry)
    widget = QWidget()
    assert tile_help.attach_tile_help(widget, tile.id) is True
    assert tile_help.attached_tile_id(widget) == tile.id

    keys = {sc.key().toString() for sc in widget.findChildren(QShortcut)}
    assert tile_help.HELP_SHORTCUT in keys

    # Idempotent: re-embedding the same tool must not stack shortcuts.
    assert tile_help.attach_tile_help(widget, tile.id) is False
    assert len(widget.findChildren(QShortcut)) == 1
    widget.deleteLater()


def test_attach_tile_help_builds_a_help_menu_on_a_window(qapp, registry) -> None:
    """``build_help_menu`` is no longer dead code (#8846)."""
    tile = _first_ready_tile_with_help(registry)
    window = QMainWindow()
    assert tile_help.attach_tile_help(window, tile.id) is True

    menubar = window.menuBar()
    assert menubar is not None
    menus = [action.menu() for action in menubar.actions() if action.menu() is not None]
    help_menus = [
        m for m in menus if m is not None and m.title().replace("&", "") == "Help"
    ]
    assert help_menus, "no Help menu was created"
    first_help = help_menus[0]
    assert first_help is not None
    labels = [a.text().replace("&", "") for a in first_help.actions()]
    assert "This Tool's Help" in labels
    assert "About" in labels
    window.deleteLater()


def test_attach_tile_help_has_no_f1_shortcut_collision_with_menu(
    qapp, registry
) -> None:
    """Ensure no menu action steals or collides with F1 shortcut (#10868)."""
    tile = _first_ready_tile_with_help(registry)
    window = QMainWindow()
    assert tile_help.attach_tile_help(window, tile.id) is True

    # Check all actions in the window and its menu bar
    menubar = window.menuBar()
    assert menubar is not None
    conflicting_actions = [
        action.text()
        for action in menubar.actions()
        if action.shortcut().toString() == HELP_SHORTCUT
    ]
    for menu_action in menubar.actions():
        menu = menu_action.menu()
        if menu is not None:
            for sub_action in menu.actions():
                if sub_action.shortcut().toString() == HELP_SHORTCUT:
                    conflicting_actions.append(sub_action.text())

    assert not conflicting_actions, (
        f"Actions with conflicting F1 shortcut found: {conflicting_actions}"
    )

    # Verify that the widget has exactly one QShortcut for F1
    f1_shortcuts = [
        sc
        for sc in window.findChildren(QShortcut)
        if sc.key().toString() == HELP_SHORTCUT
    ]
    assert len(f1_shortcuts) == 1
    window.deleteLater()


def test_attach_tile_help_clears_preexisting_menu_f1_shortcut(qapp, registry) -> None:
    """_attach_help_menu defensively clears any existing F1 menu shortcuts (#10868)."""
    tile = _first_ready_tile_with_help(registry)
    window = QMainWindow()
    menubar = window.menuBar()
    assert menubar is not None
    menu = menubar.addMenu("&Help")
    assert menu is not None
    rogue_action = QAction("&Rogue Help", window)
    rogue_action.setShortcut(HELP_SHORTCUT)
    menu.addAction(rogue_action)

    assert rogue_action.shortcut().toString() == HELP_SHORTCUT

    # When tile help is attached, rogue F1 shortcut should be stripped
    tile_help._attach_help_menu(window, tile.id)
    assert rogue_action.shortcut().isEmpty() or rogue_action.shortcut().toString() == ""
    window.deleteLater()


def test_attach_tile_help_is_a_noop_without_a_tile_id(qapp) -> None:
    widget = QWidget()
    assert tile_help.attach_tile_help(widget, None) is False
    assert tile_help.attached_tile_id(widget) is None
    widget.deleteLater()


def test_attach_tile_help_never_breaks_a_launch(qapp, registry) -> None:
    """Help is an affordance, not a dependency.

    A widget that cannot carry a Qt shortcut — a plain object, a test double,
    a C++-side object already destroyed — must make ``attach_tile_help``
    return ``False``, not raise into the tool's launch path.
    """
    tile = _first_ready_tile_with_help(registry)

    class NotAWidget:
        pass

    assert tile_help.attach_tile_help(NotAWidget(), tile.id) is False

    assert tile_help.attach_tile_help(MagicMock(spec=QMainWindow), tile.id) is False


def test_help_dock_loads_every_declared_page(qapp, registry) -> None:
    """The dock must render each tile's registry page, not a "not found"."""
    from src.launchers.help_dialogs import ContextHelpDock

    dock = ContextHelpDock()
    failures: list[str] = []
    for tile in registry.tiles:
        if not tile.help or tile.maturity == "hidden":
            continue
        dock.update_context(tile.id)
        text = dock.text_area.toPlainText()
        if not text.strip() or "No documentation file has been written" in text:
            failures.append(tile.id)
    assert not failures, f"dock could not load help for: {failures}"
    dock.deleteLater()


def test_help_dock_prefers_the_registry_over_the_legacy_rule_table(
    qapp, registry
) -> None:
    """Registry `help:` wins over the hand-maintained fallback rules."""
    from src.launchers.help_dialogs import ContextHelpDock

    tile = registry.get("mujoco_unified")
    assert tile is not None and tile.help, "mujoco_unified should declare help"
    dock = ContextHelpDock()
    candidates = dock._doc_candidates("mujoco_unified")
    assert [p.as_posix() for p in candidates] == [(REPO_ROOT / tile.help).as_posix()]
    dock.deleteLater()


def test_show_tile_help_renders_the_page(qapp, registry) -> None:
    tile = _first_ready_tile_with_help(registry)
    dialog = tile_help.show_tile_help(None, tile.id)
    try:
        assert tile.name in dialog.windowTitle()
        browsers = dialog.findChildren(QTextBrowser)
        assert browsers, "help dialog has no text browser"
        assert browsers[0].toPlainText().strip()
    finally:
        dialog.close()
        dialog.deleteLater()
