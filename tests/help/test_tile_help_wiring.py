"""Anti-regression tests for the registry-driven help system (issue #9413).

These tests are the guard for defects #8843 (in-app help broken at the root)
and #8846 (24 of 25 tool GUIs had no help affordance and ``build_help_menu``
was dead code). Concretely they assert that

* every help page a tile declares exists on disk and is loadable,
* every ``ready`` / ``beta`` tile declares a page of its own,
* the help dock resolves a tile's page through the registry rather than a
  hand-maintained rule table,
* ``attach_tile_help`` installs an F1 shortcut on plain widgets and a Help
  menu (built by the previously-dead ``build_help_menu``) on windows, and
* every component registered in ``UI_HELP_TOPICS`` resolves to help content.
"""

from __future__ import annotations

import pytest

from scripts.registry.generate_registry_artifacts import (
    HELP_REQUIRED_MATURITIES,
    help_violations,
)
from src.config.tile_registry import REPO_ROOT, load_tile_registry
from src.shared.python.ui import tile_help
from src.shared.python.gui_pkg.help_content import UI_HELP_TOPICS, get_component_help

# Registry/doc assertions and the Qt affordance checks are all fast,
# in-process and headless-safe, so the whole module is a unit suite.
pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry():
    return load_tile_registry()


# ---------------------------------------------------------------------------
# Data-level guarantees (no Qt required)
# ---------------------------------------------------------------------------


def test_every_declared_help_page_exists(registry) -> None:
    """A tile may not point at a help page that is not in the checkout."""
    missing = [
        (tile.id, tile.help)
        for tile in registry.tiles
        if tile.help and not (REPO_ROOT / tile.help).is_file()
    ]
    assert not missing, f"tiles declare non-existent help pages: {missing}"


def test_every_non_hidden_tile_help_is_loadable(registry) -> None:
    """``load_help_markdown`` must return the page body, not a fallback."""
    unloadable: list[str] = []
    for tile in registry.tiles:
        if tile.maturity == "hidden" or tile.hidden or not tile.help:
            continue
        body = tile_help.load_help_markdown(tile.id, registry=registry)
        # Compare against the file rather than sniffing for fallback phrases:
        # a page is free to *describe* something missing, and one stub does.
        if body != (REPO_ROOT / tile.help).read_text(encoding="utf-8"):
            unloadable.append(tile.id)
    assert not unloadable, f"help pages that would not load: {unloadable}"


def test_every_ready_and_beta_tile_declares_help(registry) -> None:
    """A new ready/beta tile cannot ship without a help page."""
    undeclared = sorted(
        tile.id
        for tile in registry.tiles
        if tile.maturity in HELP_REQUIRED_MATURITIES and not tile.help
    )
    assert not undeclared, f"these ready/beta tiles declare no help page: {undeclared}"


def test_every_ready_and_beta_tile_has_its_own_page(registry) -> None:
    """The shared five-topic pages are no longer a tile's only help.

    Before #9413 one `simulation_controls.md` served 16 tiles and
    `engine_selection.md` served 6. Each tile now owns
    `docs/help/<tile_id>.md`.
    """
    shared = sorted(
        tile.id
        for tile in registry.tiles
        if tile.maturity in HELP_REQUIRED_MATURITIES
        and tile.help != f"docs/help/{tile.id}.md"
    )
    assert not shared, f"tiles still pointing at a shared help page: {shared}"


def test_help_gate_passes_for_the_committed_registry() -> None:
    """The gate wired into ``--check`` must be green on the committed tree."""
    assert help_violations(REPO_ROOT) == []


def test_window_class_tile_ids_name_real_tiles(registry) -> None:
    """The class-name fallback must not point at tiles that do not exist."""
    unknown = sorted(
        tile_id
        for tile_id in tile_help.WINDOW_CLASS_TILE_IDS.values()
        if registry.get(tile_id) is None
    )
    assert not unknown, f"WINDOW_CLASS_TILE_IDS names unknown tiles: {unknown}"


def test_tile_id_for_window_resolves_by_class_name() -> None:
    """Windows that cannot declare HELP_TILE_ID still resolve to their tile.

    `pinocchio_golf/gui.py` and its `ui/main_window.py` both define
    `PinocchioGUI` and both carry pre-existing mypy debt, so the tile id is
    declared centrally rather than by editing those modules (#9413).
    """

    class PinocchioGUI:
        pass

    class SomethingElse:
        pass

    assert tile_help.tile_id_for_window(PinocchioGUI()) == "pinocchio_golf"
    assert tile_help.tile_id_for_window(SomethingElse()) is None


def test_help_relpath_for_unknown_tile_is_none() -> None:
    assert tile_help.help_relpath_for("no_such_tile") is None
    assert tile_help.help_relpath_for("") is None


def test_load_help_markdown_explains_a_missing_declaration() -> None:
    """The fallback names what was searched instead of going silent (#8843)."""
    body = tile_help.load_help_markdown("no_such_tile")
    assert "no_such_tile" in body
    assert "does not declare a help page" in body
    assert "docs/architecture/PROJECT_MAP.md" in body


def test_help_status_partitions_every_tile(registry) -> None:
    status = tile_help.help_status(registry)
    total = sum(len(ids) for ids in status.values())
    assert total == len(registry.tiles)
    assert status["missing"] == []


# ---------------------------------------------------------------------------
# Component-level help (#8843: 10+ of 35 components resolved to nothing)
# ---------------------------------------------------------------------------


def test_every_registered_ui_component_resolves_to_help() -> None:
    unresolved = sorted(
        component
        for component in UI_HELP_TOPICS
        if get_component_help(component) is None
    )
    assert not unresolved, f"components with no help content: {unresolved}"


def test_component_help_entries_are_substantive() -> None:
    for component in UI_HELP_TOPICS:
        content = get_component_help(component)
        assert content is not None
        assert content["title"]
        assert len(content["description"].strip()) > 80, component


# ---------------------------------------------------------------------------
# Qt wiring (#8846)
# ---------------------------------------------------------------------------

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _first_ready_tile_with_help(registry):
    for tile in registry.tiles:
        if tile.help and tile.maturity == "ready":
            return tile
    pytest.skip("no ready tile declares a help page")


def test_attach_tile_help_installs_f1_on_a_plain_widget(qapp, registry) -> None:
    from PyQt6.QtGui import QShortcut
    from PyQt6.QtWidgets import QWidget

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
    from PyQt6.QtWidgets import QMainWindow

    tile = _first_ready_tile_with_help(registry)
    window = QMainWindow()
    assert tile_help.attach_tile_help(window, tile.id) is True

    menus = [
        action.menu()
        for action in window.menuBar().actions()
        if action.menu() is not None
    ]
    help_menus = [m for m in menus if m.title().replace("&", "") == "Help"]
    assert help_menus, "no Help menu was created"
    labels = [a.text().replace("&", "") for a in help_menus[0].actions()]
    assert "This Tool's Help" in labels
    assert "About" in labels
    window.deleteLater()


def test_attach_tile_help_is_a_noop_without_a_tile_id(qapp) -> None:
    from PyQt6.QtWidgets import QWidget

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

    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QMainWindow

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
    from PyQt6.QtWidgets import QTextBrowser

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
