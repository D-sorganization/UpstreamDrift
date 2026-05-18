"""Tests for :mod:`src.launchers.feature_menu`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.launchers.feature_menu import (
    FEATURE_ENTRIES,
    FeatureMenuEntry,
    build_feature_menu_entries,
    is_feature_available,
    register_feature_menu,
)


def test_feature_entries_have_unique_ids() -> None:
    """No two entries may share a feature_id (DbC: identity invariant)."""
    ids = [e.feature_id for e in FEATURE_ENTRIES]
    assert len(set(ids)) == len(ids), f"Duplicate feature_ids: {ids}"


def test_feature_entries_have_unique_shortcuts() -> None:
    """Shortcuts must be unique to avoid Qt ambiguous-shortcut warnings."""
    shortcuts = [e.shortcut for e in FEATURE_ENTRIES]
    assert len(set(shortcuts)) == len(shortcuts), f"Duplicate shortcuts: {shortcuts}"


def test_all_expected_features_present() -> None:
    """Every Tools surfacing PR must have a corresponding feature entry."""
    expected = {
        "os_terminal",  # #2882
        "python_repl",  # #2883
        "workspace",  # #2883
        "jupyter",  # #2889
        "mcp_servers",  # #2884
    }
    assert {e.feature_id for e in FEATURE_ENTRIES} >= expected


def test_expected_shortcuts() -> None:
    """The documented Ctrl+Shift+X shortcuts must match the spec."""
    expected = {
        "os_terminal": "Ctrl+Shift+T",
        "python_repl": "Ctrl+Shift+R",
        "workspace": "Ctrl+Shift+W",
        "jupyter": "Ctrl+Shift+J",
        "mcp_servers": "Ctrl+Shift+M",
    }
    actual = {e.feature_id: e.shortcut for e in FEATURE_ENTRIES}
    for feat_id, shortcut in expected.items():
        assert actual[feat_id] == shortcut


def test_is_feature_available_unknown_raises() -> None:
    with pytest.raises(ValueError):
        is_feature_available("not_a_real_feature")


def test_is_feature_available_empty_raises() -> None:
    with pytest.raises(ValueError):
        is_feature_available("")


def test_build_entries_include_unavailable_default() -> None:
    """Default behaviour returns the full canonical list."""
    entries = build_feature_menu_entries()
    assert len(entries) == len(FEATURE_ENTRIES)


def test_build_entries_filter_unavailable() -> None:
    """include_unavailable=False filters by probe result."""
    entries = build_feature_menu_entries(include_unavailable=False)
    for entry in entries:
        assert entry.availability_probe()


def test_register_feature_menu_requires_launcher() -> None:
    with pytest.raises(ValueError):
        register_feature_menu(None, MagicMock())


def test_register_feature_menu_requires_menubar() -> None:
    with pytest.raises(ValueError):
        register_feature_menu(MagicMock(), None)


def test_register_feature_menu_creates_actions(qt_real, qapp) -> None:  # noqa: ARG001
    """Wired actions match the entries that pass their probes."""
    from PyQt6.QtWidgets import QMainWindow

    window = QMainWindow()
    actions = register_feature_menu(window, window.menuBar())
    # Always-on entries must be present:
    for fid in ("os_terminal", "python_repl", "workspace", "mcp_servers"):
        assert fid in actions
        assert actions[fid].shortcut().toString() == next(
            e.shortcut for e in FEATURE_ENTRIES if e.feature_id == fid
        )


def test_auto_hide_jupyter_when_unavailable(
    qt_real, qapp, monkeypatch
) -> None:  # noqa: ARG001
    """Jupyter entry is omitted when nbformat is unavailable."""
    from PyQt6.QtWidgets import QMainWindow

    import src.launchers.feature_menu as fm

    # Replace the jupyter entry's probe with a False-returning probe and
    # rebuild a temporary entries tuple so the auto-hide path triggers.
    new_entries = tuple(
        FeatureMenuEntry(
            feature_id=e.feature_id,
            label=e.label,
            shortcut=e.shortcut,
            status_tip=e.status_tip,
            availability_probe=(
                fm._nbformat_available
                if e.feature_id == "jupyter"
                else e.availability_probe
            ),
            factory=e.factory,
        )
        for e in FEATURE_ENTRIES
    )
    monkeypatch.setattr(fm, "FEATURE_ENTRIES", new_entries)
    monkeypatch.setattr(fm, "_nbformat_available", lambda: False)

    window = QMainWindow()
    actions = fm.register_feature_menu(window, window.menuBar())
    assert "jupyter" not in actions


def test_factory_dispatches_to_launcher_hook() -> None:
    """The default factory calls the launcher's open_sidekick_tab method."""
    entry = next(e for e in FEATURE_ENTRIES if e.feature_id == "os_terminal")
    launcher = MagicMock()
    entry.factory(launcher)
    launcher.open_sidekick_tab.assert_called_once_with("os_terminal")


def test_factory_warns_when_launcher_missing_hook(caplog) -> None:
    """Factory logs a warning and toasts when no hook is wired."""
    entry = next(e for e in FEATURE_ENTRIES if e.feature_id == "python_repl")
    launcher = MagicMock(spec=["show_toast"])
    entry.factory(launcher)
    launcher.show_toast.assert_called_once()
