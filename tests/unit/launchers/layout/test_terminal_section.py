"""Tests for :mod:`src.launchers.preferences.terminal_section`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.launchers.preferences.terminal_section import (
    TerminalSection,
    discover_shells_safe,
)


def test_discover_shells_returns_nonempty_list() -> None:
    """Fallback path always produces at least one shell."""
    shells = discover_shells_safe()
    assert shells, "Expected at least one shell in fallback list"
    for shell in shells:
        assert {"id", "label", "path"} <= set(shell.keys())


def test_discover_shells_handles_tools_exception(monkeypatch) -> None:
    """If discover_shells() raises, we fall back to the local list."""
    import src.launchers.preferences.terminal_section as mod

    def _explode():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        mod, "_fallback_shells", lambda: [{"id": "x", "label": "X", "path": "x"}]
    )
    # Force the tools path to fail by monkeypatching the import lookup —
    # easier: monkeypatch discover_shells_safe's inner import via sys.modules.
    import sys

    fake = MagicMock()
    fake.discover_shells = _explode
    monkeypatch.setitem(sys.modules, "sidekick.terminal.shells", fake)
    monkeypatch.setitem(sys.modules, "sidekick", MagicMock())
    monkeypatch.setitem(sys.modules, "sidekick.terminal", MagicMock())

    shells = mod.discover_shells_safe()
    assert shells == [{"id": "x", "label": "X", "path": "x"}]


def test_terminal_section_build_widget(qt_real, qapp) -> None:  # noqa: ARG001
    """build_widget returns a QGroupBox tagged with the section id."""
    section = TerminalSection()
    widget = section.build_widget()
    assert widget.objectName() == "prefs_section_terminal"


def test_terminal_section_persists_selection(qt_real, qapp) -> None:  # noqa: ARG001
    """Selecting a shell triggers the injected set_default callback."""
    saved: list[str] = []
    section = TerminalSection(
        get_default=lambda: "cmd",
        set_default=lambda v: saved.append(v),
    )
    _widget = section.build_widget()
    if section._combo is not None and section._combo.count() > 1:
        section._combo.setCurrentIndex(1)
        # editing combo programmatically should have triggered the callback
        assert saved and saved[-1] == section._combo.itemData(1)


def test_terminal_section_restores_default(qt_real, qapp) -> None:  # noqa: ARG001
    """When get_default returns an id present in the combo, it gets selected."""
    shells = discover_shells_safe()
    if not shells:
        pytest.skip("No shells discovered to test restore against")
    target_id = shells[0]["id"]
    section = TerminalSection(get_default=lambda: target_id)
    _widget = section.build_widget()
    assert section.selected_shell_id == target_id
