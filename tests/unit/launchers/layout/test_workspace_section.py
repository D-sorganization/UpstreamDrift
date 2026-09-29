"""Tests for :mod:`src.launchers.preferences.workspace_section`."""

from __future__ import annotations

from src.launchers.preferences.workspace_section import (
    LAYOUT_MODES,
    WorkspaceSection,
)


def test_layout_modes_includes_sidebar_and_matlab_home() -> None:
    ids = {mid for mid, _label in LAYOUT_MODES}
    assert "SIDEBAR" in ids
    assert "MATLAB_HOME" in ids


def test_workspace_section_build_widget(qt_real, qapp) -> None:  # noqa: ARG001
    section = WorkspaceSection()
    widget = section.build_widget()
    assert widget.objectName() == "prefs_section_workspace"


def test_workspace_section_restores_default(qt_real, qapp) -> None:  # noqa: ARG001
    section = WorkspaceSection(get_default=lambda: "MATLAB_HOME")
    _widget = section.build_widget()
    assert section.selected_mode == "MATLAB_HOME"


def test_workspace_section_persists_selection(qt_real, qapp) -> None:  # noqa: ARG001
    saved: list[str] = []
    section = WorkspaceSection(
        get_default=lambda: "SIDEBAR",
        set_default=lambda v: saved.append(v),
    )
    _widget = section.build_widget()
    # Switch to MATLAB_HOME programmatically (index 1)
    assert section._combo is not None
    section._combo.setCurrentIndex(1)
    assert saved and saved[-1] == "MATLAB_HOME"


def test_workspace_section_unknown_default_falls_through(qt_real, qapp) -> None:  # noqa: ARG001
    """An unknown default leaves the combo at its first item."""
    section = WorkspaceSection(get_default=lambda: "NONEXISTENT")
    _widget = section.build_widget()
    assert section.selected_mode == LAYOUT_MODES[0][0]
