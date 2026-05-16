"""Tests for :mod:`src.launchers.preferences.jupyter_section`."""

from __future__ import annotations

import pytest

from src.launchers.preferences.jupyter_section import JupyterSection


def test_jupyter_section_build_widget(qt_real, qapp) -> None:  # noqa: ARG001
    section = JupyterSection()
    widget = section.build_widget()
    assert widget.objectName() == "prefs_section_jupyter"


def test_jupyter_section_unavailable_path(qt_real, qapp, monkeypatch) -> None:  # noqa: ARG001
    """When nbformat is unavailable the section renders an info label."""
    import src.launchers.preferences.jupyter_section as mod

    monkeypatch.setattr(mod, "is_feature_available", lambda _id: False)
    section = JupyterSection()
    widget = section.build_widget()
    assert widget.objectName() == "prefs_section_jupyter"
    # Inputs are NOT created on the unavailable path:
    assert section._dir_edit is None
    assert section._kernel_edit is None


def test_jupyter_section_available_persists(qt_real, qapp, monkeypatch) -> None:  # noqa: ARG001
    """Editing the directory line edit persists via set_notebook_dir."""
    import src.launchers.preferences.jupyter_section as mod

    monkeypatch.setattr(mod, "is_feature_available", lambda _id: True)
    captured: list[str] = []
    section = JupyterSection(
        get_notebook_dir=lambda: "/tmp/nb",
        set_notebook_dir=lambda v: captured.append(v),
    )
    section.build_widget()
    assert section._dir_edit is not None
    section._dir_edit.setText("/var/tmp/notebooks")
    section._dir_edit.editingFinished.emit()
    assert captured and captured[-1] == "/var/tmp/notebooks"


def test_jupyter_section_kernel_persistence(qt_real, qapp, monkeypatch) -> None:  # noqa: ARG001
    import src.launchers.preferences.jupyter_section as mod

    monkeypatch.setattr(mod, "is_feature_available", lambda _id: True)
    saved_kernel: list[str] = []
    section = JupyterSection(
        get_kernel=lambda: "python3",
        set_kernel=lambda v: saved_kernel.append(v),
    )
    section.build_widget()
    assert section._kernel_edit is not None
    section._kernel_edit.setText("ir")
    section._kernel_edit.editingFinished.emit()
    assert saved_kernel and saved_kernel[-1] == "ir"


@pytest.mark.parametrize("available", [True, False])
def test_jupyter_section_reports_availability(qapp, monkeypatch, available) -> None:  # noqa: ARG001
    import src.launchers.preferences.jupyter_section as mod

    monkeypatch.setattr(mod, "is_feature_available", lambda _id: available)
    section = JupyterSection()
    assert section.is_available is available
