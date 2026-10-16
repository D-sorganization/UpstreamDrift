"""OpenSim tab names a DLL conflict instead of 'Not Installed' (#12883)."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PyQt6")

from src.shared.python.data_io.native_dll_guard import NativeDllConflictError  # noqa: E402

_GUI_MOD_NAME = "src.engines.physics_engines.opensim.python.opensim_gui"
_RUNTIME = "src.shared.python.engine_core.opensim_runtime"


def _make_gui(qapp, monkeypatch, exc: BaseException | None):
    gui_mod = importlib.import_module(_GUI_MOD_NAME)
    runtime = importlib.import_module(_RUNTIME)

    def _fake_import(*_a, **_k):
        if exc is not None:
            raise exc
        return object()

    monkeypatch.setattr(
        gui_mod,
        "probe_opensim",
        lambda: runtime.probe_opensim(_fake_import),
    )
    return gui_mod.OpenSimGolfGUI(model_path=None)


@pytest.mark.unit
def test_dll_conflict_status_names_the_conflict(qapp, monkeypatch) -> None:
    msg = "DLL load failed: ezc3d.dll is already loaded"
    window = _make_gui(qapp, monkeypatch, NativeDllConflictError(msg))
    try:
        status = window._main_widget.lbl_status.text()
        details = window._main_widget.lbl_details.text()
        assert "Not Installed" not in status
        assert "Cannot Load" in status
        assert msg in details
    finally:
        window.close()


@pytest.mark.unit
def test_missing_opensim_still_reports_not_installed(qapp, monkeypatch) -> None:
    window = _make_gui(
        qapp, monkeypatch, ModuleNotFoundError("No module named 'opensim'")
    )
    try:
        assert "Not Installed" in window._main_widget.lbl_status.text()
    finally:
        window.close()
