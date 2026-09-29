"""Regression tests for the writable delegated state on OpenSimGolfGUI.

Issue #5071: a refactor previously turned ``model``, ``model_path``,
``result``, and ``initialization_error`` into read-only ``@property``
shims that delegated to the embedded :class:`MainWidget`. That broke
existing callers (and test harnesses) that assigned new values onto a
window instance after construction. These tests pin the public surface
back to "writable, with delegation".
"""

from __future__ import annotations

import importlib

import pytest

# The dashboard pulls in PyQt6 and Matplotlib's Qt backend. Skip the
# whole module rather than failing collection on hosts where PyQt6 is
# missing.
PyQt6 = pytest.importorskip("PyQt6")

_GUI_MOD_NAME = "src.engines.physics_engines.opensim.python.opensim_gui"


def _import_gui_module():
    try:
        return importlib.import_module(_GUI_MOD_NAME)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"OpenSim GUI module not importable: {exc}")


@pytest.fixture
def gui(qapp):
    """Construct a real ``OpenSimGolfGUI`` window with no model."""
    gui_mod = _import_gui_module()
    window = gui_mod.OpenSimGolfGUI(model_path=None)
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.unit
def test_model_path_is_writable(gui) -> None:
    """Assigning ``model_path`` must propagate to the inner widget."""
    gui.model_path = "/tmp/example.osim"
    assert gui.model_path == "/tmp/example.osim"
    assert gui._main_widget.model_path == "/tmp/example.osim"


@pytest.mark.unit
def test_model_is_writable(gui) -> None:
    """Test harnesses that inject ``model`` must continue to work."""
    sentinel = object()
    gui.model = sentinel  # type: ignore[assignment]
    assert gui.model is sentinel
    assert gui._main_widget.model is sentinel


@pytest.mark.unit
def test_result_is_writable(gui) -> None:
    """Assigning ``result`` must propagate to the inner widget."""
    sentinel = object()
    gui.result = sentinel
    assert gui.result is sentinel
    assert gui._main_widget.result is sentinel


@pytest.mark.unit
def test_initialization_error_is_writable(gui) -> None:
    """Assigning ``initialization_error`` must propagate."""
    gui.initialization_error = "boom"
    assert gui.initialization_error == "boom"
    assert gui._main_widget.initialization_error == "boom"
