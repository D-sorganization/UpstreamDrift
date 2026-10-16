"""The C3D viewer must build embedded, not via the subprocess fallback (#12995)."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("matplotlib")


@pytest.fixture
def qapp_or_skip():
    """Return a QApplication, creating one if the suite has not already."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.unit
def test_apps_package_relative_imports_reach_src() -> None:
    """``from ...c3d_reader`` inside apps.services must resolve."""
    from src.launchers.adapters.simscape_embed import (
        APPS_PACKAGE_NAME,
        load_simscape_apps_package,
    )

    load_simscape_apps_package()
    mod = importlib.import_module(f"{APPS_PACKAGE_NAME}.services.c3d_loader")
    assert hasattr(mod, "C3DDataReader")


@pytest.mark.unit
def test_embedded_c3d_viewer_creates_widget(qapp_or_skip) -> None:
    """``create_main_widget`` returns a QWidget instead of raising ImportError."""
    from PyQt6.QtWidgets import QWidget

    from src.launchers.adapters.simscape_embed import load_simscape_apps_package
    from src.shared.python.launcher_embed import get_embeddable_tool

    load_simscape_apps_package()
    tool = get_embeddable_tool("c3d_viewer")
    assert tool is not None
    widget = tool.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        tool.cleanup()
        widget.deleteLater()
