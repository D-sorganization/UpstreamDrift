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
def test_embedded_c3d_viewer_creates_widget(qapp_or_skip, monkeypatch) -> None:
    """``create_main_widget`` returns a QWidget instead of raising ImportError."""
    import sys

    from PyQt6.QtWidgets import QWidget

    from src.launchers.adapters.simscape_embed import (
        APPS_PACKAGE_NAME,
        TOOL_ID,
        load_simscape_apps_package,
    )
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )

    # Another test in the same worker may have registered the adapter from a
    # different import path; exercise the launcher's own loader in isolation.
    monkeypatch.delitem(EMBEDDABLE_TOOL_REGISTRY, TOOL_ID, raising=False)
    monkeypatch.delitem(sys.modules, APPS_PACKAGE_NAME, raising=False)
    load_simscape_apps_package()
    tool = get_embeddable_tool("c3d_viewer")
    assert tool is not None
    widget = tool.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        tool.cleanup()
        widget.deleteLater()
