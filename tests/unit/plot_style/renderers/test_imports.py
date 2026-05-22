"""Import and lazy-export tests for plot_style renderers."""

from __future__ import annotations

import sys
import types

import pytest


def test_renderers_exports_matplotlib_renderer() -> None:
    from src.shared.python.plot_style import renderers
    from src.shared.python.plot_style.renderers.matplotlib import (
        MatplotlibMarkerRenderer,
    )

    assert renderers.MatplotlibMarkerRenderer is MatplotlibMarkerRenderer
    assert "MatplotlibMarkerRenderer" in renderers.__all__


def test_renderers_lazy_pyqtgl_export_uses_optional_module(monkeypatch) -> None:
    from src.shared.python.plot_style import renderers

    class FakePyQtGLMarkerRenderer:
        pass

    fake_module = types.ModuleType("src.shared.python.plot_style.renderers.pyqtgl")
    fake_module.PyQtGLMarkerRenderer = FakePyQtGLMarkerRenderer
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    assert renderers.__getattr__("PyQtGLMarkerRenderer") is FakePyQtGLMarkerRenderer
    assert "PyQtGLMarkerRenderer" in renderers.__all__


def test_renderers_lazy_unknown_export_raises_attribute_error() -> None:
    from src.shared.python.plot_style import renderers

    with pytest.raises(AttributeError, match="NotARenderer"):
        renderers.__getattr__("NotARenderer")
