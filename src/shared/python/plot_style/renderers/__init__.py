"""Renderer implementations sub-package.

Concrete :class:`~src.shared.python.plot_style.contracts.MarkerRenderer`
backends. Currently bundled:

* :class:`MatplotlibMarkerRenderer` — 2D ``Axes`` and 3D ``Axes3D``.
* :class:`PyQtGLMarkerRenderer` — pyqtgraph / OpenGL 3D backend, gated
  behind the optional ``body-part-viz-gl`` extra. Importing the symbol
  directly from this package will raise ``ImportError`` if the extra is
  not installed; that import is lazy so ``import plot_style`` itself
  stays cheap and dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .matplotlib import MatplotlibMarkerRenderer
    from .pyqtgl import PyQtGLMarkerRenderer

__all__ = ["MatplotlibMarkerRenderer", "PyQtGLMarkerRenderer"]

_RENDERER_MODULES: dict[str, tuple[str, str]] = {
    "MatplotlibMarkerRenderer": (".matplotlib", "MatplotlibMarkerRenderer"),
    "PyQtGLMarkerRenderer": (".pyqtgl", "PyQtGLMarkerRenderer"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute access for optional / heavy renderers."""
    target = _RENDERER_MODULES.get(name)
    if target is not None:
        module_path, symbol = target
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        val = getattr(module, symbol)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_RENDERER_MODULES.keys()))
