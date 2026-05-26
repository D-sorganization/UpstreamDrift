"""Plot-style contracts, dataclasses, and renderer implementations.

This package owns the canonical marker-styling stack used across every
plotting surface in UpstreamDrift (C3D Viewer, Motion-Match Preview,
cross-engine dashboard, per-engine viz modules):

- ``contracts`` — runtime-checkable Protocols (``ColorResolver``,
  ``MarkerRenderer``, ``MarkerShapeRenderer``).
- ``colors`` / ``colormaps`` / ``registry`` — color scales, semantic
  colormap aliases, and the custom-colormap registry.
- ``channels`` — ``DataChannel`` abstraction for per-frame scalar
  sources (e.g. clubhead-velocity-magnitude) plus derivative / slice /
  magnitude helpers.
- ``markers`` / ``shapes`` — ``MarkerStyle``, ``MarkerShape``,
  ``CustomMeshSpec`` and the built-in shape primitives.
- ``renderers`` — ``MatplotlibMarkerRenderer`` (canonical 2D/3D
  scatter backend) and ``PyQtGLMarkerRenderer`` (lazy-loaded via
  ``__getattr__`` to keep PyQt6 optional).
- ``resolvers`` — Static / Palette / DataDriven color resolvers
  registered in ``RESOLVER_REGISTRY``.
- ``widgets`` — optional Qt widgets (``MarkerStylePicker``,
  ``ColorPicker``, ``ColormapPicker``, ``DataChannelEditor``) imported
  only when PyQt6 is available; their names join ``__all__`` via the
  UNION resolution from #4807.
- ``persistence`` / ``preset_library`` — ``PlotStyleSet`` /
  ``PlotStyleSpec`` JSON v1 schema and the built-in preset library.

See ADR-0011 (Plot Style Toolkit). Epic #4796 shipped the full stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import RGBATuple
from .channels import (
    DataChannel,
    derivative_channel,
    magnitude_channel,
    slice_channel,
)
from .colormaps import (
    SEMANTIC_COLORMAP_ALIASES,
    ColormapId,
    CustomColormap,
    resolve_colormap_alias,
)
from .colors import ColorScale, DataDrivenColor, PaletteColor, StaticColor
from .contracts import ColorResolver, MarkerRenderer, MarkerShapeRenderer
from .markers import CustomMeshSpec, MarkerShape, MarkerStyle
from .persistence import SCHEMA_VERSION, PlotStyleSet, PlotStyleSpec
from .preset_library import BUILTIN_PRESET_NAMES, PresetLibrary
from .registry import (
    get_colormap,
    list_colormaps,
    register_custom_colormap,
    unregister_custom_colormap,
)
from .renderers import MatplotlibMarkerRenderer
from .resolvers import RESOLVER_REGISTRY

# Qt widgets are an optional surface — we expose their names in
# ``__all__`` (UNION resolution per #4807) but only import them when
# PyQt6 is available so that headless / non-GUI consumers can still
# import this package.
try:
    from .widgets import (  # noqa: F401
        ColormapPicker,
        ColorPicker,
        DataChannelEditor,
        MarkerStylePicker,
    )

    _WIDGET_NAMES: tuple[str, ...] = (
        "ColormapPicker",
        "ColorPicker",
        "DataChannelEditor",
        "MarkerStylePicker",
    )
except Exception:  # pragma: no cover - optional GUI dependency (PyQt6)  # noqa: BLE001
    _WIDGET_NAMES = ()

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .renderers.pyqtgl import PyQtGLMarkerRenderer

__all__ = [
    "BUILTIN_PRESET_NAMES",
    "RESOLVER_REGISTRY",
    "SCHEMA_VERSION",
    "SEMANTIC_COLORMAP_ALIASES",
    "ColorResolver",
    "ColorScale",
    "ColormapId",
    "CustomColormap",
    "CustomMeshSpec",
    "DataChannel",
    "DataDrivenColor",
    "MarkerRenderer",
    "MarkerShape",
    "MarkerShapeRenderer",
    "MarkerStyle",
    "MatplotlibMarkerRenderer",
    "PaletteColor",
    "PlotStyleSet",
    "PlotStyleSpec",
    "PresetLibrary",
    "PyQtGLMarkerRenderer",
    "RGBATuple",
    "StaticColor",
    "derivative_channel",
    "get_colormap",
    "list_colormaps",
    "magnitude_channel",
    "register_custom_colormap",
    "resolve_colormap_alias",
    "slice_channel",
    "unregister_custom_colormap",
]
# UNION resolution (#4807): widget names are appended only when the
# optional PyQt6 import succeeded above.
__all__.extend(_WIDGET_NAMES)
__all__.sort()


def __getattr__(name: str) -> Any:
    """Lazy attribute access for optional / heavy renderers."""
    if name == "PyQtGLMarkerRenderer":
        from .renderers.pyqtgl import (  # noqa: PLC0415 - intentional lazy import
            PyQtGLMarkerRenderer,
        )

        return PyQtGLMarkerRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
