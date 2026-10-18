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
  ``__getattr__`` to keep headless imports fast and dependency-free).
- ``resolvers`` — Static / Palette / DataDriven color resolvers
  registered in ``RESOLVER_REGISTRY``.
- ``widgets`` — optional Qt widgets (``MarkerStylePicker``,
  ``ColorPicker``, ``ColormapPicker``, ``DataChannelEditor``) loaded
  lazily via ``__getattr__`` to keep PyQt6 optional for headless
  consumers (#13253).
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
from .color_utils import hex_to_rgba, rgba_to_hex
from .colormaps import (
    SEMANTIC_COLORMAP_ALIASES,
    ColormapId,
    CustomColormap,
    resolve_colormap_alias,
)
from .colors import (
    ColorScale,
    DataDrivenColor,
    FORCE_KIND_PALETTE,
    PaletteColor,
    StaticColor,
)
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
from .resolvers import RESOLVER_REGISTRY

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ColorPicker": (".widgets", "ColorPicker"),
    "ColormapPicker": (".widgets", "ColormapPicker"),
    "DataChannelEditor": (".widgets", "DataChannelEditor"),
    "MarkerStylePicker": (".widgets", "MarkerStylePicker"),
    "MatplotlibMarkerRenderer": (".renderers.matplotlib", "MatplotlibMarkerRenderer"),
    "PyQtGLMarkerRenderer": (".renderers.pyqtgl", "PyQtGLMarkerRenderer"),
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .renderers.matplotlib import MatplotlibMarkerRenderer
    from .renderers.pyqtgl import PyQtGLMarkerRenderer
    from .widgets import (
        ColorPicker,
        ColormapPicker,
        DataChannelEditor,
        MarkerStylePicker,
    )

__all__ = sorted([
    "BUILTIN_PRESET_NAMES",
    "ColorPicker",
    "ColorResolver",
    "ColorScale",
    "ColormapId",
    "ColormapPicker",
    "CustomColormap",
    "CustomMeshSpec",
    "DataChannel",
    "DataChannelEditor",
    "DataDrivenColor",
    "FORCE_KIND_PALETTE",
    "MarkerRenderer",
    "MarkerShape",
    "MarkerShapeRenderer",
    "MarkerStyle",
    "MarkerStylePicker",
    "MatplotlibMarkerRenderer",
    "PaletteColor",
    "PlotStyleSet",
    "PlotStyleSpec",
    "PresetLibrary",
    "PyQtGLMarkerRenderer",
    "RGBATuple",
    "RESOLVER_REGISTRY",
    "SCHEMA_VERSION",
    "SEMANTIC_COLORMAP_ALIASES",
    "StaticColor",
    "derivative_channel",
    "get_colormap",
    "hex_to_rgba",
    "list_colormaps",
    "magnitude_channel",
    "register_custom_colormap",
    "resolve_colormap_alias",
    "rgba_to_hex",
    "slice_channel",
    "unregister_custom_colormap",
])


def __getattr__(name: str) -> Any:
    """Lazy attribute access for optional / heavy renderers and widgets."""
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_path, symbol = target
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        val = getattr(module, symbol)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
