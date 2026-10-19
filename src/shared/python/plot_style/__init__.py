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
  scatter backend) and ``PyQtGLMarkerRenderer``, both lazy-loaded via
  ``__getattr__`` to keep matplotlib and PyQt6 optional.
- ``resolvers`` — Static / Palette / DataDriven color resolvers
  registered in ``RESOLVER_REGISTRY``.
- ``widgets`` — optional Qt widgets (``MarkerStylePicker``,
  ``ColorPicker``, ``ColormapPicker``, ``DataChannelEditor``) imported
  lazily on first attribute access; their names join ``__all__`` via
  the UNION resolution from #4807 when PyQt6 is installed.
- ``persistence`` / ``preset_library`` — ``PlotStyleSet`` /
  ``PlotStyleSpec`` JSON v1 schema and the built-in preset library.

See ADR-0011 (Plot Style Toolkit). Epic #4796 shipped the full stack.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
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

# Qt widgets and the matplotlib renderer are optional, heavy surfaces.
# They are re-exported lazily through ``__getattr__`` so that headless
# consumers (force_overlay, loop_nullspace, docs builder) can import
# this package without loading PyQt6 or matplotlib.
_LAZY_EXPORTS: dict[str, str] = {
    "ColormapPicker": ".widgets",
    "ColorPicker": ".widgets",
    "DataChannelEditor": ".widgets",
    "MarkerStylePicker": ".widgets",
    "MatplotlibMarkerRenderer": ".renderers.matplotlib",
    "PyQtGLMarkerRenderer": ".renderers.pyqtgl",
}


# Optional Qt widgets join ``__all__`` only when PyQt6 is already loaded in
# sys.modules. Headless consumers (force_overlay, loop_nullspace, docs)
# never query find_spec("PyQt6") or trigger import-blocker guards at module
# load time; interactive/GUI consumers access them lazily via __getattr__.
_WIDGET_NAMES: tuple[str, ...] = (
    ("ColormapPicker", "ColorPicker", "DataChannelEditor", "MarkerStylePicker")
    if "PyQt6" in sys.modules
    else ()
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .renderers.matplotlib import MatplotlibMarkerRenderer
    from .renderers.pyqtgl import PyQtGLMarkerRenderer
    from .widgets import (
        ColormapPicker,
        ColorPicker,
        DataChannelEditor,
        MarkerStylePicker,
    )

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
    "FORCE_KIND_PALETTE",
    "DataChannel",
    "DataDrivenColor",
    "FORCE_KIND_PALETTE",
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
    "hex_to_rgba",
    "list_colormaps",
    "magnitude_channel",
    "register_custom_colormap",
    "resolve_colormap_alias",
    "rgba_to_hex",
    "slice_channel",
    "unregister_custom_colormap",
]
__all__.extend(_WIDGET_NAMES)
__all__.sort()


def __getattr__(name: str) -> Any:
    """Lazy attribute access for optional / heavy renderers and widgets."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, name)
    except (ImportError, OSError) as exc:
        raise AttributeError(
            f"module {__name__!r} attribute {name!r} is unavailable: {exc}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
