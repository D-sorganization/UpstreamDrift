"""Plot-style contracts and dataclasses.

This package defines the abstract surface (Protocols + frozen
dataclasses) used by every marker / trace styling implementation in
UpstreamDrift.

Concrete implementations of color resolvers, renderers, Qt widgets, and
marker-shape primitives live in the ``resolvers``, ``renderers``,
``widgets``, and ``shapes`` sub-packages and are added in follow-up
issues of EPIC #4796.
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
except Exception:  # pragma: no cover - optional GUI dependency (PyQt6)
    _WIDGET_NAMES = ()

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .renderers.pyqtgl import PyQtGLMarkerRenderer

__all__ = [
    "BUILTIN_PRESET_NAMES",
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
