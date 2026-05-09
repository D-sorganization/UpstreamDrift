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
from .registry import (
    get_colormap,
    list_colormaps,
    register_custom_colormap,
    unregister_custom_colormap,
)
from .renderers import MatplotlibMarkerRenderer

__all__ = [
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
