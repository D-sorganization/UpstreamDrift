"""Model-independent scalar colour display controller (#15396)."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

from .contracts import ColorOverrideRenderer
from .scalar_shading import ScalarColorScale


class ScalarColorDisplay:
    """Apply one :class:`ScalarColorScale` through :class:`ColorOverrideRenderer`."""

    def __init__(
        self,
        renderer: ColorOverrideRenderer,
        segment_handles: Mapping[str, str],
        base_colors: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(renderer, ColorOverrideRenderer):
            raise TypeError("renderer must support set_color(handle, color)")
        if not isinstance(segment_handles, Mapping):
            raise TypeError("segment_handles must be a mapping")
        if any(
            not isinstance(value, str) or not value
            for pair in segment_handles.items()
            for value in pair
        ):
            raise ValueError("segment identifiers and handles must be nonempty strings")
        if len(set(segment_handles.values())) != len(segment_handles):
            raise ValueError("each segment must have a distinct renderer handle")
        if base_colors is not None and not isinstance(base_colors, Mapping):
            raise TypeError("base_colors must be a segment-to-#RRGGBB mapping")
        self._renderer = renderer
        self._handles = dict(segment_handles)
        self._base_colors = dict(base_colors or {})
        self._scale = ScalarColorScale()
        self._values: Mapping[str, float | None] | None = None

    def configure(self, scale: ScalarColorScale) -> None:
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        self._scale = scale
        self._apply()

    def set_values(self, values: Mapping[str, float | None] | None) -> None:
        if values is not None and not isinstance(values, Mapping):
            raise TypeError("values must be a mapping or None")
        self._values = None if values is None else dict(values)
        self._apply()

    def _apply(self) -> None:
        for segment, handle in self._handles.items():
            sample = None if self._values is None else self._values.get(segment)
            base = self._base_colors.get(segment, self._scale.neutral_color)
            color = self._scale.shade_hex(sample, base)
            if color is None:
                color = base
            self._renderer.set_color(handle, color)
