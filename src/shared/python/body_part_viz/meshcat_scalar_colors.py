"""MeshCat scalar colouring for explicitly bound segment leaves (#15396)."""

from __future__ import annotations

from collections.abc import Mapping, MutableSet

from .meshcat_segment_colors import MeshcatSegmentColorBindings
from .scalar_shading import ScalarColorScale


class MeshcatScalarColors(MeshcatSegmentColorBindings):
    """Colour bound MeshCat leaves from a synchronous scalar sample map."""

    def apply(
        self,
        values: Mapping[str, float | None] | None,
        scale: ScalarColorScale,
        *,
        clipped_segments: MutableSet[str] | None = None,
    ) -> None:
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        if values is not None and not isinstance(values, Mapping):
            raise TypeError("values must be a mapping or None")
        active_values = values if scale.enabled else None

        def _saturated(sample: float | None) -> bool:
            return scale.evaluate(sample, "#000000").saturated

        self.apply_map(
            active_values,
            lambda sample, base: scale.shade_rgb(sample, base),
            restore_on_missing=False,
            clipped_segments=clipped_segments,
            saturation_flag=_saturated,
        )


__all__ = ["MeshcatScalarColors"]
