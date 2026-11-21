"""MeshCat transport for quantity-agnostic segment scalar shading (#15399)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

from .scalar_shading import ScalarColorScale
from .segment_scalar_frame import SegmentScalarFrame

__all__ = ["MeshcatSegmentScalarColors"]


class MeshcatSegmentScalarColors:
    """Colour explicitly bound leaf objects from per-segment scalars."""

    def __init__(
        self,
        set_property: Callable[[str, str, list[float]], object],
        bindings: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> None:
        if not callable(set_property) or not isinstance(bindings, Mapping):
            raise TypeError("a property setter and binding mapping are required")
        self._setter = set_property
        self._bindings: dict[str, dict[str, tuple[float, ...]]] = {}
        paths: set[str] = set()
        for segment, objects in bindings.items():
            if (
                not isinstance(segment, str)
                or not segment
                or not isinstance(objects, Mapping)
            ):
                raise ValueError(
                    "bindings require nonempty segment IDs and object maps"
                )
            copied = {}
            for path, rgba in objects.items():
                if (
                    not isinstance(path, str)
                    or not path.endswith("/<object>")
                    or path in paths
                ):
                    raise ValueError("each leaf object path must be bound exactly once")
                values = tuple(rgba)
                if len(values) != 4 or any(
                    isinstance(v, bool)
                    or not isinstance(v, Real)
                    or not math.isfinite(v)
                    or not 0.0 <= float(v) <= 1.0
                    for v in values
                ):
                    raise ValueError(
                        "base RGBA must have four finite channels in [0, 1]"
                    )
                copied[path] = tuple(float(v) for v in values)
                paths.add(path)
            self._bindings[segment] = copied
        self._overridden: set[str] = set()

    def apply(self, frame: SegmentScalarFrame | None, scale: ScalarColorScale) -> None:
        """Apply synchronous scalars, or restore previously overridden colors."""
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        if frame is not None and not isinstance(frame, SegmentScalarFrame):
            raise TypeError("frame must be SegmentScalarFrame or None")
        values = None if frame is None else frame.values
        for segment, objects in self._bindings.items():
            sample = None if values is None else values.get(segment)
            for path, base in objects.items():
                rgb = scale.shade_rgb(sample, base[:3])
                if rgb is not None:
                    self._setter(path, "color", [*rgb, base[3]])
                    self._overridden.add(path)
                elif path in self._overridden:
                    self._setter(path, "color", list(base))
                    self._overridden.remove(path)
