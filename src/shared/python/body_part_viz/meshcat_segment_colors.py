"""Shared MeshCat leaf bindings for per-segment colour overlays."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableSet, Sequence
from numbers import Real


class MeshcatSegmentColorBindings:
    """Colour explicitly bound leaf objects; optional restore on missing samples."""

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
            copied: dict[str, tuple[float, ...]] = {}
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
                    or not 0.0 <= float(v) <= 1
                    for v in values
                ):
                    raise ValueError(
                        "base RGBA must have four finite channels in [0, 1]"
                    )
                copied[path] = tuple(float(v) for v in values)
                paths.add(path)
            self._bindings[segment] = copied
        self._overridden: set[str] = set()

    def apply_map(
        self,
        values: Mapping[str, float | None] | None,
        shade_rgb: Callable[
            [float | None, tuple[float, float, float]],
            tuple[float, float, float] | None,
        ],
        *,
        restore_on_missing: bool,
        clipped_segments: MutableSet[str] | None = None,
        saturation_flag: Callable[[float | None], bool] | None = None,
    ) -> None:
        for segment, objects in self._bindings.items():
            sample = None if values is None else values.get(segment)
            for path, base in objects.items():
                rgb = shade_rgb(sample, base[:3])
                if rgb is not None:
                    self._setter(path, "color", [*rgb, base[3]])
                    self._overridden.add(path)
                    if (
                        clipped_segments is not None
                        and saturation_flag is not None
                        and saturation_flag(sample)
                    ):
                        clipped_segments.add(segment)
                elif restore_on_missing and path in self._overridden:
                    self._setter(path, "color", list(base))
                    self._overridden.remove(path)


__all__ = ["MeshcatSegmentColorBindings"]
