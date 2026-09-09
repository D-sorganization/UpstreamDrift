"""Shared MeshCat color transport for explicit segment-to-object bindings.

Pass ``lambda path, prop, value: viewer[path].set_property(prop, value)``
for meshcat-python, or Drake's public ``meshcat.SetProperty`` method. Objects
must already exist; paths must address leaf objects, never parent containers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

from .axial_loads import AxialLoadFrame
from .force_colors import ForceColorScale


class MeshcatForceColors:
    """Color only explicitly bound objects, preserving their supplied base alpha.

    The host owns geometry, bindings and current base RGBA. Recreate this adapter
    after material/model replacement. No native or engine imports are needed.
    MeshCat's documented color property uses four normalized RGBA channels.
    """

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
                    or not 0 <= float(v) <= 1
                    for v in values
                ):
                    raise ValueError(
                        "base RGBA must have four finite channels in [0, 1]"
                    )
                copied[path] = tuple(float(v) for v in values)
                paths.add(path)
            self._bindings[segment] = copied
        self._overridden: set[str] = set()

    def apply(self, frame: AxialLoadFrame | None, scale: ForceColorScale) -> None:
        """Apply synchronous loads, or restore previously overridden base colors."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        if frame is not None and not isinstance(frame, AxialLoadFrame):
            raise TypeError("frame must be AxialLoadFrame or None")
        for segment, objects in self._bindings.items():
            force = None if frame is None else frame.values_n.get(segment)
            color = scale.color(force, "")
            for path, base in objects.items():
                if color:
                    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
                    self._setter(path, "color", [*rgb, base[3]])
                    self._overridden.add(path)
                elif path in self._overridden:
                    self._setter(path, "color", list(base))
                    self._overridden.remove(path)
