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


class MeshcatForceColorSession:
    """Shared host lifecycle for explicitly qualified, synchronous load frames.

    Hosts call update on each displayed state. Consumers bind native leaf objects
    for that exact model and submit axial frames; model replacement clears both.
    No force source, body axis or scene path is inferred by this controller.
    """

    def __init__(self) -> None:
        self._model: object = None
        self._time_s = 0.0
        self._adapter: MeshcatForceColors | None = None
        self._frame: AxialLoadFrame | None = None
        self._scale = ForceColorScale()

    def update(self, model: object, time_s: float) -> None:
        """Synchronize the display clock and discard bindings for replaced models."""
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, Real)
            or not math.isfinite(time_s)
        ):
            raise ValueError("display time must be finite")
        if model is not self._model:
            self._adapter = None
            self._frame = None
            self._model = model
        self._time_s = float(time_s)
        self._apply()

    def bind(self, adapter: MeshcatForceColors, model: object) -> None:
        """Attach an adapter for the current model, restoring any previous binding."""
        if not isinstance(adapter, MeshcatForceColors):
            raise TypeError("adapter must be MeshcatForceColors")
        if model is None or model is not self._model:
            raise ValueError("bindings must identify the currently displayed model")
        if self._adapter is not None:
            self._adapter.apply(None, self._scale)
        self._adapter = adapter
        self._frame = None

    def set_frame(self, frame: AxialLoadFrame | None) -> None:
        """Submit qualified loads; stale frames restore original colors."""
        if frame is not None and not isinstance(frame, AxialLoadFrame):
            raise TypeError("frame must be AxialLoadFrame or None")
        self._frame = frame
        self._apply()

    def set_axial_color_scale(self, scale: ForceColorScale) -> None:
        """Apply shared settings immediately, including disabled restoration."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        self._scale = scale
        self._apply()

    def _apply(self) -> None:
        if self._adapter is None:
            return
        frame = self._frame
        if frame is not None and not math.isclose(
            frame.time_s, self._time_s, rel_tol=0, abs_tol=1e-12
        ):
            frame = None
        self._adapter.apply(frame, self._scale)


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
