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
from .meshcat_segment_colors import MeshcatSegmentColorBindings


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


class MeshcatForceColors(MeshcatSegmentColorBindings):
    """Color only explicitly bound objects, preserving their supplied base alpha."""

    def apply(self, frame: AxialLoadFrame | None, scale: ForceColorScale) -> None:
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        if frame is not None and not isinstance(frame, AxialLoadFrame):
            raise TypeError("frame must be AxialLoadFrame or None")
        values = None if frame is None else frame.values_n
        if not scale.enabled:
            values = None
        self.apply_map(
            values,
            lambda sample, base: scale.shade_rgb(sample, base),
            restore_on_missing=True,
        )
