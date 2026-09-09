"""Frame-aligned load data and a model-independent color display controller."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

from .contracts import ColorOverrideRenderer
from .force_colors import ForceColorScale


@dataclass(frozen=True)
class SegmentLoadSeries:
    """Immutable axial loads at declared sample times and segment sections.

    Producers own force extraction and synchronization with geometry. Positive
    values mean tension. None/nonfinite samples mean unavailable, never zero.
    Source describes the producer/section convention; motion alone is insufficient.
    """

    time_s: Sequence[float]
    values_n: Mapping[str, Sequence[float | None]]
    source: str
    units: str = "N"
    sign_convention: str = "tension-positive"

    def __post_init__(self) -> None:
        if self.units != "N" or self.sign_convention != "tension-positive":
            raise ValueError("loads require N units and tension-positive convention")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must describe the load producer and section")
        times = tuple(self.time_s)
        if not times or any(
            isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t)
            for t in times
        ):
            raise ValueError("time_s must contain finite real sample times")
        if any(right <= left for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("time_s must be strictly increasing")
        if not isinstance(self.values_n, Mapping):
            raise TypeError("values_n must be a segment-to-samples mapping")
        copied = {}
        for segment, values in self.values_n.items():
            if not isinstance(segment, str) or not segment.strip():
                raise ValueError("segment identifiers must be nonempty strings")
            samples = tuple(values)
            if len(samples) != len(times):
                raise ValueError(f"{segment}: load sample count must match time_s")
            if any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, Real))
                for value in samples
            ):
                raise TypeError("load samples must be real numbers or None")
            copied[segment] = tuple(
                float(value) if value is not None and math.isfinite(value) else None
                for value in samples
            )
        object.__setattr__(self, "time_s", tuple(float(t) for t in times))
        object.__setattr__(self, "values_n", MappingProxyType(copied))


class ForceColorDisplay:
    """Apply one scale through a narrow renderer capability, independent of model.

    Handles are supplied by the host after adding shapes. Configure and set_loads
    immediately redraw colors at the current frame; replacing loads resets to zero.
    """

    def __init__(
        self, renderer: ColorOverrideRenderer, segment_handles: Mapping[str, str]
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
        self._renderer = renderer
        self._handles = dict(segment_handles)
        self._scale = ForceColorScale()
        self._loads: SegmentLoadSeries | None = None
        self._frame = 0

    def configure(self, scale: ForceColorScale) -> None:
        """Apply validated settings immediately, including restoring colors on off."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        self._scale = scale
        self._apply()

    def set_loads(self, loads: SegmentLoadSeries | None) -> None:
        """Replace a recording and reset frame; None clears stale force colors."""
        if loads is not None and not isinstance(loads, SegmentLoadSeries):
            raise TypeError("loads must be SegmentLoadSeries or None")
        self._loads = loads
        self._frame = 0
        self._apply()

    def update_frame(self, frame_idx: int) -> None:
        """Recolor on seek/playback; invalid indices do not mutate the display."""
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, Integral):
            raise TypeError("frame_idx must be an integer")
        if frame_idx < 0 or (
            self._loads is not None and frame_idx >= len(self._loads.time_s)
        ):
            raise IndexError("frame_idx outside load recording")
        self._frame = int(frame_idx)
        self._apply()

    def _apply(self) -> None:
        for segment, handle in self._handles.items():
            samples = None if self._loads is None else self._loads.values_n.get(segment)
            value = None if samples is None else samples[self._frame]
            color = (
                self._scale.color(value, "")
                if self._scale.enabled and value is not None
                else ""
            )
            self._renderer.set_color(handle, color or None)
