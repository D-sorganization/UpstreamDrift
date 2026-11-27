"""Frame-aligned load data and a model-independent color display controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from .contracts import ColorOverrideRenderer
from .force_colors import ForceColorScale
from .series_validation import (
    normalize_segment_samples,
    validate_strictly_increasing_times,
)


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
        times = validate_strictly_increasing_times(self.time_s)
        copied = normalize_segment_samples(
            times,
            self.values_n,
            segment_value_label="load sample count",
        )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "values_n", copied)

    def value_at(self, segment: str, frame_idx: int) -> float | None:
        """Read a segment sample without exposing the series' storage layout."""
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, Integral):
            raise TypeError("frame_idx must be an integer")
        if not 0 <= frame_idx < len(self.time_s):
            raise IndexError("frame_idx outside load recording")
        samples = self.values_n.get(segment)
        return None if samples is None else samples[frame_idx]


class ForceColorDisplay:
    """Apply one scale through a narrow renderer capability, independent of model.

    Handles are supplied by the host after adding shapes. Configure and set_loads
    immediately redraw colors at the current frame; replacing loads resets to zero.
    """

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
        # Segment base colors let translucent overlays blend with the original;
        # segments without one blend against the scale's neutral color.
        self._base_colors = dict(base_colors or {})
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
            value = (
                None
                if self._loads is None
                else self._loads.value_at(segment, self._frame)
            )
            base = self._base_colors.get(segment, self._scale.neutral_color)
            color = self._scale.shade_hex(value, base)
            self._renderer.set_color(handle, color)
