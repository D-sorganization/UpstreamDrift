"""Time alignment across cameras that share no exposure clock.

Three cameras on three USB root ports each stamp frames in the host's
monotonic clock at *arrival*. ADR-0041 forbids promoting arrival time to
exposure time, so this module never rewrites a frame's timestamp. It measures
the relationship between cameras — from a strobe that every camera sees in the
same instant — and records the result, with its uncertainty, in the session
manifest for the reconstruction stage to apply or reject.

Uncertainty model: a flash that lands anywhere inside one frame interval is
first visible in the next frame, so a single event locates the strobe to
within one interval of that camera. Offsets between two cameras therefore
carry both cameras' intervals, combined in quadrature.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

from .sources import HOST_MONOTONIC

TimingStatus = Literal["available", "unavailable"]
FLASH_METHOD = "flash_event"


@dataclass(frozen=True)
class EventEstimate:
    """When a strobe became visible to one camera, in that camera's frame times."""

    index: int
    t_ns: int
    uncertainty_ns: int


class ViewTiming(BaseModel):
    """One camera's timing relative to the reference view."""

    model_config = ConfigDict(frozen=True)

    view: str
    status: TimingStatus
    frames: int
    frame_interval_ns: int | None = None
    rate_deviation_ppm: float | None = None
    event_index: int | None = None
    event_t_ns: int | None = None
    offset_ns: int | None = None
    uncertainty_ns: int | None = None
    reason: str | None = None


class TimingRecord(BaseModel):
    """Manifest ``timing`` block: how the cameras' arrival clocks relate."""

    model_config = ConfigDict(frozen=True)

    method: str = FLASH_METHOD
    clock_domain: str = HOST_MONOTONIC
    reference_view: str
    status: TimingStatus
    views: tuple[ViewTiming, ...]
    note: str = (
        "offsets relate arrival clocks via a shared strobe; they are evidence for "
        "the reconstruction stage, not corrections applied to frames"
    )


def _validate_series(t_ns: np.ndarray, brightness: np.ndarray) -> None:
    require(t_ns.ndim == 1 and brightness.ndim == 1, "series must be 1-D")
    require(
        len(t_ns) == len(brightness),
        "t_ns and brightness must align",
        (len(t_ns), len(brightness)),
    )
    require(len(t_ns) > 0, "series must be non-empty")


def median_interval_ns(t_ns: np.ndarray) -> int | None:
    """Median inter-frame interval, or ``None`` with fewer than two frames."""
    require(t_ns.ndim == 1, "t_ns must be 1-D")
    if len(t_ns) < 2:
        return None
    return int(np.median(np.diff(t_ns.astype(np.int64))))


def rate_deviation_ppm(t_ns: np.ndarray, nominal_fps: float) -> float | None:
    """Measured frame period versus nominal, in parts per million (+ = slow)."""
    require(nominal_fps > 0, "nominal_fps must be positive", nominal_fps)
    interval = median_interval_ns(t_ns)
    if interval is None:
        return None
    nominal_ns = 1e9 / nominal_fps
    return (interval - nominal_ns) / nominal_ns * 1e6


def detect_flash_index(
    brightness: np.ndarray, *, min_jump: float = 64.0, baseline_frames: int = 5
) -> int | None:
    """Index of the first frame at least ``min_jump`` brighter than the baseline.

    The baseline is the median of the first ``baseline_frames`` frames, which
    must therefore precede the strobe. Returns ``None`` when no frame qualifies.
    """
    require(
        brightness.ndim == 1 and len(brightness) > 0,
        "brightness must be 1-D, non-empty",
    )
    require(min_jump > 0, "min_jump must be positive", min_jump)
    require(baseline_frames >= 1, "baseline_frames must be >= 1", baseline_frames)
    values = brightness.astype(np.float64)
    baseline = float(np.median(values[:baseline_frames]))
    candidates = np.flatnonzero(values[baseline_frames:] - baseline >= min_jump)
    return int(candidates[0]) + baseline_frames if candidates.size else None


def flash_event(
    t_ns: np.ndarray,
    brightness: np.ndarray,
    *,
    min_jump: float = 64.0,
    baseline_frames: int = 5,
) -> EventEstimate | None:
    """Locate the strobe in one camera's series; uncertainty is one frame interval."""
    _validate_series(t_ns, brightness)
    index = detect_flash_index(
        brightness, min_jump=min_jump, baseline_frames=baseline_frames
    )
    if index is None:
        return None
    if index > 0:
        interval = int(t_ns[index]) - int(t_ns[index - 1])
    else:
        interval = median_interval_ns(t_ns) or 0
    return EventEstimate(
        index=index, t_ns=int(t_ns[index]), uncertainty_ns=max(interval, 0)
    )


@dataclass(frozen=True)
class _Detection:
    """Strobe-detection settings threaded through alignment."""

    min_jump: float = 64.0
    baseline_frames: int = 5


def _view_timing(
    view: str,
    t_ns: np.ndarray,
    brightness: np.ndarray,
    nominal_fps: float,
    reference: EventEstimate | None,
    detection: _Detection,
) -> ViewTiming:
    frames = int(len(t_ns))
    interval = median_interval_ns(t_ns)
    deviation = rate_deviation_ppm(t_ns, nominal_fps)
    event = (
        flash_event(
            t_ns,
            brightness,
            min_jump=detection.min_jump,
            baseline_frames=detection.baseline_frames,
        )
        if frames
        else None
    )
    if event is None:
        return ViewTiming(
            view=view,
            status="unavailable",
            frames=frames,
            frame_interval_ns=interval,
            rate_deviation_ppm=deviation,
            reason="no strobe detected",
        )
    if reference is None:
        return ViewTiming(
            view=view,
            status="unavailable",
            frames=frames,
            frame_interval_ns=interval,
            rate_deviation_ppm=deviation,
            event_index=event.index,
            event_t_ns=event.t_ns,
            reason="reference view has no strobe",
        )
    combined = math.hypot(event.uncertainty_ns, reference.uncertainty_ns)
    return ViewTiming(
        view=view,
        status="available",
        frames=frames,
        frame_interval_ns=interval,
        rate_deviation_ppm=deviation,
        event_index=event.index,
        event_t_ns=event.t_ns,
        offset_ns=event.t_ns - reference.t_ns,
        uncertainty_ns=int(round(combined)),
    )


def align_views(
    reference_view: str,
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    nominal_fps: Mapping[str, float],
    *,
    min_jump: float = 64.0,
    baseline_frames: int = 5,
) -> TimingRecord:
    """Relate every view's arrival clock to ``reference_view`` through one strobe.

    ``series`` maps view -> ``(t_ns, brightness)``; ``nominal_fps`` maps view ->
    requested rate. Postcondition: one :class:`ViewTiming` per view, in
    ``series`` order; ``status`` is ``available`` only when every view has an
    offset.
    """
    require(
        reference_view in series, "reference_view must be in series", reference_view
    )
    require(set(nominal_fps) >= set(series), "nominal_fps must cover every view")
    detection = _Detection(min_jump=min_jump, baseline_frames=baseline_frames)
    ref_t, ref_b = series[reference_view]
    _validate_series(ref_t, ref_b)
    reference = flash_event(
        ref_t, ref_b, min_jump=min_jump, baseline_frames=baseline_frames
    )
    views = tuple(
        _view_timing(view, t, b, nominal_fps[view], reference, detection)
        for view, (t, b) in series.items()
    )
    status: TimingStatus = (
        "available" if all(v.status == "available" for v in views) else "unavailable"
    )
    return TimingRecord(reference_view=reference_view, status=status, views=views)
