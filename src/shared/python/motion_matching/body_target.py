"""Canonical ``BodyTarget`` dataclass and validation.

Source-agnostic, frozen, validated-on-construction container for full-body
marker trajectories on a uniform simulation timegrid. Mirrors
:class:`ClubTarget` in spirit; the two are designed to share an
:class:`AlignOptions` clock so a body and club target can be combined into a
multi-source motion-matching cost without per-call resampling.

Public API:
    BodyTarget                  -- frozen dataclass for measured body markers.
    BodyEvent                   -- companion dataclass for named events.
    MAX_BODY_POSITION_NORM_M    -- sanity bound for human-scale motion (m).
    BODY_TARGET_SCHEMA_VERSION  -- integer schema version for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .club_target import TIME_EPS, SourceProvenance

MAX_BODY_POSITION_NORM_M: float = 3.0
"""Maximum |xyz| for any finite marker sample (metres)."""

BODY_TARGET_SCHEMA_VERSION: int = 1
"""Integer schema version for forward-compatible serialization."""


@dataclass(frozen=True)
class BodyEvent:
    """A labeled event located on the body-target timegrid.

    Attributes:
        label:   Free-form non-empty event label (e.g. ``"impact"``).
        frame:   Integer frame index on the resampled grid (``0 <= frame < N``).
        time_s:  Wall-clock time of the event on the timegrid, seconds.
    """

    label: str
    frame: int
    time_s: float

    def __post_init__(self) -> None:
        """Validate event fields."""
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("BodyEvent.label must be a non-empty string")
        if not isinstance(self.frame, int) or isinstance(self.frame, bool):
            raise TypeError("BodyEvent.frame must be int")


@dataclass(frozen=True)
class BodyTarget:
    """Canonical full-body marker trajectory on a uniform timegrid.

    Validated at construction; any violation of the validation rules raises
    :class:`ValueError` (or :class:`TypeError` for the source-type rule).
    Frozen so loaders are forced to produce a fully-formed, validated artifact
    rather than mutating one in place.

    Attributes:
        time:             ``(N,)`` seconds, strictly increasing, ``time[0] == 0``.
        marker_xyz:       ``(N, M, 3)`` metres; NaN allowed for occluded samples.
        marker_names:     length-``M`` tuple of unique non-empty marker names.
        impact_idx:       Frame index of impact on the resampled grid (``0..N-1``).
        events:           Tuple of :class:`BodyEvent` annotations (may be empty).
        source:           :class:`SourceProvenance` describing the file of origin.
        coordinate_frame: Always ``"z_up_right_handed"`` for now.
    """

    time: np.ndarray
    marker_xyz: np.ndarray
    marker_names: tuple[str, ...]
    impact_idx: int
    events: tuple[BodyEvent, ...]
    source: SourceProvenance
    coordinate_frame: Literal["z_up_right_handed"] = "z_up_right_handed"
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Run all validation rules at construction."""
        _validate_body_target(self)


def _validate_time(time: np.ndarray) -> int:
    """Check the time vector and return its length ``N``."""
    if not isinstance(time, np.ndarray) or time.ndim != 1:
        raise ValueError(
            f"time must be a 1-D ndarray, got shape {getattr(time, 'shape', None)}"
        )
    n = int(time.shape[0])
    if n < 2:
        raise ValueError(f"time must have at least 2 samples (got {n})")
    if abs(float(time[0])) > TIME_EPS:
        raise ValueError(f"time[0] must be 0, got {time[0]!r}")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    return n


def _validate_marker_names(names: tuple[str, ...]) -> int:
    """Check marker_names tuple and return ``M``."""
    if not isinstance(names, tuple):
        raise ValueError("marker_names must be a tuple")
    m = len(names)
    if m < 3:
        raise ValueError(f"marker_names must have at least 3 entries (got {m})")
    if any((not isinstance(s, str)) or not s for s in names):
        raise ValueError("marker_names entries must be non-empty strings")
    if len(set(names)) != m:
        raise ValueError("marker_names must be unique")
    return m


def _validate_marker_xyz(arr: np.ndarray, n: int, m: int) -> None:
    """Shape, magnitude, and finite-frame coverage checks for marker_xyz."""
    if not isinstance(arr, np.ndarray) or arr.shape != (n, m, 3):
        raise ValueError(
            f"marker_xyz must have shape ({n}, {m}, 3), got {getattr(arr, 'shape', None)}"
        )
    finite = np.isfinite(arr)
    if finite.any():
        # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~1.5x faster than np.linalg.norm(..., axis=-1)
        clean_arr = np.where(finite.all(axis=-1, keepdims=True), arr, 0.0)
        norms = np.sqrt(np.einsum("...i,...i->...", clean_arr, clean_arr))
        sample_finite = finite.all(axis=-1)
        if np.any((norms >= MAX_BODY_POSITION_NORM_M) & sample_finite):
            max_n = float(norms[sample_finite].max()) if sample_finite.any() else 0.0
            raise ValueError(
                f"marker_xyz has |r| >= {MAX_BODY_POSITION_NORM_M} m (max {max_n:.3f})"
            )
    # At least one frame must have >= 50% of markers fully finite.
    per_frame_marker_finite = finite.all(axis=-1)  # (N, M)
    fraction = per_frame_marker_finite.mean(axis=-1)
    if not np.any(fraction >= 0.5):
        raise ValueError(
            "marker_xyz must contain at least one frame with >=50% finite markers"
        )


def _validate_events(events: tuple[BodyEvent, ...], n: int) -> None:
    """Check the events tuple."""
    if not isinstance(events, tuple):
        raise ValueError("events must be a tuple of BodyEvent")
    labels: list[str] = []
    for ev in events:
        if not isinstance(ev, BodyEvent):
            raise TypeError("events entries must be BodyEvent instances")
        if not (0 <= ev.frame < n):
            raise ValueError(f"event frame {ev.frame} out of range [0, {n})")
        labels.append(ev.label)
    if len(set(labels)) != len(labels):
        raise ValueError("event labels must be unique")


def _validate_body_target(t: BodyTarget) -> None:
    """Enforce the BodyTarget validation rules."""
    n = _validate_time(t.time)
    m = _validate_marker_names(t.marker_names)
    _validate_marker_xyz(t.marker_xyz, n, m)
    if not (0 <= int(t.impact_idx) < n):
        raise ValueError(f"impact_idx must be in [0, {n}), got {t.impact_idx}")
    _validate_events(t.events, n)
    if not isinstance(t.source, SourceProvenance):
        raise TypeError("source must be a SourceProvenance instance")
    if t.coordinate_frame != "z_up_right_handed":
        raise ValueError(
            f"coordinate_frame must be 'z_up_right_handed', got {t.coordinate_frame!r}"
        )
