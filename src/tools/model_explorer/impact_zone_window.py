"""Derive the impact-zone playback window from a contact summary (#15524).

Qt-free adapter between ``export_impact_run_playback_summary`` (its
``contact_window_s``) and :class:`ImpactZoneClock` (``impact_s`` and
``half_width_s``). It fails closed: a run whose contact was not qualified, or
whose window is malformed or outside the run, yields ``None`` so the panel
falls back to plain playback instead of slowing the wrong part of the swing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.physics.impact_interval_handoff import UD_IMPACT_PLAYBACK_SCHEMA
from src.tools.model_explorer.impact_zone_clock import ImpactZoneClock

__all__ = [
    "DEFAULT_MIN_HALF_WIDTH_S",
    "DEFAULT_PAD_S",
    "ImpactZoneWindow",
    "build_impact_zone_clock",
    "impact_zone_window",
]

#: Lead-in and tail kept slow on each side of the contact interval, seconds.
DEFAULT_PAD_S = 0.01
#: Contacts are ~0.5 ms; never let the slow zone shrink below this half-width.
DEFAULT_MIN_HALF_WIDTH_S = 0.005


@dataclass(frozen=True)
class ImpactZoneWindow:
    """Centre and half-width of the slow-motion zone, in simulation seconds."""

    impact_s: float
    half_width_s: float


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _contact_bounds(summary: Mapping[str, Any]) -> tuple[float, float] | None:
    window = summary.get("contact_window_s")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        return None
    if not all(_is_real(v) for v in window):
        return None
    start, end = float(window[0]), float(window[1])
    return (start, end) if end > start else None


def impact_zone_window(
    summary: Mapping[str, Any],
    time_s: NDArray[np.float64],
    *,
    pad_s: float = DEFAULT_PAD_S,
    min_half_width_s: float = DEFAULT_MIN_HALF_WIDTH_S,
) -> ImpactZoneWindow | None:
    """Return the slow zone for a qualified contact summary, else ``None``.

    Preconditions: ``pad_s >= 0`` and ``min_half_width_s > 0`` (else
    ``ValueError``). Postcondition: a returned window is centred inside the
    run's time span and its half-width is at least ``min_half_width_s``.
    """
    if not (_is_real(pad_s) and pad_s >= 0):
        raise ValueError("pad_s must be finite and non-negative")
    if not (_is_real(min_half_width_s) and min_half_width_s > 0):
        raise ValueError("min_half_width_s must be finite and positive")
    if not isinstance(summary, Mapping):
        return None
    if summary.get("format") != UD_IMPACT_PLAYBACK_SCHEMA:
        return None
    if summary.get("qualified") is not True:
        return None
    bounds = _contact_bounds(summary)
    if bounds is None:
        return None
    centre = 0.5 * (bounds[0] + bounds[1])
    times = np.asarray(time_s, dtype=float)
    if times.size == 0 or not (times[0] <= centre <= times[-1]):
        return None
    half = max(0.5 * (bounds[1] - bounds[0]) + pad_s, min_half_width_s)
    return ImpactZoneWindow(impact_s=centre, half_width_s=half)


def build_impact_zone_clock(
    summary: Mapping[str, Any],
    time_s: NDArray[np.float64],
    slow_factor: float,
    speed: float = 1.0,
    *,
    pad_s: float = DEFAULT_PAD_S,
) -> ImpactZoneClock | None:
    """Build an :class:`ImpactZoneClock` for ``summary``, or ``None``.

    An unqualified or unusable summary returns ``None`` without constructing
    a clock. Otherwise ``slow_factor`` and ``speed`` are validated by the
    clock itself and a violation raises ``ValueError``.
    """
    window = impact_zone_window(summary, time_s, pad_s=pad_s)
    if window is None:
        return None
    return ImpactZoneClock(
        time_s, window.impact_s, window.half_width_s, slow_factor, speed
    )
