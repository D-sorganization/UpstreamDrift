"""Club state at the instant the face reaches the ball (#15150).

A sampled swing meets its ball between two samples. The sample before can lie
a whole step short of it (0.14 m and 7 degrees of face closing on a 360 Hz
driver swing), and every sample after carries the strike: a head matched to a
captured swing slows by a third there. :func:`series_at_contact` fits the
samples that end before contact and carries the fit to the instant its face
plane touches the ball. Numbers come from ``impact_state_settings.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import polynomial

from src.shared.python.impact_parameters import ClubheadSeries

SETTINGS_PATH = Path(__file__).with_name("impact_state_settings.json")
_KEYS = ("stencil_samples", "fit_order", "search_window_s", "max_extrapolation_steps")
#: Halvings of the bracket of the contact instant (2**-60 of a sample step).
_BISECTIONS = 60


@dataclass(frozen=True)
class ImpactStateSettings:
    """Validated ``impact_state_settings.json`` (``ValueError`` when invalid)."""

    stencil_samples: int
    fit_order: int
    search_window_s: float
    max_extrapolation_steps: float

    def __post_init__(self) -> None:
        counts = (self.stencil_samples, self.fit_order)
        if not all(type(value) is int for value in counts):
            raise ValueError("stencil_samples and fit_order must be integers")
        if not 1 <= self.fit_order < self.stencil_samples:
            raise ValueError("fit_order must be in [1, stencil_samples)")
        for name in ("search_window_s", "max_extrapolation_steps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def load_impact_state_settings(path: Path | None = None) -> ImpactStateSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return ImpactStateSettings(*(raw[key] for key in _KEYS))


def _unit(vector: Any) -> Any:
    return vector / np.linalg.norm(vector)


def _last_before_contact(
    series: ClubheadSeries,
    ball: Any,
    radius: float,
    impact_time_s: float,
    window: float,
) -> int | None:
    """Last sample in the window whose face plane has not reached the ball."""
    normal = np.asarray(series.face_normal, dtype=float)
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    gap = np.einsum("ij,ij->i", ball - series.face_center_m, normal) - radius
    near = np.abs(series.times_s - impact_time_s) <= window
    reached = np.nonzero(near[:-1] & (gap[:-1] > 0.0) & (gap[1:] <= 0.0))[0]
    return int(reached[0]) if reached.size else None


def _contact_offset_s(gap: Any, limit_s: float) -> float | None:
    """Root of the decreasing ``gap`` in ``[0, limit_s]``, ``None`` without one."""
    if gap(0.0) <= 0.0:
        return 0.0
    if gap(limit_s) > 0.0:
        return None
    low, high = 0.0, limit_s
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if gap(middle) > 0.0 else (low, middle)
    return high


def series_at_contact(
    series: ClubheadSeries,
    ball_centre_m: object,
    radius_m: float,
    impact_time_s: float,
    settings: ImpactStateSettings | None = None,
) -> tuple[ClubheadSeries, int] | None:
    """``series`` with the state at ball contact added, and that sample's index.

    The state is a polynomial fit in time of the last ``stencil_samples``
    samples whose face plane has not reached the ball, evaluated where its
    face plane touches the ball surface; the velocity is the derivative of the
    fitted face centre. ``None`` without face normals, when the face does not
    reach the ball within ``search_window_s`` of ``impact_time_s``, with fewer
    than 2 samples before contact, or when the fit does not reach the ball
    within ``max_extrapolation_steps``.

    Preconditions: a ``ClubheadSeries`` (``TypeError``); a finite 3-vector
    ``ball_centre_m``, positive finite ``radius_m``, finite ``impact_time_s``
    (``ValueError``). Postconditions: the samples before contact are
    unchanged and the added sample's face plane is one ball radius from the
    ball centre (to the fit residual when the fit starts past the ball).
    """
    if not isinstance(series, ClubheadSeries):
        raise TypeError("series must be a ClubheadSeries")
    ball = np.asarray(ball_centre_m, dtype=float)
    if ball.shape != (3,) or not np.isfinite(ball).all():
        raise ValueError("ball_centre_m must be a finite 3-vector")
    radius = float(radius_m)
    if not (math.isfinite(radius) and radius > 0.0 and math.isfinite(impact_time_s)):
        raise ValueError("radius_m must be positive and impact_time_s finite")
    chosen = settings or load_impact_state_settings()
    if series.face_normal is None:
        return None
    window = chosen.search_window_s
    last = _last_before_contact(series, ball, radius, impact_time_s, window)
    first = max((last or 0) - chosen.stencil_samples + 1, 0)
    if last is None or last == first:
        return None
    times = series.times_s
    since = times[first : last + 1] - times[last]
    order = min(chosen.fit_order, last - first)

    def fitted(values: Any) -> Any:
        return polynomial.polyfit(since, np.asarray(values)[first : last + 1], order)

    centre, normal = fitted(series.face_center_m), fitted(series.face_normal)

    def gap(offset_s: float) -> float:
        unit = _unit(polynomial.polyval(offset_s, normal))
        return float((ball - polynomial.polyval(offset_s, centre)) @ unit) - radius

    limit = chosen.max_extrapolation_steps * float(times[last + 1] - times[last])
    offset = _contact_offset_s(gap, limit)
    if offset is None:
        return None
    when = float(times[last]) + offset
    before, after = times < when, times > when
    at = int(np.count_nonzero(before))

    def spliced(values: Any, state: Any) -> Any:
        return np.concatenate([values[before], [state], values[after]])

    def axis(values: Any) -> Any:
        if values is None:
            return None
        return spliced(values, _unit(polynomial.polyval(offset, fitted(values))))

    velocity = polynomial.polyval(offset, polynomial.polyder(centre))
    moved = replace(
        series,
        times_s=spliced(times, when),
        face_center_m=spliced(series.face_center_m, polynomial.polyval(offset, centre)),
        velocity_mps=spliced(series.velocity_mps, velocity),
        face_normal=axis(series.face_normal),
        toe_axis=axis(series.toe_axis),
        grip_axis=axis(series.grip_axis),
    )
    return moved, at
