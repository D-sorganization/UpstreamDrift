"""Measured plate overlay for the ground-reaction plots (OF-07, #13087).

Reads a placed vertical plate trial at the model's own sample times and shapes
it as plot traces that sit beside the model's computed ground reaction:

* ``measured_<foot>_force_n`` - ``x`` and ``y`` are ``None`` (a vertical plate
  does not observe shear; unobserved is never zero), ``z`` is the load;
* ``measured_<foot>_cop_m`` - the centre of pressure on the force-sensing
  plane, ``None`` where the foot is unloaded.

A model frame the trial does not cover is ``None`` in every trace, never
extrapolated.  ``qualified`` carries the trial's provenance: an overlay from an
unqualified device must be shown as such.  Force events (OF-04) are published
on the model clock under the ``force.`` namespace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.contracts import FOOT_LABELS
from src.shared.python.force_plates.conversion import observability
from src.shared.python.force_plates.events import (
    force_event_markers,
    merge_event_markers,
)
from src.shared.python.force_plates.readings import (
    FootSample,
    PlacedTrial,
    ReadingOptions,
    reading_at,
)

__all__ = [
    "MEASURED_PREFIX",
    "MeasuredOverlay",
    "attach_measured_overlay",
    "measured_overlay",
    "overlay_on_series",
]

MEASURED_PREFIX = "measured_"
#: Matches the sample-snap tolerance of ``readings.reading_at``.
_SNAP_ULPS = 4
_Trace = dict[str, list[float | None]]


@dataclass(frozen=True)
class MeasuredOverlay:
    """Measured traces on the model's sample times.

    ``traces`` maps ``measured_<foot>_{force_n,cop_m}`` to
    ``{"x", "y", "z", "magnitude"}`` lists; ``covered`` is True for each model
    frame the trial spans; ``events`` are ``force.<name>`` times on the model
    clock; ``observability`` says which components a vertical plate observes.
    """

    traces: Mapping[str, Mapping[str, list[float | None]]]
    covered: tuple[bool, ...]
    qualified: bool
    observability: Mapping[str, bool]
    events: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", MappingProxyType(dict(self.traces)))
        object.__setattr__(
            self, "observability", MappingProxyType(dict(self.observability))
        )
        object.__setattr__(self, "events", MappingProxyType(dict(self.events)))


def _empty_trace() -> _Trace:
    return {axis: [] for axis in ("x", "y", "z", "magnitude")}


def _append(trace: _Trace, x: float | None, y: float | None, z: float | None) -> None:
    """Add one sample; the magnitude of a partly unobserved vector is unknown."""
    trace["x"].append(x)
    trace["y"].append(y)
    trace["z"].append(z)
    trace["magnitude"].append(None)


def _covers(
    placed: PlacedTrial,
    model_time_s: float,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> bool:
    start, stop = placed.span_s()
    t = alignment.trial_time(model_time_s, options.offset_shift_s)
    scale = max(
        abs(model_time_s), abs(alignment.offset_s), abs(options.offset_shift_s), abs(t)
    )
    tol = _SNAP_ULPS * math.ulp(scale)
    return start - tol <= t <= stop + tol


def _append_foot(force: _Trace, cop: _Trace, sample: FootSample | None) -> None:
    if sample is None:
        _append(force, None, None, None)
        _append(cop, None, None, None)
        return
    _append(force, None, None, sample.load_n)
    point = sample.cop_m
    if point is None:
        _append(cop, None, None, None)
    else:
        _append(cop, float(point[0]), float(point[1]), float(point[2]))


def _check_times(times_s: Sequence[float]) -> list[float]:
    times = [float(t) for t in times_s]
    if not all(math.isfinite(t) for t in times):
        raise ValueError("times_s must be finite")
    return times


def measured_overlay(
    placed: PlacedTrial,
    alignment: PlateTimeAlignment,
    times_s: Sequence[float],
    options: ReadingOptions,
) -> MeasuredOverlay:
    """Measured traces of ``placed`` at the model times ``times_s``.

    Frames outside the trial span are ``None``, never extrapolated.  Raises
    ``TypeError`` / ``ValueError`` for wrong types or non-finite times, and
    whatever :func:`reading_at` raises (e.g. a curvature that cannot be
    estimated without a declared floor).  Postcondition: every trace has one
    entry per model time.
    """
    times = _check_times(times_s)
    force = {foot: _empty_trace() for foot in FOOT_LABELS}
    cop = {foot: _empty_trace() for foot in FOOT_LABELS}
    covered: list[bool] = []
    for when in times:
        inside = _covers(placed, when, alignment, options)
        covered.append(inside)
        reading = reading_at(placed, when, alignment, options) if inside else None
        for foot in FOOT_LABELS:
            sample = None if reading is None else reading.feet[foot]
            _append_foot(force[foot], cop[foot], sample)
    traces: dict[str, Mapping[str, list[float | None]]] = {}
    for foot in FOOT_LABELS:
        traces[f"{MEASURED_PREFIX}{foot}_force_n"] = force[foot]
        traces[f"{MEASURED_PREFIX}{foot}_cop_m"] = cop[foot]
    trial = placed.trial
    return MeasuredOverlay(
        traces=traces,
        covered=tuple(covered),
        qualified=trial.provenance.qualified,
        observability=observability(),
        events=force_event_markers(trial, alignment=alignment),
    )


def attach_measured_overlay(
    series: GroundReactionPlotSeries, overlay: MeasuredOverlay
) -> GroundReactionPlotSeries:
    """A copy of ``series`` carrying ``overlay`` (traces, events, provenance).

    Raises ``TypeError`` for wrong types and ``ValueError`` when the overlay was
    not computed on ``series.time_s`` (length), or the series already carries
    a measured overlay.
    """
    if not isinstance(series, GroundReactionPlotSeries):
        raise TypeError("series must be a GroundReactionPlotSeries")
    if not isinstance(overlay, MeasuredOverlay):
        raise TypeError("overlay must be a MeasuredOverlay")
    if len(overlay.covered) != len(series.time_s):
        raise ValueError("overlay length differs from the series time axis")
    if series.measured_qualified is not None:
        raise ValueError("series already carries a measured overlay")
    events = merge_event_markers(series.events, overlay.events)
    return replace(
        series,
        traces={**series.traces, **overlay.traces},
        events=events,
        measured_qualified=overlay.qualified,
        observability=dict(overlay.observability),
    )


def overlay_on_series(
    series: GroundReactionPlotSeries,
    placed: PlacedTrial,
    alignment: PlateTimeAlignment,
    options: ReadingOptions,
) -> GroundReactionPlotSeries:
    """``series`` with the measured overlay of ``placed`` on its model times."""
    overlay = measured_overlay(placed, alignment, series.time_s, options)
    return attach_measured_overlay(series, overlay)
