"""Plot data series for the grip force and couple plots (GCV-10, #11716).

Pure data, no GUI: both the PyQt widget and ``GET /api/analysis/grip-wrench``
(and so the web charts) read the same :class:`GripPlotSeries`.  Every wrench is
the loading exerted **by the hand on the club** in the world frame; the couple
is about the grip midpoint (see :mod:`grip_wrench`).  A sample that cannot be
computed is ``None`` in the JSON form (NaN in the arrays), never zero, and
``split_method`` records how the left/right split was obtained.

Traces (each ``{"x", "y", "z", "magnitude"}`` lists, unit in ``TRACE_UNITS``):

* ``left_force_n``, ``right_force_n``, ``net_force_n`` - per hand and net at the
  midpoint;
* ``couple_nm`` (world) and ``couple_local_nm`` (club-local, when a rotation was
  supplied to the analysis);
* ``contact_force_moment_nm`` (moments of force of the hand forces) and
  ``applied_free_torque_nm`` (sum of the hand free torques): the split of the
  couple;
* ``mof_left_nm``, ``mof_right_nm``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from typing import Any

import numpy as np

from src.shared.python.biomechanics.grip_wrench import GripAnalysis, GripSeries

__all__ = [
    "TRACE_LABELS",
    "TRACE_UNITS",
    "GripPlotSeries",
    "build_grip_plot_series",
    "plot_series_to_json",
]

TRACE_LABELS: dict[str, str] = {
    "left_force_n": "Left Hand Force",
    "right_force_n": "Right Hand Force",
    "net_force_n": "Net Force at Midpoint",
    "couple_nm": "Couple at Midpoint (World)",
    "couple_local_nm": "Couple at Midpoint (Club)",
    "contact_force_moment_nm": "Contact Force Moment",
    "applied_free_torque_nm": "Applied Free Torque",
    "mof_left_nm": "Left Moment of Force",
    "mof_right_nm": "Right Moment of Force",
}
TRACE_UNITS: dict[str, str] = {
    k: ("N" if k.endswith("_n") else "N*m") for k in TRACE_LABELS
}

#: GripSeries attribute behind every trace.
_ATTR: dict[str, str] = {
    "left_force_n": "left_force_n",
    "right_force_n": "right_force_n",
    "net_force_n": "net_force_n",
    "couple_nm": "couple_nm",
    "couple_local_nm": "couple_local_nm",
    "contact_force_moment_nm": "contact_force_moment_nm",
    "applied_free_torque_nm": "applied_free_torque_nm",
    "mof_left_nm": "mof_left_nm",
    "mof_right_nm": "mof_right_nm",
}


def _none_if_nan(values: np.ndarray) -> list[float | None]:
    return [None if not math.isfinite(float(v)) else float(v) for v in values]


def _trace(arr: np.ndarray) -> dict[str, list[float | None]]:
    mag = np.linalg.norm(arr, axis=1)  # NaN rows stay NaN
    return {
        "x": _none_if_nan(arr[:, 0]),
        "y": _none_if_nan(arr[:, 1]),
        "z": _none_if_nan(arr[:, 2]),
        "magnitude": _none_if_nan(mag),
    }


@dataclass(frozen=True)
class GripPlotSeries:
    """Time series ready to plot; ``None`` entries are unavailable samples."""

    time_s: tuple[float, ...]
    traces: Mapping[str, Mapping[str, list[float | None]]]
    split_method: str
    split_method_by_sample: tuple[str, ...]
    unavailable_reasons: tuple[str, ...]
    available: bool
    reason: str = ""
    events: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping (``None`` for unavailable samples, no NaN)."""
        return {
            "available": self.available,
            "reason": self.reason or None,
            "time_s": list(self.time_s),
            "split_method": self.split_method,
            "split_method_by_sample": list(self.split_method_by_sample),
            "unavailable_reasons": list(self.unavailable_reasons),
            "events": dict(self.events),
            "units": dict(TRACE_UNITS),
            "labels": dict(TRACE_LABELS),
            "traces": {k: {a: list(v) for a, v in t.items()} for k, t in self.traces.items()},
        }


def _summary_method(methods: Sequence[str]) -> str:
    seen = sorted(set(methods))
    return seen[0] if len(seen) == 1 else "mixed"


def build_grip_plot_series(
    time_s: Sequence[float],
    analyses: Sequence[GripAnalysis],
    *,
    events: Mapping[str, float] | None = None,
) -> GripPlotSeries:
    """Stack ``analyses`` into plot traces.

    Args:
        time_s: strictly one finite time per analysis.
        analyses: one :class:`GripAnalysis` per sample.
        events: named event times (for example ``{"impact": 0.9}``) drawn as
            markers.

    Postconditions: every trace has ``len(time_s)`` entries; a sample whose
    quantity is unavailable is ``None``; ``available`` is False only when no
    sample has a net force.

    Raises:
        ValueError: on empty input, a length mismatch or a non-finite
            time/event.
    """
    if len(time_s) == 0 or len(time_s) != len(analyses):
        raise ValueError("time_s and analyses must be non-empty and equal length")
    times = np.asarray(time_s, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("time_s must be finite")
    ev = {str(k): float(v) for k, v in (events or {}).items()}
    if not all(math.isfinite(v) for v in ev.values()):
        raise ValueError("event times must be finite")
    series = GripSeries.from_analyses(times, analyses)
    traces = {name: _trace(getattr(series, attr)) for name, attr in _ATTR.items()}
    any_net = any(v is not None for v in traces["net_force_n"]["x"])
    reasons = [a.unavailable_reason for a in analyses if a.unavailable_reason]
    return GripPlotSeries(
        time_s=tuple(float(t) for t in times),
        traces=traces,
        split_method=_summary_method(series.split_method),
        split_method_by_sample=series.split_method,
        unavailable_reasons=series.unavailable_reason,
        available=any_net,
        reason="" if any_net else (reasons[0] if reasons else "no grip data"),
        events=ev,
    )


def plot_series_to_json(series: GripPlotSeries) -> str:
    """Strict JSON text (``allow_nan=False``) of ``series``."""
    return json.dumps(series.to_dict(), allow_nan=False)
