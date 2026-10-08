"""Grip wrench plot service for simulation runs (GCV-10, #11716).

Resolves per-sample :class:`GripAnalysis` objects for a run and returns the
shared plot series.  Nothing is invented: a run with no recorded grip analyses
and no engine provider yields an *unavailable* payload with a reason.

Sources, in order:

* ``run.simulation_data["grip_analyses"]``: ``{"times_s": [...], "analyses":
  [GripAnalysis, ...]}`` recorded by the run;
* ``run.engine.get_grip_analyses()`` returning ``(times_s, analyses)``.

``run.simulation_data["impact_time_s"]`` (or the ``impact_time_s`` argument)
becomes the ``impact`` event marker.
"""

from __future__ import annotations

import math
from typing import Any

from src.shared.python.biomechanics.grip_plot_model import (
    TRACE_LABELS,
    TRACE_UNITS,
    build_grip_plot_series,
)
from src.shared.python.biomechanics.grip_wrench import GripAnalysis

__all__ = ["NO_GRIP_REASON", "compute_grip_plot", "resolve_grip_analyses"]

NO_GRIP_REASON = (
    "run carries no grip wrench: record 'grip_analyses' in the run data or use an "
    "engine that provides get_grip_analyses() (MuJoCo, Drake and MyoSuite full-body "
    "models; OpenSim reports it unavailable)"
)


def _checked(times: Any, analyses: Any) -> tuple[list[float], list[GripAnalysis]]:
    analyses = list(analyses)
    times = [float(t) for t in times]
    if not analyses or len(times) != len(analyses):
        raise ValueError("grip_analyses needs equal, non-empty times_s and analyses")
    if not all(isinstance(a, GripAnalysis) for a in analyses):
        raise TypeError("grip analyses must be GripAnalysis instances")
    return times, analyses


def resolve_grip_analyses(run: Any) -> tuple[list[float], list[GripAnalysis]] | None:
    """The run's grip analyses as ``(times_s, analyses)``, or ``None``."""
    recorded = run.simulation_data.get("grip_analyses")
    if isinstance(recorded, dict):
        return _checked(recorded.get("times_s", ()), recorded.get("analyses", ()))
    provider = getattr(run.engine, "get_grip_analyses", None)
    if callable(provider):
        result = provider()
        if result is not None:
            return _checked(*result)
    return None


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "time_s": [],
        "split_method": "unavailable",
        "split_method_by_sample": [],
        "unavailable_reasons": [],
        "events": {},
        "units": dict(TRACE_UNITS),
        "labels": dict(TRACE_LABELS),
        "traces": {},
    }


def compute_grip_plot(
    run: Any, *, impact_time_s: float | None = None
) -> dict[str, Any]:
    """Plot payload for ``run`` (see ``GripPlotSeries.to_dict``).

    Raises:
        ValueError: for malformed grip data or a non-finite ``impact_time_s``
            (HTTP 400); data absence becomes an unavailable payload.
    """
    if impact_time_s is not None and not math.isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite")
    resolved = resolve_grip_analyses(run)
    if resolved is None:
        return _unavailable(NO_GRIP_REASON)
    times, analyses = resolved
    impact = impact_time_s
    if impact is None:
        recorded = run.simulation_data.get("impact_time_s")
        impact = float(recorded) if recorded is not None else None
    events = {} if impact is None else {"impact": impact}
    return build_grip_plot_series(times, analyses, events=events).to_dict()
