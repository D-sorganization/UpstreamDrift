"""Ground-reaction plot service for simulation runs (GCV-5, #11711).

Resolves the GCV-1 :class:`GroundReactionSeries` of a run and returns the
shared plot series.  Nothing is invented: a run with no recorded series and no
engine provider yields an *unavailable* payload with a reason.

Sources, in order:

* ``run.simulation_data["ground_reaction_series"]``: the series recorded by
  the run;
* ``run.engine.get_ground_reaction_series()`` returning the series or ``None``;
* ``run.recorder.get_ground_reaction_series()`` returning the series or
  ``None`` (the ``GenericPhysicsRecorder`` stacked breakdown, GCV-5, #11711).

Optional run data: ``body_weight_n`` (or ``body_mass_kg``) for the
body-weight traces, ``events`` (name to time) and ``impact_time_s`` for the
event markers.  The keyword arguments override the run data.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from src.shared.python.biomechanics.ground_reaction import GroundReactionSeries
from src.shared.python.biomechanics.ground_reaction_plot_model import (
    build_ground_reaction_plot_series,
    unavailable_ground_reaction_plot,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2

__all__ = [
    "NO_GROUND_REACTION_REASON",
    "compute_ground_reaction_plot",
    "resolve_ground_reaction_series",
]

NO_GROUND_REACTION_REASON = (
    "run carries no ground reaction: record 'ground_reaction_series' in the run "
    "data, use an engine that provides get_ground_reaction_series(), or record "
    "with a recorder that does (Simscape reports it unavailable until GCV-3)"
)


def _checked(series: Any) -> GroundReactionSeries:
    if not isinstance(series, GroundReactionSeries):
        raise TypeError(
            f"ground reaction must be a GroundReactionSeries, got {type(series)}"
        )
    return series


def resolve_ground_reaction_series(run: Any) -> GroundReactionSeries | None:
    """The run's ground-reaction series, or ``None`` when it has none.

    Raises:
        TypeError: if the recorded or provided value is not a series.
    """
    recorded = run.simulation_data.get("ground_reaction_series")
    if recorded is not None:
        return _checked(recorded)
    provider = getattr(run.engine, "get_ground_reaction_series", None)
    if callable(provider):
        result = provider()
        if result is not None:
            return _checked(result)
    recorder_provider = getattr(
        getattr(run, "recorder", None), "get_ground_reaction_series", None
    )
    if callable(recorder_provider):
        result = recorder_provider()
        if result is not None:
            return _checked(result)
    return None


def _body_weight_n(data: Mapping[str, Any], override: float | None) -> float | None:
    if override is not None:
        return float(override)
    if data.get("body_weight_n") is not None:
        return float(data["body_weight_n"])
    if data.get("body_mass_kg") is not None:
        return float(data["body_mass_kg"]) * GRAVITY_M_S2
    return None


def _events(data: Mapping[str, Any], impact_time_s: float | None) -> dict[str, float]:
    events = {str(k): float(v) for k, v in dict(data.get("events") or {}).items()}
    impact = impact_time_s if impact_time_s is not None else data.get("impact_time_s")
    if impact is not None:
        events["impact"] = float(impact)
    return events


def compute_ground_reaction_plot(
    run: Any,
    *,
    body_weight_n: float | None = None,
    impact_time_s: float | None = None,
) -> dict[str, Any]:
    """Plot payload for ``run`` (see ``GroundReactionPlotSeries.to_dict``).

    Raises:
        ValueError: for a non-finite ``impact_time_s`` or invalid body weight or
            event data (HTTP 400); data absence becomes an unavailable payload.
        TypeError: if the run's ground-reaction data is not a series.
    """
    if impact_time_s is not None and not math.isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite")
    series = resolve_ground_reaction_series(run)
    if series is None:
        return unavailable_ground_reaction_plot(NO_GROUND_REACTION_REASON).to_dict()
    data = run.simulation_data
    return build_ground_reaction_plot_series(
        series,
        body_weight_n=_body_weight_n(data, body_weight_n),
        events=_events(data, impact_time_s),
    ).to_dict()
