"""Impact-parameters service for runs and matched swings (GCV-17, #11723).

Resolves a world-frame :class:`ClubheadSeries` for a simulation run, extracts
the target-relative impact parameters with the shared GCV-15 core and returns
the shared display card.  Nothing is invented: a run with no clubhead series,
or one below the minimum speed, yields an *unavailable* card with a reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    ImpactCard,
    build_impact_card,
    parse_target_dir,
)

__all__ = ["compute_impact_card", "resolve_clubhead_series", "unavailable_card"]

NO_SERIES_REASON = (
    "run carries no clubhead series: record 'clubhead_series' in the run data or use "
    "an engine that provides get_clubhead_series()"
)
_MUJOCO_CLUB_BODY = "clubhead"


def unavailable_card(reason: str, units: str = "mph") -> ImpactCard:
    """Card with no rows and ``available=False``; never zero-filled."""
    return ImpactCard(
        units=units,
        impact_time_s=float("nan"),
        impact_time_source="unavailable",
        frame={},
        rows=(),
        d_plane={},
        available=False,
        reason=reason,
    )


def _series_from_mapping(data: dict[str, Any]) -> ClubheadSeries:
    keys = ("times_s", "face_center_m", "velocity_mps")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"clubhead_series is missing {missing}")
    return ClubheadSeries(
        **{k: data[k] for k in keys},
        face_normal=data.get("face_normal"),
        toe_axis=data.get("toe_axis"),
        grip_axis=data.get("grip_axis"),
        face_unobservable_reason=data.get("face_unobservable_reason"),
    )


def _series_from_mujoco_run(run: Any) -> ClubheadSeries | None:
    model = getattr(run.engine, "model", None)
    data = run.simulation_data
    if model is None or not all(
        k in data for k in ("times", "joint_positions", "joint_velocities")
    ):
        return None
    try:
        import mujoco
    except ImportError:
        return None
    if not isinstance(model, mujoco.MjModel):
        return None
    q = np.asarray(data["joint_positions"], dtype=float)
    v = np.asarray(data["joint_velocities"], dtype=float)
    if q.ndim != 2 or q.shape[1] != model.nq or v.shape[1:] != (model.nv,):
        return None
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _MUJOCO_CLUB_BODY) < 0:
        return None
    from src.shared.python.impact_parameters.adapters import (
        clubhead_series_from_mujoco,
    )

    return clubhead_series_from_mujoco(model, data["times"], q, v, _MUJOCO_CLUB_BODY)


def resolve_clubhead_series(run: Any) -> ClubheadSeries | None:
    """Return the run's ClubheadSeries, or ``None`` when it has none."""
    recorded = run.simulation_data.get("clubhead_series")
    if isinstance(recorded, ClubheadSeries):
        return recorded
    if isinstance(recorded, dict):
        return _series_from_mapping(recorded)
    provider = getattr(run.engine, "get_clubhead_series", None)
    if callable(provider):
        series = provider()
        if series is not None and not isinstance(series, ClubheadSeries):
            raise TypeError("get_clubhead_series() must return a ClubheadSeries")
        return series
    return _series_from_mujoco_run(run)


def compute_impact_card(
    run: Any,
    *,
    target_dir: str | None = None,
    handedness: str = "right",
    units: str = "mph",
    impact_index: int | None = None,
) -> ImpactCard:
    """Impact card for ``run`` relative to ``target_dir``.

    Raises ``ValueError`` for malformed ``target_dir``/``handedness``/``units``
    (HTTP 400).  Data problems become an unavailable card.
    """
    frame = TargetFrame(target_dir=parse_target_dir(target_dir), handedness=handedness)
    series = resolve_clubhead_series(run)
    if series is None:
        return unavailable_card(NO_SERIES_REASON, units)
    try:
        params = extract_impact_parameters(series, frame, impact_index=impact_index)
    except ValueError as exc:
        if impact_index is not None and "out of range" in str(exc):
            raise
        return unavailable_card(str(exc), units)
    return build_impact_card(params, units=units)
