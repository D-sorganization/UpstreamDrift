"""GRF and COP observability, event-anchored peaks, and normalisation contracts.

GM-2 / Issue #14425 (Epic #14421).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.biomechanics.metric_definitions.schema import (
    NormalisationKind,
    SourceKind,
)
from src.shared.python.contracts import ensure, require

Array: TypeAlias = NDArray[np.float64]

STANDARD_GRAVITY_M_S2: float = 9.80665
SUPPORTED_MODELLED_ENGINES: tuple[str, ...] = ("mujoco", "drake", "opensim")
LEAD_CROSSING_FRACTION: float = 0.5
COP_TARGET_DIRECTION_DEFAULT: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class GRFChannelObservability:
    """Declared observability of GRF and COP channels."""

    vertical_force_observed: bool = True
    cop_observed: bool = True
    shear_observed: bool = False
    free_moment_observed: bool = False


@dataclass(frozen=True)
class GRFSourceAttribution:
    """Source origin of ground-reaction series."""

    source: SourceKind
    engine: str | None = None

    def __post_init__(self) -> None:
        src = (
            self.source
            if isinstance(self.source, SourceKind)
            else SourceKind(self.source)
        )
        if src == SourceKind.modelled:
            require(
                bool(self.engine and self.engine.strip()),
                "Modelled source engine name required",
            )
            eng = (self.engine or "").strip().lower()
            require(
                eng in SUPPORTED_MODELLED_ENGINES,
                f"engine must be one of {SUPPORTED_MODELLED_ENGINES}",
            )
        else:
            require(self.engine is None, "Measured source must not carry engine")


def validate_unobserved_channel(value: float | None, *, observed: bool) -> bool:
    """Enforce contract that unobserved channels must remain NaN / sentinel."""
    if not observed:
        is_nan = value is None or (
            isinstance(value, (float, np.floating)) and math.isnan(value)
        )
        require(is_nan, "Unobserved channel must remain NaN")
    return True


def sanitize_channel_value(value: float, *, observed: bool) -> float:
    """Return float('nan') when unobserved to guarantee sentinel integrity."""
    return float("nan") if not observed else float(value)


def format_channel_value(value: float, *, observed: bool, unit: str = "N") -> str:
    """Render channel value; unobserved channels are rendered as 'NaN', never '0'."""
    return "NaN" if not observed or math.isnan(value) else f"{value:.2f} {unit}"


def normalize_vertical_force(
    fz_n: ArrayLike,
    *,
    body_mass_kg: float,
    normalisation: NormalisationKind = NormalisationKind.percent_body_weight,
    gravity: float = STANDARD_GRAVITY_M_S2,
) -> Array:
    """Normalize vertical ground reaction force under registry normalisation contract."""
    require(body_mass_kg > 0.0, "body_mass_kg must be positive")
    require(gravity > 0.0, "gravity must be positive")
    arr: Array = np.asarray(fz_n, dtype=np.float64)
    res: Array = (
        arr * (100.0 / (body_mass_kg * gravity))
        if normalisation == NormalisationKind.percent_body_weight
        else arr.copy()
    )
    ensure(res.shape == arr.shape, "Shape mismatch")
    return res


@dataclass(frozen=True)
class EventAnchoredPeak:
    """Peak vertical force and event-anchored timings."""

    peak_value: float
    peak_time_s: float
    dt_top_s: float
    dt_impact_s: float
    normalisation: NormalisationKind


@dataclass(frozen=True)
class GRFPeakSummary:
    """Lead and trail foot peak vertical forces."""

    lead: EventAnchoredPeak
    trail: EventAnchoredPeak


def _single_foot_peak(
    t: Array, f: Array, top: float, imp: float, norm: NormalisationKind
) -> EventAnchoredPeak:
    valid = np.flatnonzero(np.isfinite(f))
    if valid.size == 0:
        return EventAnchoredPeak(
            float("nan"), float("nan"), float("nan"), float("nan"), norm
        )
    idx = int(valid[int(np.argmax(f[valid]))])
    tp = float(t[idx])
    return EventAnchoredPeak(
        float(f[idx]),
        tp,
        tp - top if math.isfinite(top) else float("nan"),
        tp - imp if math.isfinite(imp) else float("nan"),
        norm,
    )


def compute_event_anchored_peaks(
    times_s: ArrayLike,
    fz_lead_n: ArrayLike,
    fz_trail_n: ArrayLike,
    *,
    t_top_s: float,
    t_impact_s: float,
    body_mass_kg: float | None = None,
    normalisation: NormalisationKind = NormalisationKind.percent_body_weight,
    gravity: float = STANDARD_GRAVITY_M_S2,
) -> GRFPeakSummary:
    """Compute peak vertical force for lead and trail feet relative to swing events."""
    t, lead, trail = (
        np.asarray(times_s, dtype=np.float64),
        np.asarray(fz_lead_n, dtype=np.float64),
        np.asarray(fz_trail_n, dtype=np.float64),
    )
    require(len(t) == len(lead) == len(trail), "Input lengths must match")
    if normalisation == NormalisationKind.percent_body_weight:
        require(
            body_mass_kg is not None and body_mass_kg > 0.0, "body_mass_kg required"
        )
        assert body_mass_kg is not None
        mass = float(body_mass_kg)
        lead = normalize_vertical_force(
            lead, body_mass_kg=mass, normalisation=normalisation, gravity=gravity
        )
        trail = normalize_vertical_force(
            trail, body_mass_kg=mass, normalisation=normalisation, gravity=gravity
        )
    return GRFPeakSummary(
        lead=_single_foot_peak(t, lead, t_top_s, t_impact_s, normalisation),
        trail=_single_foot_peak(t, trail, t_top_s, t_impact_s, normalisation),
    )


@dataclass(frozen=True)
class COPKinematics:
    """Target-direction COP displacement range and velocities."""

    range_m: float
    peak_velocity_m_s: float
    mean_velocity_m_s: float
    direction: tuple[float, float, float]


def compute_cop_target_kinematics(
    times_s: ArrayLike,
    cop_positions: ArrayLike,
    *,
    target_direction: Sequence[float] = COP_TARGET_DIRECTION_DEFAULT,
) -> COPKinematics:
    """Compute COP excursion range and velocity along the specified target direction."""
    t: Array = np.asarray(times_s, dtype=np.float64)
    cop: Array = np.asarray(cop_positions, dtype=np.float64)
    td: Array = np.asarray(target_direction, dtype=np.float64)
    require(len(t) >= 2, "At least two time samples required")
    norm = float(np.linalg.norm(td))
    require(norm > 0.0, "target_direction must be non-zero")
    u: Array = td / norm
    if cop.ndim == 1:
        proj: Array = cop
    elif cop.ndim == 2 and cop.shape[1] == 2:
        u2 = u[:2]
        n2 = float(np.linalg.norm(u2))
        require(n2 > 0.0, "2D direction must be non-zero")
        proj = cop @ (u2 / n2)
    else:
        proj = cop @ u

    dtup = (float(u[0]), float(u[1]), float(u[2]) if len(u) > 2 else 0.0)
    valid = proj[np.isfinite(proj)]
    if valid.size == 0:
        return COPKinematics(float("nan"), float("nan"), float("nan"), dtup)
    rng = float(np.max(valid) - np.min(valid))
    vel = np.gradient(proj, t)
    vf = vel[np.isfinite(vel)]
    return COPKinematics(
        rng,
        float(np.max(np.abs(vf))) if vf.size > 0 else float("nan"),
        float(np.mean(vf)) if vf.size > 0 else float("nan"),
        dtup,
    )


@dataclass(frozen=True)
class LeadFractionCrossing:
    """Timing of lead-fraction transition relative to swing events."""

    crossing_time_s: float
    dt_top_s: float
    dt_impact_s: float
    threshold: float
    crossed: bool


def compute_lead_fraction_crossing(
    times_s: ArrayLike,
    fz_lead_n: ArrayLike,
    fz_trail_n: ArrayLike,
    *,
    t_top_s: float,
    t_impact_s: float,
    threshold: float = LEAD_CROSSING_FRACTION,
    min_total_fz_n: float = 10.0,
) -> LeadFractionCrossing:
    """Find first upward crossing of lead vertical force fraction through threshold."""
    t, lead, trail = (
        np.asarray(times_s, dtype=np.float64),
        np.asarray(fz_lead_n, dtype=np.float64),
        np.asarray(fz_trail_n, dtype=np.float64),
    )
    require(len(t) == len(lead) == len(trail), "Input lengths must match")
    total = lead + trail
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total < min_total_fz_n, np.nan, lead / total)

    hits = np.flatnonzero(
        (frac[:-1] < threshold)
        & (frac[1:] >= threshold)
        & np.isfinite(frac[:-1])
        & np.isfinite(frac[1:])
    )
    if hits.size == 0:
        return LeadFractionCrossing(
            float("nan"), float("nan"), float("nan"), threshold, False
        )
    i = int(hits[0])
    dt, df = float(t[i + 1] - t[i]), float(frac[i + 1] - frac[i])
    tc = (
        float(t[i])
        if abs(df) < 1e-12
        else float(t[i]) + (threshold - float(frac[i])) * (dt / df)
    )
    return LeadFractionCrossing(
        tc,
        tc - t_top_s if math.isfinite(t_top_s) else float("nan"),
        tc - t_impact_s if math.isfinite(t_impact_s) else float("nan"),
        threshold,
        True,
    )


__all__ = [
    "COP_TARGET_DIRECTION_DEFAULT",
    "LEAD_CROSSING_FRACTION",
    "STANDARD_GRAVITY_M_S2",
    "SUPPORTED_MODELLED_ENGINES",
    "COPKinematics",
    "EventAnchoredPeak",
    "GRFChannelObservability",
    "GRFPeakSummary",
    "GRFSourceAttribution",
    "LeadFractionCrossing",
    "compute_cop_target_kinematics",
    "compute_event_anchored_peaks",
    "compute_lead_fraction_crossing",
    "format_channel_value",
    "normalize_vertical_force",
    "sanitize_channel_value",
    "validate_unobserved_channel",
]
