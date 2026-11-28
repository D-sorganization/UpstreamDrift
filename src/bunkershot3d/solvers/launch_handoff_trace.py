"""Signed F0 launch handoff trace and timestep convergence gate (#15805, WP-2)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.core.contracts import require

from ..ball.lie import BallProperties
from ..ball.pipeline import HEAD_FRAME_TO_FLIGHT_TRANSFORM
from ..ball.splash import BallLaunchResult
from .envelope import GRAVITY_M_S2
from .exceptions import SolverError, SolverInputError

DEFAULT_BALL_MASS_KG: float = BallProperties().mass_kg
DEFAULT_GRAVITY_M_S2: float = GRAVITY_M_S2
DEFAULT_MIN_TRANSFER_RATIO: float = 0.10
DEFAULT_MAX_DT_REL_DIFF: float = 0.02
DEFAULT_CARRY_REL_TOL: float = 0.05
DEFAULT_CARRY_ABS_TOL_M: float = 0.05
_EPS: float = 1e-12
_FLIGHT_ROT = np.asarray(HEAD_FRAME_TO_FLIGHT_TRANSFORM, dtype=np.float64)


def _req(name: str, value: float, *, pos: bool = False) -> float:
    """Validate that ``value`` is finite and non-negative (or > 0 if ``pos``)."""
    val = float(value)
    bad = (val <= 0.0) if pos else (val < 0.0)
    if not math.isfinite(val) or bad:
        raise SolverInputError(
            f"{name} must be {'> 0' if pos else '>= 0'}, got {value!r}"
        )
    return val


def _vec3(name: str, value: ArrayLike) -> tuple[float, float, float]:
    """Coerce ``value`` to a finite 3-tuple of floats."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise SolverInputError(f"{name} must be a finite 3-vector, got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class LaunchHandoffTrace:
    """Signed SI-unit momentum trace across the club-sand-ball handoff."""

    contact_duration_s: float
    impulse_club_n_s: tuple[float, float, float]
    impulse_club_mag_n_s: float
    ball_mass_kg: float
    ball_launch_velocity_m_s: tuple[float, float, float]
    ball_momentum_kg_m_s: tuple[float, float, float]
    ball_momentum_mag_n_s: float
    transfer_ratio: float
    horizontal_alignment: float
    sign_inverted: bool
    collapsed_transfer: bool
    over_unity_transfer: bool
    passed: bool
    faults: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate physical units, finiteness, and pass/fault consistency."""
        _req("ball_mass_kg", self.ball_mass_kg, pos=True)
        for k in (
            "contact_duration_s",
            "impulse_club_mag_n_s",
            "ball_momentum_mag_n_s",
            "transfer_ratio",
        ):
            _req(k, float(getattr(self, k)))
        for k in (
            "impulse_club_n_s",
            "ball_launch_velocity_m_s",
            "ball_momentum_kg_m_s",
        ):
            _vec3(k, getattr(self, k))
        if self.passed == bool(self.faults):
            raise SolverInputError("passed flag must match empty faults")


@dataclass(frozen=True)
class TimestepConvergenceReport:
    """Result of the timestep-halving launch-speed convergence gate."""

    dt_s: float
    half_dt_s: float
    speed_dt_m_s: float
    speed_half_dt_m_s: float
    rel_diff: float
    max_rel_diff: float
    passed: bool
    fault: str | None = None

    def __post_init__(self) -> None:
        """Validate timestep parameters and gate consistency."""
        for name in ("dt_s", "half_dt_s", "max_rel_diff"):
            _req(name, float(getattr(self, name)), pos=True)
        if self.passed != (self.fault is None):
            raise SolverInputError("passed flag must match fault presence")


@dataclass(frozen=True)
class LaunchCarryCheck:
    """Carry reproduction comparison between recorded and replayed launch."""

    recorded_carry_m: float
    replayed_carry_m: float
    abs_error_m: float
    rel_error: float
    tolerance_m: float
    apex_m: float
    flight_model: str
    passed: bool
    fault: str | None = None

    def __post_init__(self) -> None:
        """Validate carry measurements, error bounds, and diagnostic state."""
        for name in (
            "recorded_carry_m",
            "replayed_carry_m",
            "abs_error_m",
            "tolerance_m",
            "apex_m",
        ):
            _req(name, float(getattr(self, name)))
        if self.passed != (self.fault is None):
            raise SolverInputError("passed flag must match fault presence")


def _contact_duration(t: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    """Integrate active contact duration over sample midpoints."""
    if not bool(mask.any()):
        return 0.0
    n = int(t.shape[0])
    w = np.zeros(n, dtype=np.float64)
    w[0], w[-1] = 0.5 * (t[1] - t[0]), 0.5 * (t[-1] - t[-2])
    if n > 2:
        w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return float(w[mask].sum())


def _integrate_impulse(
    times_s: ArrayLike,
    forces_n: ArrayLike,
    contact_mask: ArrayLike | None,
    *,
    reaction_on_head: bool,
    head_to_flight_frame: bool,
) -> tuple[tuple[float, float, float], float]:
    """Integrate reaction force over contact duration into flight-frame J_club."""
    t = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if t.shape[0] < 2 or not np.all(np.isfinite(t)) or not bool(np.all(np.diff(t) > 0)):
        raise SolverInputError("times_s must have >= 2 strictly increasing values")
    f = np.asarray(forces_n, dtype=np.float64)
    if f.shape != (t.shape[0], 3) or not np.all(np.isfinite(f)):
        raise SolverInputError("forces_n must be a finite (n, 3) array")
    mask = (
        np.einsum("ij,ij->i", f, f) > 0.0
        if contact_mask is None
        else np.asarray(contact_mask, dtype=bool).reshape(-1)
    )
    if mask.shape != t.shape:
        raise SolverInputError("contact_mask length must match times_s")
    rot = _FLIGHT_ROT.T if head_to_flight_frame else np.eye(3)
    f_club = (-f if reaction_on_head else f.copy()) @ rot
    j = np.trapezoid(np.where(mask[:, None], f_club, 0.0), x=t, axis=0)
    return (float(j[0]), float(j[1]), float(j[2])), _contact_duration(t, mask)


def _classify_faults(
    j: tuple[float, float, float],
    p: tuple[float, float, float],
    v: tuple[float, float, float],
    min_ratio: float,
) -> tuple[bool, bool, bool, float, float, tuple[str, ...]]:
    """Classify sign inversion, collapsed transfer, and conservation faults."""
    jm, pm = math.hypot(*j), math.hypot(*p)
    ratio = pm / jm if jm > _EPS else 0.0
    faults: list[str] = []
    col = over = inv = False
    if jm <= _EPS:
        col = True
        faults.append("Missing club impulse (|J_club| <= 1e-12 N*s).")
    elif pm <= _EPS:
        col = True
        faults.append(f"Missing ball momentum: |p_ball|=0 vs |J_club|={jm:.4g} N*s.")
    elif ratio < min_ratio:
        col = True
        faults.append(f"Collapsed transfer: {pm:.4g} vs {jm:.4g} N*s ({ratio:.4f}).")
    if jm > _EPS and ratio > 1.0 + 1e-9:
        over = True
        faults.append(f"Ball momentum {pm:.4g} exceeds |J_club|={jm:.4g} N*s.")
    jh, ph = math.hypot(j[0], j[1]), math.hypot(p[0], p[1])
    align = (j[0] * p[0] + j[1] * p[1]) / (jh * ph) if jh > _EPS and ph > _EPS else 0.0
    if jh > _EPS and ph > _EPS and align <= 0.0:
        inv = True
        faults.append(f"Horizontal sign inversion (cos={align:.4f}).")
    if pm > _EPS and v[2] <= 0.0:
        inv = True
        faults.append(f"Vertical launch sign inversion: vz={v[2]:.4g} m/s <= 0.")
    return inv, col, over, ratio, align, tuple(faults)


def trace_f0_launch_handoff(
    times_s: ArrayLike,
    forces_n: ArrayLike,
    ball_velocity_m_s: ArrayLike,
    *,
    ball_mass_kg: float = DEFAULT_BALL_MASS_KG,
    contact_mask: ArrayLike | None = None,
    min_transfer_ratio: float = DEFAULT_MIN_TRANSFER_RATIO,
    reaction_on_head: bool = True,
    head_to_flight_frame: bool = True,
) -> LaunchHandoffTrace:
    """Integrate SI reaction force and verify signed ball momentum handoff."""
    mass = _req("ball_mass_kg", ball_mass_kg, pos=True)
    min_r = _req("min_transfer_ratio", min_transfer_ratio, pos=True)
    require(min_r <= 1.0, "min_transfer_ratio must not exceed 1.0")
    v = _vec3("ball_velocity_m_s", ball_velocity_m_s)
    j, dur = _integrate_impulse(
        times_s,
        forces_n,
        contact_mask,
        reaction_on_head=reaction_on_head,
        head_to_flight_frame=head_to_flight_frame,
    )
    p = (mass * v[0], mass * v[1], mass * v[2])
    inv, col, over, r, al, flt = _classify_faults(j, p, v, min_r)
    jm, pm = math.hypot(*j), math.hypot(*p)
    return LaunchHandoffTrace(
        dur, j, jm, mass, v, p, pm, r, al, inv, col, over, not flt, flt
    )


def _extract_speed(outcome: float | BallLaunchResult) -> float:
    """Extract scalar launch speed [m/s] from a numeric result or launch record."""
    return (
        float(outcome.ball_speed_m_s)
        if isinstance(outcome, BallLaunchResult)
        else float(outcome)
    )


def verify_timestep_halving_convergence(
    evaluate_launch_speed: Callable[[float], float | BallLaunchResult],
    dt_s: float,
    *,
    max_rel_diff: float = DEFAULT_MAX_DT_REL_DIFF,
    strict: bool = False,
) -> TimestepConvergenceReport:
    """Compare launch speed at ``dt_s`` vs ``dt_s / 2`` and fail closed if > 2%."""
    dt, tol = _req("dt_s", dt_s, pos=True), _req("max_rel_diff", max_rel_diff, pos=True)
    half_dt = 0.5 * dt
    v1, v2 = (
        _extract_speed(evaluate_launch_speed(dt)),
        _extract_speed(evaluate_launch_speed(half_dt)),
    )
    if not math.isfinite(v1) or not math.isfinite(v2) or v1 <= 0.0 or v2 <= 0.0:
        err = f"Timestep gate failed closed: invalid speed ({v1!r}, {v2!r})."
        if strict:
            raise SolverError(err)
        return TimestepConvergenceReport(dt, half_dt, v1, v2, math.inf, tol, False, err)
    rel_diff = abs(v1 - v2) / abs(v2)
    passed = rel_diff <= tol
    msg = f"Timestep convergence failure: rel_diff={rel_diff:.4%} > {tol:.4%}."
    fault_msg: str | None = None if passed else msg
    if strict and not passed:
        raise SolverError(fault_msg or "timestep convergence gate failed")
    return TimestepConvergenceReport(
        dt, half_dt, v1, v2, rel_diff, tol, passed, fault_msg
    )


def verify_launch_carry_reproduction(
    launch: BallLaunchResult | ArrayLike,
    recorded_carry_m: float,
    *,
    rel_tol: float = DEFAULT_CARRY_REL_TOL,
    abs_tol_m: float = DEFAULT_CARRY_ABS_TOL_M,
    gravity_m_s2: float = DEFAULT_GRAVITY_M_S2,
    flight_carry_fn: Callable[[BallLaunchResult], float] | None = None,
) -> LaunchCarryCheck:
    """Replay recorded launch vector and detect carry mismatch beyond tolerance."""
    rec, r_tol = _req("recorded_carry_m", recorded_carry_m), _req("rel_tol", rel_tol)
    a_tol, g = (
        _req("abs_tol_m", abs_tol_m),
        _req("gravity_m_s2", gravity_m_s2, pos=True),
    )
    vel = _vec3(
        "launch",
        launch.ball_velocity if isinstance(launch, BallLaunchResult) else launch,
    )
    vz = vel[2]
    apex = (vz * vz) / (2.0 * g) if vz > 0.0 else 0.0
    if flight_carry_fn is not None:
        if not isinstance(launch, BallLaunchResult):
            raise SolverInputError("flight_carry_fn requires a BallLaunchResult")
        replayed = _req("flight_carry_fn", float(flight_carry_fn(launch)))
        model = "flight_entry"
    else:
        replayed = math.hypot(vel[0], vel[1]) * (2.0 * vz / g) if vz > 0.0 else 0.0
        model = "ballistic"
    err, denom = abs(replayed - rec), max(abs(rec), abs(replayed))
    rel_err, tol_m = err / denom if denom > _EPS else 0.0, max(a_tol, r_tol * denom)
    passed = err <= tol_m and (vz > 0.0 or rec <= a_tol)
    msg = f"Launch carry mismatch ({model}): {replayed:.4f} vs {rec:.4f} m."
    fault: str | None = None if passed else msg
    return LaunchCarryCheck(
        rec, replayed, err, rel_err, tol_m, apex, model, passed, fault
    )
