"""Python mirror of MATLAB ``compute_cost.m``.

This module computes the scalar swing-matching cost defined in
``motion_matching/shared/COST_FUNCTION_SPEC.md``. Numeric output must agree
with the MATLAB reference implementation (#015) to within ``1e-9`` on
identical inputs; see ``test_python_matches_matlab_compute_cost``.

Public API:
    CostOptions    -- frozen dataclass mirroring ``default_cost_options.m``.
    CostBreakdown  -- per-term breakdown returned alongside the scalar ``J``.
    SimOutput      -- minimal dataclass for the fields read from ``sim_fn``.
    compute_cost   -- main entry point.
    compute_total_work -- regularizer helper, also useful standalone.

``sim_fn`` is injected as a callable so tests can drive the cost function
without the full Simscape/Python forward model wired up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts.decorators import (
    postcondition,
    precondition,
)

from ._geodesic import quaternion_geodesic_angles
from .club_target import ClubTarget

__all__ = [
    "CostBreakdown",
    "CostOptions",
    "SimOutput",
    "compute_cost",
    "compute_total_work",
]

RegularizerName = Literal["total_work", "peak_power", "torque_l2", "coeff_l2"]
QRepr = Literal["quaternion", "rotmat"]
TimeAlignment = Literal["impact", "address", "none"]


@dataclass(frozen=True)
class CostOptions:
    """Mirror of ``default_cost_options.m``.

    Attributes:
        w_position:        Position-term weight (per metre^2).
        w_orientation:     Orientation-term weight (per radian^2).
        w_anchor_impact:   Impact-anchor multiplier on the position term.
        regularizer:       Which regularizer to apply.
        lambda_:           Regularizer strength (``lambda`` is reserved).
        q_orientation_repr: Quaternion vs rotation-matrix representation.
        time_alignment:    How target / sim are time-aligned upstream.
        resample_to_hz:    Target sample rate of the shared timegrid.
    """

    w_position: float = 1.0
    w_orientation: float = 0.1
    w_anchor_impact: float = 10.0
    regularizer: RegularizerName = "total_work"
    lambda_: float = 1e-4
    q_orientation_repr: QRepr = "quaternion"
    time_alignment: TimeAlignment = "impact"
    resample_to_hz: float = 1000.0


@dataclass(frozen=True)
class CostBreakdown:
    """Per-term breakdown summing to ``total``.

    Each field is the *weighted* contribution to ``J`` so that
    ``position + orientation + impact_anchor + regularizer == total``.
    """

    position: float
    orientation: float
    impact_anchor: float
    regularizer: float
    total: float


@dataclass(frozen=True)
class SimOutput:
    """Subset of the MATLAB ``sim_out`` struct read by the cost function.

    Attributes:
        butt:      ``(N, 3)`` simulated butt position over the timegrid (m).
        clubhead:  ``(N, 3)`` simulated clubhead position (m).
        club_quat: ``(N, 4)`` unit quaternions ``[w, x, y, z]``.
        time:      ``(N,)`` monotonic time in seconds. Required for the
                   ``total_work`` and ``torque_l2`` regularizers.
        tau:       ``(N, n_joints)`` joint torques (N*m). Optional unless
                   the regularizer reads it.
        omega:     ``(N, n_joints)`` joint angular velocities (rad/s).
                   Optional unless the regularizer reads it.
    """

    butt: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    time: NDArray[np.float64] | None = None
    tau: NDArray[np.float64] | None = None
    omega: NDArray[np.float64] | None = None


# --- compute_total_work --------------------------------------------------


def _require_field(
    value: NDArray[np.float64] | None,
    name: str,
) -> NDArray[np.float64]:
    """Return ``value`` or raise ``ValueError`` if it is ``None``."""
    if value is None:
        raise ValueError(f"sim_out.{name} is required for this regularizer")
    return value


@postcondition(
    lambda result: np.isfinite(result) and result >= 0.0,
    "compute_total_work must return finite, non-negative scalar",
)
def compute_total_work(sim_out: SimOutput) -> float:
    """Total mechanical work integral across all joints.

    ``W = trapz(time, sum(|tau * omega|, axis=1))``. Mirrors
    ``compute_total_work.m``.
    """
    time = _require_field(sim_out.time, "time").reshape(-1)
    tau = _require_field(sim_out.tau, "tau")
    omega = _require_field(sim_out.omega, "omega")
    if tau.shape != omega.shape:
        raise ValueError(
            f"tau and omega must have the same shape; got {tau.shape} vs {omega.shape}"
        )
    if tau.shape[0] != time.shape[0]:
        raise ValueError(
            "tau/omega rows must match length(time); "
            f"got {tau.shape[0]} vs {time.shape[0]}"
        )
    integrand = np.sum(np.abs(tau * omega), axis=1)
    return float(np.trapezoid(integrand, time))


# --- per-term helpers (LOD <= 2) ----------------------------------------


def _position_term(sim_out: SimOutput, target: ClubTarget) -> float:
    """``mean_n( ||r_butt_diff||^2 + ||r_ch_diff||^2 )``."""
    db = sim_out.butt - target.butt
    dc = sim_out.clubhead - target.clubhead
    per_frame = np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1)
    return float(np.mean(per_frame))


def _orientation_term(sim_out: SimOutput, target: ClubTarget) -> float:
    """``mean_n( d_geo(R_sim, R_meas)^2 )`` via quaternion dot."""
    angles = quaternion_geodesic_angles(sim_out.club_quat, target.club_quat)
    return float(np.mean(angles * angles))


def _anchor_term(sim_out: SimOutput, target: ClubTarget) -> float:
    """``||r_ch_sim(t_impact) - r_ch_meas(t_impact)||^2``.

    MATLAB uses 1-based indexing; ``ClubTarget.impact_idx`` follows that
    convention so this helper subtracts 1 here.
    """
    n = target.clubhead.shape[0]
    k = int(target.impact_idx)
    if not (1 <= k <= n):
        raise ValueError(f"impact_idx must be in [1, {n}], got {k}")
    d = sim_out.clubhead[k - 1] - target.clubhead[k - 1]
    return float(np.dot(d, d))


def _regularizer_term(
    theta: NDArray[np.float64],
    sim_out: SimOutput,
    opts: CostOptions,
) -> float:
    """Dispatch on ``opts.regularizer``."""
    name = opts.regularizer
    if name == "total_work":
        return compute_total_work(sim_out)
    if name == "peak_power":
        tau = _require_field(sim_out.tau, "tau")
        omega = _require_field(sim_out.omega, "omega")
        return float(np.max(np.sum(np.abs(tau * omega), axis=1)))
    if name == "torque_l2":
        time = _require_field(sim_out.time, "time").reshape(-1)
        tau = _require_field(sim_out.tau, "tau")
        return float(np.trapezoid(np.sum(tau * tau, axis=1), time))
    if name == "coeff_l2":
        return float(np.dot(theta, theta))
    raise ValueError(
        f"unknown regularizer {name!r}; expected one of "
        "'total_work', 'peak_power', 'torque_l2', 'coeff_l2'"
    )


def _check_traj(
    arr: NDArray[np.float64],
    n: int,
    cols: int,
    name: str,
) -> None:
    """Raise ``ValueError`` if ``arr`` is not a real ``(n, cols)`` matrix."""
    if arr.ndim != 2 or arr.shape != (n, cols):
        raise ValueError(f"{name} must have shape ({n}, {cols}); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")


# --- public entry point --------------------------------------------------


def _check_compute_cost_args(
    theta: NDArray[np.float64],
    target: ClubTarget,
    sim_fn: Callable[[NDArray[np.float64]], SimOutput],
    opts: CostOptions,
) -> bool:
    """Precondition predicate for :func:`compute_cost`."""
    return (
        isinstance(theta, np.ndarray)
        and theta.ndim == 1
        and theta.size > 0
        and bool(np.all(np.isfinite(theta)))
        and isinstance(target, ClubTarget)
        and callable(sim_fn)
        and isinstance(opts, CostOptions)
    )


def _check_compute_cost_result(
    result: tuple[float, CostBreakdown],
) -> bool:
    """Postcondition predicate for :func:`compute_cost`."""
    j, terms = result
    if not (np.isfinite(j) and j >= 0.0):
        return False
    if abs(terms.total - j) > 1e-12 * max(1.0, abs(j)) * 8:
        return False
    return all(
        x >= 0.0
        for x in (
            terms.position,
            terms.orientation,
            terms.impact_anchor,
            terms.regularizer,
        )
    )


@precondition(
    _check_compute_cost_args,
    "compute_cost requires finite 1-D theta, ClubTarget target, callable sim_fn, "
    "CostOptions opts",
)
@postcondition(
    _check_compute_cost_result,
    "compute_cost must return finite J >= 0 with non-negative breakdown summing to J",
)
def compute_cost(
    theta: NDArray[np.float64],
    target: ClubTarget,
    sim_fn: Callable[[NDArray[np.float64]], SimOutput],
    opts: CostOptions = CostOptions(),
) -> tuple[float, CostBreakdown]:
    """Scalar swing-matching cost mirroring ``compute_cost.m``.

    Args:
        theta:  Real, finite 1-D coefficient vector.
        target: Validated :class:`ClubTarget` (#017).
        sim_fn: Callable ``theta -> SimOutput``. Injected for testability.
        opts:   :class:`CostOptions`; defaults match ``default_cost_options.m``.

    Returns:
        ``(J, terms)`` where ``J`` is the total cost and ``terms`` is the
        weighted breakdown that sums to ``J``.

    Raises:
        ValueError: If ``sim_fn`` returns mismatched shapes, NaN/Inf,
            or an out-of-range ``impact_idx``.
    """
    sim_out = sim_fn(theta)
    if not isinstance(sim_out, SimOutput):
        raise TypeError(f"sim_fn must return SimOutput, got {type(sim_out).__name__}")

    n = target.time.shape[0]
    _check_traj(sim_out.butt, n, 3, "sim_out.butt")
    _check_traj(sim_out.clubhead, n, 3, "sim_out.clubhead")
    _check_traj(sim_out.club_quat, n, 4, "sim_out.club_quat")

    pos = opts.w_position * _position_term(sim_out, target)
    ori = opts.w_orientation * _orientation_term(sim_out, target)
    anc = opts.w_anchor_impact * _anchor_term(sim_out, target)
    reg = opts.lambda_ * _regularizer_term(theta, sim_out, opts)
    total = pos + ori + anc + reg

    terms = CostBreakdown(
        position=pos,
        orientation=ori,
        impact_anchor=anc,
        regularizer=reg,
        total=total,
    )
    return total, terms
