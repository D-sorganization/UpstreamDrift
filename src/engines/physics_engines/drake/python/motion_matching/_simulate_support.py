"""Helpers extracted from Drake ``simulate.py`` (issue #12843 budget refactor).

Everything here is pydrake-lazy: ``pydrake`` is only imported inside functions
that need it, so the module loads on hosts without Drake. ``simulate.py``
re-exports the public names (``evaluate_bernstein_torque`` ...) and keeps the
plant-construction glue that tests patch (``load_humanoid_into_plant``).

This module imports nothing from ``simulate`` (no import cycle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.polynomial_torque import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
)
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque as _evaluate_polynomial_torque_matrix,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from pydrake.multibody.plant import MultibodyPlant

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutConfig",
    "evaluate_bernstein_torque",
    "torque_at",
]


# ---------------------------------------------------------------------------
# Torque evaluation
# ---------------------------------------------------------------------------


def evaluate_bernstein_torque(
    coeffs: NDArray[np.float64],
    t: float,
    T_s: float = 1.0,
    t0: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate per-joint degree-6 torques in Bernstein polynomial basis.

    tau_j(t) = sum_{k=0}^6 c_{j,k} B_{k,6}((t - t0) / T_s)
    where B_{k,6}(s) = comb(6, k) * s^k * (1 - s)^(6 - k).

    Args:
        coeffs: 2D array of shape ``(n_joints, 7)`` containing control points.
        t: Time in seconds.
        T_s: Horizon duration in seconds.
        t0: Time offset in seconds.

    Returns:
        1D array of shape ``(n_joints,)`` with applied torques at time ``t``.
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim != 2:
        raise ValueError(f"coeffs must be 2D (n_joints, 7); got ndim={coeffs_arr.ndim}")
    if coeffs_arr.shape[1] != COEFFS_PER_JOINT:
        raise ValueError(
            f"coeffs must have {COEFFS_PER_JOINT} columns; got shape {coeffs_arr.shape}"
        )
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")
    if T_s <= 0.0:
        raise ValueError(f"T_s must be positive, got {T_s!r}")

    s = (t - t0) / T_s
    k_vals = np.arange(COEFFS_PER_JOINT, dtype=np.float64)
    b_weights = np.array(
        [comb(POLY_DEGREE, k) for k in range(COEFFS_PER_JOINT)], dtype=np.float64
    )
    basis = b_weights * (s**k_vals) * ((1.0 - s) ** (POLY_DEGREE - k_vals))
    return coeffs_arr @ basis


def torque_at(
    coeffs: NDArray[np.float64], t: float, basis: str, T_s: float, t0: float
) -> NDArray[np.float64]:
    """Joint torques at time ``t`` for the ``power`` or ``bernstein`` basis."""
    if basis == "bernstein":
        return evaluate_bernstein_torque(coeffs, t, T_s=T_s, t0=t0)
    return _evaluate_polynomial_torque_matrix(coeffs, t - t0)


def build_polynomial_torque_system(
    theta: NDArray[np.float64],
    n_actuators: int,
    basis: str = "power",
    T_s: float = 1.0,
    t0: float = 0.0,
) -> Any:
    """Build a Drake ``LeafSystem`` that emits the polynomial torque."""
    # Explicit import per CLAUDE.md.
    from pydrake.systems.framework import BasicVector, LeafSystem  # noqa: PLC0415

    coeffs = np.ascontiguousarray(theta, dtype=np.float64).reshape(
        n_actuators, COEFFS_PER_JOINT
    )

    class _PolynomialTorqueSource(LeafSystem):
        """Stateflow-equivalent per-joint continuous torque polynomial."""

        def __init__(self) -> None:
            LeafSystem.__init__(self)
            self._coeffs = coeffs
            self._n = n_actuators
            self._basis = basis
            self._T_s = T_s
            self._t0 = t0
            self.DeclareVectorOutputPort(
                "tau",
                BasicVector(n_actuators),
                self._calc_output,
            )

        def _calc_output(self, context: Any, output: Any) -> None:
            t = float(context.get_time())
            output.SetFromVector(
                torque_at(self._coeffs, t, self._basis, self._T_s, self._t0)
            )

    return _PolynomialTorqueSource()


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

_FALLBACK_BODIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("club_grip", "grip"), ("mid_hands", "grip", "right_hand")),
    (("clubhead", "club_head"), ("club_head", "club_shaft")),
    (("mid_hands",), ("club_grip", "grip")),
)


def _resolve_body_name(plant: Any, body_name: str) -> str | None:
    """Return ``body_name`` or the first existing fallback; ``None`` if absent."""
    has_body = getattr(plant, "HasBodyNamed", None)
    if has_body is not None and bool(has_body(body_name)):
        return body_name
    candidates: tuple[str, ...] = ()
    for names, fallbacks in _FALLBACK_BODIES:
        if body_name in names:
            candidates = fallbacks
            break
    for candidate in candidates:
        if has_body is not None and bool(has_body(candidate)):
            return candidate
    return None


def resolve_world_pose(
    plant: MultibodyPlant,
    plant_context: Any,
    body_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Forward-kinematics: return ``(position, quaternion[w,x,y,z])`` for ``body_name``."""
    target_name = _resolve_body_name(plant, body_name)
    if target_name is None:
        return None

    body = plant.GetBodyByName(target_name)
    transform = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        body.body_frame(),
    )
    pos = np.asarray(transform.translation(), dtype=np.float64)
    quat = transform.rotation().ToQuaternion()
    # Drake quaternion exposes w, x, y, z scalars.
    quat_arr = np.array([quat.w(), quat.x(), quat.y(), quat.z()], dtype=np.float64)
    if pos.shape != (3,) or quat_arr.shape != (4,):
        return None
    return pos, quat_arr


# ---------------------------------------------------------------------------
# Rollout recording
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Outcome of stepping the Drake simulator over the output time grid."""

    solver_status: str
    error: BaseException | None


@dataclass
class RolloutLogs:
    """Pre-allocated, NaN-filled output buffers mutated in place per step."""

    q: NDArray[np.float64]
    v: NDArray[np.float64]
    tau: NDArray[np.float64]
    grip: NDArray[np.float64]
    grip_quat: NDArray[np.float64]
    clubhead: NDArray[np.float64]
    club_quat: NDArray[np.float64]
    kinetic_energy: NDArray[np.float64]
    potential_energy: NDArray[np.float64]


@dataclass(frozen=True)
class RolloutConfig:
    """Scalar settings for :func:`record_rollout` (parameter object, #12843)."""

    n_actuators: int
    grip_body_name: str
    clubhead_body_name: str
    basis: str = "power"
    T_s: float = 1.0
    t0: float = 0.0
    compute_energy: bool = True


def allocate_logs(n_t: int, n_q: int, n_v: int, n_actuators: int) -> RolloutLogs:
    """Return NaN-filled log buffers for an ``n_t``-sample rollout."""

    def _nan(*shape: int) -> NDArray[np.float64]:
        return np.full(shape, np.nan, dtype=np.float64)

    return RolloutLogs(
        q=_nan(n_t, n_q),
        v=_nan(n_t, n_v),
        tau=_nan(n_t, n_actuators),
        grip=_nan(n_t, 3),
        grip_quat=_nan(n_t, 4),
        clubhead=_nan(n_t, 3),
        club_quat=_nan(n_t, 4),
        kinetic_energy=_nan(n_t),
        potential_energy=_nan(n_t),
    )


def _energy_sample(plant: Any, method: str, plant_ctx: Any) -> tuple[float, bool]:
    """Return ``(value, ok)``; ``value`` is NaN and ``ok`` False when unusable."""
    if not hasattr(plant, method):
        return float("nan"), False
    try:
        val = float(getattr(plant, method)(plant_ctx))
    except (TypeError, ValueError):
        return float("nan"), False
    if not np.isfinite(val):
        return float("nan"), False
    return val, True


def _record_energy(plant: Any, plant_ctx: Any, logs: RolloutLogs, idx: int) -> bool:
    """Log KE/PE for sample ``idx``; return True if either was unusable."""
    has_partial = False
    for method, buf in (
        ("CalcKineticEnergy", logs.kinetic_energy),
        ("CalcPotentialEnergy", logs.potential_energy),
    ):
        buf[idx], ok = _energy_sample(plant, method, plant_ctx)
        has_partial = has_partial or not ok
    return has_partial


def _record_poses(
    plant: Any, plant_ctx: Any, logs: RolloutLogs, idx: int, cfg: RolloutConfig
) -> bool:
    """Log grip/clubhead poses for sample ``idx``; True if either is missing."""
    has_partial = False
    grip_pose = resolve_world_pose(plant, plant_ctx, cfg.grip_body_name)
    if grip_pose is not None:
        logs.grip[idx, :], logs.grip_quat[idx, :] = grip_pose
    else:
        has_partial = True
    club_pose = resolve_world_pose(plant, plant_ctx, cfg.clubhead_body_name)
    if club_pose is not None:
        logs.clubhead[idx, :], logs.club_quat[idx, :] = club_pose
    else:
        has_partial = True
    return has_partial


def record_rollout(
    *,
    simulator: Any,
    plant: MultibodyPlant,
    grid: NDArray[np.float64],
    theta: NDArray[np.float64],
    logs: RolloutLogs,
    config: RolloutConfig,
) -> RolloutResult:
    """Step the simulator over ``grid`` and record state / torque / FK logs."""
    n_act = config.n_actuators
    coeffs = (
        theta.reshape(n_act, COEFFS_PER_JOINT)
        if n_act > 0
        else np.zeros((0, COEFFS_PER_JOINT), dtype=np.float64)
    )
    has_partial = False
    for idx, t_target in enumerate(grid):
        if t_target > 0.0:
            try:
                simulator.AdvanceTo(float(t_target))
            except RuntimeError as exc:
                logger.exception(
                    "Drake AdvanceTo diverged at t=%.6f s (step %d/%d)",
                    float(t_target),
                    idx,
                    len(grid),
                )
                return RolloutResult(solver_status="failed", error=exc)

        plant_ctx = plant.GetMyContextFromRoot(simulator.get_context())
        logs.q[idx, :] = plant.GetPositions(plant_ctx)
        logs.v[idx, :] = plant.GetVelocities(plant_ctx)
        if n_act > 0:
            logs.tau[idx, :] = torque_at(
                coeffs, float(t_target), config.basis, config.T_s, config.t0
            )
        has_partial = _record_poses(plant, plant_ctx, logs, idx, config) or has_partial
        if config.compute_energy:
            has_partial = _record_energy(plant, plant_ctx, logs, idx) or has_partial

    return RolloutResult(
        solver_status="partial" if has_partial else "success", error=None
    )


# ---------------------------------------------------------------------------
# simulate_with_coefficients building blocks
# ---------------------------------------------------------------------------


def normalize_theta(theta: Any) -> NDArray[np.float64]:
    """Coerce ``theta`` to a contiguous finite 1-D float array of 7*n entries.

    Raises:
        ValueError: If ``theta`` is not 1-D, non-finite, empty or not a
            multiple of the per-joint coefficient count.
    """
    arr = np.ascontiguousarray(theta, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"theta must be 1-D; got shape {arr.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "theta must contain only finite values"
        raise ValueError(msg)
    if arr.shape[0] % COEFFS_PER_JOINT != 0 or arr.shape[0] == 0:
        msg = (
            f"theta length must be a positive multiple of {COEFFS_PER_JOINT} "
            f"(7 coefficients per joint); got {arr.shape[0]}"
        )
        raise ValueError(msg)
    return arr


def check_initial_pose_type(initial_pose: Any) -> None:
    """Raise ``TypeError`` unless ``initial_pose`` is ``None`` or a dict."""
    if initial_pose is not None and not isinstance(initial_pose, dict):
        msg = (
            "initial_pose must be a dict with optional keys 'q' / 'v' or "
            f"None; got {type(initial_pose).__name__}"
        )
        raise TypeError(msg)


def check_theta_matches_actuators(theta: NDArray[np.float64], n_actuators: int) -> None:
    """Raise ``ValueError`` if ``theta`` is sized for a different actuator count."""
    expected = n_actuators * COEFFS_PER_JOINT
    if expected > 0 and theta.shape[0] != expected:
        msg = (
            "theta is sized for "
            f"{theta.shape[0] // COEFFS_PER_JOINT} joint(s) "
            f"(length {theta.shape[0]}) but the plant has {n_actuators} "
            f"actuator(s), so it expects theta of length {expected} "
            f"({n_actuators} * {COEFFS_PER_JOINT}). Supplying a mismatched "
            "theta would leave the actuation port disconnected while tau_log "
            "still recorded nonzero torques (issue #7725). Resize theta to the "
            "plant's actuator count."
        )
        raise ValueError(msg)


def _checked_state(value: Any, expected: int, key: str) -> NDArray[np.float64]:
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.shape != (expected,):
        msg = f"initial_pose['{key}'] must have shape ({expected},); got {arr.shape}"
        raise ValueError(msg)
    return arr


def apply_initial_pose(
    plant: Any, plant_context: Any, initial_pose: dict[str, Any] | None
) -> None:
    """Write optional ``q`` / ``v`` overrides from ``initial_pose`` into the plant."""
    if initial_pose is None:
        return
    q0 = initial_pose.get("q")
    v0 = initial_pose.get("v")
    if q0 is not None:
        plant.SetPositions(
            plant_context, _checked_state(q0, plant.num_positions(), "q")
        )
    if v0 is not None:
        plant.SetVelocities(
            plant_context, _checked_state(v0, plant.num_velocities(), "v")
        )


def finite_difference_qdd(
    v_log: NDArray[np.float64], sample_rate_hz: float, enabled: bool
) -> NDArray[np.float64]:
    """Central/one-sided finite-difference accelerations from the velocity log."""
    qdd = np.zeros_like(v_log)
    if enabled and v_log.shape[0] >= 2:
        dt = 1.0 / sample_rate_hz
        qdd[1:-1, :] = (v_log[2:, :] - v_log[:-2, :]) / (2.0 * dt)
        qdd[0, :] = (v_log[1, :] - v_log[0, :]) / dt
        qdd[-1, :] = (v_log[-1, :] - v_log[-2, :]) / dt
    return qdd


def assert_finite_on_success(solver_status: str, out: Any) -> None:
    """Postcondition (cross-engine §2.2): on success, q/qd/tau are finite."""
    if solver_status != "success":
        return
    for name, arr in (("q", out.q), ("qd", out.qd), ("tau", out.tau)):
        if not np.all(np.isfinite(arr)):
            msg = f"Postcondition: SimOut.{name} contains non-finite values"
            raise AssertionError(msg)
