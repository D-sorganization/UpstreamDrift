"""
Simulation engine for the golfer upper-body model.

Integrates the constrained equations of motion and stores results
in a GolferSimulationResult for GUI and analysis access.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .physics import JointLimitsNDOF

from .constraint_solver import (
    constrained_accelerations,
    constraint_forces,
    constraint_violation,
    equations_of_motion,
    project_to_constraints,
    project_velocity,
)
from .physics_golfer import (
    N_DOF,
    GolferParams,
    State,
    TorqueFunc,
    coriolis_matrix,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    net_joint_forces,
    potential_energy,
)
from .simulation_core import SimulationConfig, integrate_ode
from .simulation_result_base import TrajectoryResultMixin
from .validation import (
    require,
    require_dt,
    require_finite_array,
    require_positive,
    require_shape,
)

# Re-export from shared utility for backwards compatibility (DRY â€” #1041)
from .torque_utils import make_polynomial_torque  # noqa: F401

_log = logging.getLogger(__name__)

# Constraint violation thresholds for warning/abort.
_CONSTRAINT_WARN_TOL = 1e-4
_CONSTRAINT_ABORT_TOL = 1e-2

# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class GolferSimulationResult(TrajectoryResultMixin):
    """Stores the complete trajectory and provides derived-quantity accessors."""

    t: np.ndarray  # shape (n_steps,)
    states: np.ndarray  # shape (n_steps, 16)
    params: GolferParams
    torque_func: TorqueFunc

    def __post_init__(self) -> None:
        self._validate_trajectory(expected_state_width=2 * N_DOF)

    def q_at(self, idx: int) -> np.ndarray:
        """Generalized coordinates at time index."""
        self._check_idx(idx)
        return self.states[idx, :N_DOF]

    def qdot_at(self, idx: int) -> np.ndarray:
        """Generalized velocities at time index."""
        self._check_idx(idx)
        return self.states[idx, N_DOF:]

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        """8Ã—8 mass matrix at time index."""
        self._check_idx(idx)
        return mass_matrix(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def positions_at(self, idx: int) -> dict:
        """Forward kinematics at time index."""
        self._check_idx(idx)
        return forward_kinematics(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def torques_at(
        self, idx: int
    ) -> tuple[float, float, float, float, float, float, float]:
        """Applied driving torques at time index."""
        self._check_idx(idx)
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        """Joint accelerations at time index."""
        self._check_idx(idx)
        return constrained_accelerations(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def joint_forces_at(self, idx: int) -> dict:
        """Net joint forces at time index."""
        self._check_idx(idx)
        q = self.q_at(idx)
        qdot = self.qdot_at(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(q, qdot, qddot, self.params)  # type: ignore[no-any-return]

    def constraint_forces_at(self, idx: int) -> np.ndarray:
        """Lagrange multiplier (constraint) forces at time index."""
        self._check_idx(idx)
        return constraint_forces(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def constraint_violation_at(self, idx: int) -> float:
        """Constraint violation magnitude at time index."""
        self._check_idx(idx)
        return constraint_violation(self.states[idx], self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        """Coriolis/centrifugal torques at time index."""
        self._check_idx(idx)
        return coriolis_matrix(self.q_at(idx), self.qdot_at(idx), self.params)  # type: ignore[no-any-return]

    def gravity_at(self, idx: int) -> np.ndarray:
        """Gravitational torques at time index."""
        self._check_idx(idx)
        return gravity_vector(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def _kinetic_energy_at(self, state: np.ndarray) -> float:
        return float(kinetic_energy(state[:N_DOF], state[N_DOF:], self.params))

    def _potential_energy_at(self, state: np.ndarray) -> float:
        return float(potential_energy(state, self.params))

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Friction torques at time index."""
        self._check_idx(idx)
        return friction_torque_vector(self.qdot_at(idx), self.params)  # type: ignore[no-any-return]

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index.

        Overrides the base to spread the 7-joint torque_func output over the
        full N_DOF=8 vector before adding friction.
        """
        self._check_idx(idx)
        tau_drive = np.zeros(N_DOF)
        tau_drive[:7] = self.torque_func(self.t[idx])
        tau_friction = self.friction_torques_at(idx)
        result: np.ndarray = tau_drive + tau_friction
        return result


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


_GOLFER_SETTINGS = frozenset(
    {"rtol", "atol", "alpha", "beta", "torque_limits", "limits", "clamp"}
)


def _joint_limit_correction(
    y: np.ndarray, params: GolferParams, limits: JointLimitsNDOF
) -> np.ndarray:
    """Return the qddot correction (M^-1 * tau_limit) from joint-limit penalties.

    Only the 7 actuated DOFs (not the club theta) carry limit torques; the
    correction uses the full 8-DOF mass matrix.
    """
    from .physics import joint_limit_torque_ndof

    q = y[:N_DOF]
    qdot = y[N_DOF:]
    tau_limit = joint_limit_torque_ndof(q[:7], qdot[:7], limits)
    M = mass_matrix(q, params)
    tau_full = np.zeros(N_DOF)
    tau_full[:7] = tau_limit
    try:
        return np.linalg.solve(M, tau_full)  # type: ignore[no-any-return]
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, tau_full, rcond=None)[0]  # type: ignore[no-any-return]


class _ConstraintDriftMonitor:
    """Tracks the worst Baumgarte constraint violation seen during a run."""

    def __init__(self) -> None:
        self.max_violation = 0.0

    def observe(self, y: np.ndarray, t: float, params: GolferParams) -> None:
        viol = constraint_violation(y, params)
        if viol > self.max_violation:
            self.max_violation = viol
        if viol > _CONSTRAINT_WARN_TOL:
            _log.warning(
                "Constraint violation %.3e at t=%.4f (warn threshold=%.3e)",
                viol,
                t,
                _CONSTRAINT_WARN_TOL,
            )

    def report(self) -> None:
        """Postcondition: log if constraint drift left the bounded range."""
        max_viol = self.max_violation
        if max_viol > _CONSTRAINT_ABORT_TOL:
            _log.error(
                "Excessive constraint drift: max violation=%.3e > abort threshold=%.3e",
                max_viol,
                _CONSTRAINT_ABORT_TOL,
            )
        elif max_viol > _CONSTRAINT_WARN_TOL:
            _log.warning("Max constraint violation=%.3e during simulation", max_viol)


def _make_golfer_rhs(
    params: GolferParams,
    torque_func: TorqueFunc,
    cfg: SimulationConfig,
    monitor: _ConstraintDriftMonitor,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the constrained golfer ODE right-hand side for ``integrate_ode``."""
    alpha = 5.0 if cfg.alpha is None else cfg.alpha
    beta = 5.0 if cfg.beta is None else cfg.beta
    torque_limits = cfg.effective_torque_limits
    limits: JointLimitsNDOF | None = cfg.limits

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        require(t is not None, "t must be provided")
        dydt = equations_of_motion(
            y, t, params, torque_func, alpha, beta, torque_limits
        )
        # Apply joint limit penalty torques if enabled
        if limits is not None:
            dydt[N_DOF:] += _joint_limit_correction(y, params, limits)
        # Monitor constraint drift (Baumgarte stabilization postcondition).
        monitor.observe(y, t, params)
        return dydt

    return ode_rhs


def run_simulation(
    params: GolferParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "RK45",
    config: SimulationConfig | None = None,
    **overrides: Any,
) -> GolferSimulationResult:
    """Integrate the constrained golfer equations of motion.

    Parameters
    ----------
    params : GolferParams
    initial_state : np.ndarray, shape (16,)
    t_end : float
    torque_func : callable
    dt : float - output time step
    method : str - ODE solver method
    config : SimulationConfig, optional
        Shared optional settings: ``rtol``/``atol`` (solver tolerances),
        ``alpha``/``beta`` (Baumgarte stabilization gains),
        ``torque_limits`` (shape (7,) per-joint saturation; ``clamp`` is the
        legacy alias used by SimulationPanel) and ``limits``
        (``JointLimitsNDOF`` joint angle limits with penalty stiffness).
        The same names may be passed as keyword arguments.

    Returns
    -------
    GolferSimulationResult
    """
    require_shape("Initial state", initial_state, (2 * N_DOF,))
    require_finite_array("Initial state", initial_state)
    require_positive("t_end", t_end)
    require_dt(dt, t_end)

    cfg = SimulationConfig.resolve(config, overrides, _GOLFER_SETTINGS)
    rtol, atol = cfg.solver_tolerances(1e-6, 1e-8)

    # Project initial conditions onto constraint manifold
    q0 = project_to_constraints(initial_state[:N_DOF], params)
    qdot0 = project_velocity(q0, initial_state[N_DOF:], params)
    y0 = np.concatenate([q0, qdot0])

    monitor = _ConstraintDriftMonitor()
    ode_rhs = _make_golfer_rhs(params, torque_func, cfg, monitor)

    t, states = integrate_ode(
        ode_rhs,
        y0,
        t_end,
        dt=dt,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    result = GolferSimulationResult(
        t=t,
        states=states,
        params=params,
        torque_func=torque_func,
    )

    require(result.n_steps >= 2, "Simulation must produce at least 2 time points")
    monitor.report()
    return result
