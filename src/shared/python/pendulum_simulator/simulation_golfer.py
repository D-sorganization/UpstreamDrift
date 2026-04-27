"""
Simulation engine for the golfer upper-body model.

Integrates the constrained equations of motion and stores results
in a GolferSimulationResult for GUI and analysis access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    total_energy,
)
from .simulation_core import integrate_ode
from .simulation_result_base import TrajectoryResultMixin

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
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return self.states[idx, :N_DOF]

    def qdot_at(self, idx: int) -> np.ndarray:
        """Generalized velocities at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return self.states[idx, N_DOF:]

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        """8Ã—8 mass matrix at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return mass_matrix(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def positions_at(self, idx: int) -> dict:
        """Forward kinematics at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return forward_kinematics(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def torques_at(
        self, idx: int
    ) -> tuple[float, float, float, float, float, float, float]:
        """Applied driving torques at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        """Joint accelerations at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return constrained_accelerations(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def joint_forces_at(self, idx: int) -> dict:
        """Net joint forces at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        q = self.q_at(idx)
        qdot = self.qdot_at(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(q, qdot, qddot, self.params)  # type: ignore[no-any-return]

    def constraint_forces_at(self, idx: int) -> np.ndarray:
        """Lagrange multiplier (constraint) forces at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return constraint_forces(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def constraint_violation_at(self, idx: int) -> float:
        """Constraint violation magnitude at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return constraint_violation(self.states[idx], self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        """Coriolis/centrifugal torques at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return coriolis_matrix(self.q_at(idx), self.qdot_at(idx), self.params)  # type: ignore[no-any-return]

    def gravity_at(self, idx: int) -> np.ndarray:
        """Gravitational torques at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return gravity_vector(self.q_at(idx), self.params)  # type: ignore[no-any-return]

    def energy_at(self, idx: int) -> dict:
        """Energy decomposition at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        state = self.states[idx]
        result = {
            "kinetic": kinetic_energy(state[:N_DOF], state[N_DOF:], self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }
        self._assert_energy_finite(result, idx)
        return result

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Friction torques at time index."""
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        return friction_torque_vector(self.qdot_at(idx), self.params)  # type: ignore[no-any-return]

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index.

        Overrides the base to spread the 7-joint torque_func output over the
        full N_DOF=8 vector before adding friction.
        """
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        tau_drive = np.zeros(N_DOF)
        tau_drive[:7] = self.torque_func(self.t[idx])
        tau_friction = self.friction_torques_at(idx)
        result: np.ndarray = tau_drive + tau_friction
        return result


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: GolferParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    alpha: float = 5.0,
    beta: float = 5.0,
    torque_limits: np.ndarray | None = None,
    limits: JointLimitsNDOF | None = None,
    clamp: np.ndarray | None = None,
) -> GolferSimulationResult:
    """Integrate the constrained golfer equations of motion.

    Parameters
    ----------
    params : GolferParams
    initial_state : np.ndarray, shape (16,)
    t_end : float
    torque_func : callable
    dt : float â€” output time step
    method : str â€” ODE solver method
    rtol, atol : float â€” solver tolerances
    alpha, beta : float â€” Baumgarte stabilization gains
    torque_limits : np.ndarray, shape (7,), optional
        Per-joint torque saturation limits.
    limits : JointLimitsNDOF, optional
        Joint angle limits with penalty stiffness.

    Returns
    -------
    GolferSimulationResult
    """
    assert initial_state.shape == (
        2 * N_DOF,
    ), f"Initial state shape must be ({2 * N_DOF},), got {initial_state.shape}"
    assert np.all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    # Merge clamp kwarg (from SimulationPanel) with torque_limits (legacy)
    effective_torque_limits = torque_limits if torque_limits is not None else clamp

    # Project initial conditions onto constraint manifold
    q0 = project_to_constraints(initial_state[:N_DOF], params)
    qdot0 = project_velocity(q0, initial_state[N_DOF:], params)
    y0 = np.concatenate([q0, qdot0])

    # Track constraint drift for postcondition check.
    _max_violation: list[float] = [0.0]

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        assert t is not None, "t must be provided"
        dydt = equations_of_motion(
            y, t, params, torque_func, alpha, beta, effective_torque_limits
        )
        # Apply joint limit penalty torques if enabled
        if limits is not None:
            from .physics import joint_limit_torque_ndof

            q = y[:N_DOF]
            qdot = y[N_DOF:]
            # Only apply limits to the 7 actuated DOFs (not club theta)
            q_actuated = q[:7]
            qdot_actuated = qdot[:7]
            tau_limit = joint_limit_torque_ndof(q_actuated, qdot_actuated, limits)
            # Add penalty via M^-1 * tau_limit (full 8-DOF mass matrix)
            M = mass_matrix(q, params)
            tau_full = np.zeros(N_DOF)
            tau_full[:7] = tau_limit
            try:
                qddot_correction = np.linalg.solve(M, tau_full)
            except np.linalg.LinAlgError:
                qddot_correction = np.linalg.lstsq(M, tau_full, rcond=None)[0]
            dydt[N_DOF:] += qddot_correction

        # Monitor constraint drift (Baumgarte stabilization postcondition).
        viol = constraint_violation(y, params)
        if viol > _max_violation[0]:
            _max_violation[0] = viol
        if viol > _CONSTRAINT_WARN_TOL:
            _log.warning(
                "Constraint violation %.3e at t=%.4f (warn threshold=%.3e)",
                viol,
                t,
                _CONSTRAINT_WARN_TOL,
            )
        return dydt

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

    assert result.n_steps >= 2, "Simulation must produce at least 2 time points"

    # Postcondition: constraint drift must remain bounded.
    max_viol = _max_violation[0]
    if max_viol > _CONSTRAINT_ABORT_TOL:
        _log.error(
            "Excessive constraint drift: max violation=%.3e > abort threshold=%.3e",
            max_viol,
            _CONSTRAINT_ABORT_TOL,
        )
    elif max_viol > _CONSTRAINT_WARN_TOL:
        _log.warning("Max constraint violation=%.3e during simulation", max_viol)

    return result
