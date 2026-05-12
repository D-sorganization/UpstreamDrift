"""
Simulation engine for the driven triple pendulum.

Integrates the equations of motion using scipy's solve_ivp and
stores results in a structured TripleSimulationResult for easy access
by the GUI and analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .physics import JointLimitsNDOF

from .physics_triple import (
    State,
    TorqueFunc,
    TriplePendulumParams,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    kinetic_energy,
    mass_matrix_components,
    net_joint_forces,
    potential_energy,
    total_energy,
)
from .simulation_core import integrate_ode
from .simulation_result_base import TrajectoryResultMixin

# Re-export from shared utility for backwards compatibility (DRY â€” #1041)
from .torque_utils import make_polynomial_torque  # noqa: F401

# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class TripleSimulationResult(TrajectoryResultMixin):
    """Stores the complete trajectory and derived quantities."""

    t: np.ndarray
    states: np.ndarray
    params: TriplePendulumParams
    torque_func: TorqueFunc

    def __post_init__(self) -> None:
        self._validate_trajectory(expected_state_width=6)

    def mass_matrix_at(self, idx: int) -> dict:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        s = self.states[idx]
        return mass_matrix_components(s[1], s[2], self.params)

    def positions_at(self, idx: int) -> dict:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        s = self.states[idx]
        return forward_kinematics(s[0], s[1], s[2], self.params)

    def torques_at(self, idx: int) -> tuple[float, float, float]:
        self._check_idx(idx)
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        state_dot = equations_of_motion(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )
        return state_dot[3:]

    def joint_forces_at(self, idx: int) -> dict:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(self.states[idx], qddot, self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        s = self.states[idx]
        return coriolis_vector(s[1], s[2], s[3], s[4], s[5], self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        s = self.states[idx]
        return gravity_vector(s[0], s[1], s[2], self.params)

    def energy_at(self, idx: int) -> dict:
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        state = self.states[idx]
        result = {
            "kinetic": kinetic_energy(state, self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }
        self._assert_energy_finite(result, idx)
        return result

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Get dissipative friction torque vector at time idx.

        Returns
        -------
        np.ndarray, shape (3,)  [NÂ·m]
        """
        assert idx is not None, "idx must be provided"
        self._check_idx(idx)
        s = self.states[idx]
        return friction_torque_vector(s[3], s[4], s[5], self.params)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: TriplePendulumParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "DOP853",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    torque_limits: np.ndarray | None = None,
    limits: JointLimitsNDOF | None = None,
    clamp: np.ndarray | None = None,
) -> TripleSimulationResult:
    """Integrate the triple pendulum equations of motion.

    Parameters
    ----------
    torque_limits : np.ndarray, shape (3,), optional
        Per-joint torque saturation limits (#1150).
        Use ``np.inf`` for unclamped joints.

    Performance notes
    -----------------
    - ``DOP853`` is an 8th-order Runge-Kutta method that is fast for
      non-stiff problems while being accurate enough for chaotic systems.
    - ``max_step`` is intentionally NOT set here â€” letting the solver
      choose an adaptive step is ~10-50x faster than forcing ``dt`` as
      the maximum step size.
    - The output is resampled to a uniform ``t_eval`` grid at spacing ``dt``
      after integration, giving the GUI a predictable frame count without
      slowing down the solver.
    - Tolerances: ``rtol=1e-6`` / ``atol=1e-8`` are appropriate for
      visualisation-quality results.  Use tighter values only when
      quantitative energy conservation is required.
    """
    assert initial_state.shape == (
        6,
    ), f"Initial state shape must be (6,), got {initial_state.shape}"
    assert all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    # Merge clamp kwarg (from SimulationPanel) with torque_limits (legacy)
    effective_torque_limits = torque_limits if torque_limits is not None else clamp

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        assert t is not None, "t must be provided"
        dydt = equations_of_motion(y, t, params, torque_func, effective_torque_limits)
        # Apply joint limit penalty torques if enabled (#1151)
        if limits is not None:
            from .physics import joint_limit_torque_ndof

            q = y[:3]
            qdot = y[3:]
            tau_limit = joint_limit_torque_ndof(q, qdot, limits)
            # Re-solve with added penalty (tau_limit enters as extra torque)
            # We add the limit torques to qddot via M^-1 * tau_limit
            from .physics_triple import mass_matrix

            M = mass_matrix(y[1], y[2], params)
            try:
                qddot_correction = np.linalg.solve(M, tau_limit)
            except np.linalg.LinAlgError:
                qddot_correction = np.linalg.lstsq(M, tau_limit, rcond=None)[0]
            dydt[3:] += qddot_correction
        return dydt

    t, states = integrate_ode(
        ode_rhs,
        initial_state,
        t_end,
        dt=dt,
        method=method,
        rtol=rtol,
        atol=atol,
        # max_step deliberately omitted â€” adaptive step is much faster
    )

    result = TripleSimulationResult(
        t=t,
        states=states,
        params=params,
        torque_func=torque_func,
    )

    assert result.n_steps >= 2, "Simulation must produce at least 2 time points"
    return result
