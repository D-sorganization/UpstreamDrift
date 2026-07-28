"""
Simulation engine for the driven double pendulum golf model.

Integrates the equations of motion using scipy's solve_ivp and
stores results in a structured SimulationResult for easy access
by the GUI and analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .native_backend import double_native_enabled, simulate_double
from .physics import (
    JointLimits,
    PendulumParams,
    State,
    TorqueClamp,
    TorqueFunc,
    base_force,
    clamp_torque,
    control_vector,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    joint_velocities,
    kinetic_energy,
    mass_matrix_components,
    net_joint_forces,
    potential_energy,
    total_energy,
)
from .simulation_core import integrate_ode
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

# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult(TrajectoryResultMixin):
    """Stores the complete trajectory and derived quantities."""

    t: np.ndarray
    states: np.ndarray
    params: PendulumParams
    torque_func: TorqueFunc
    limits: JointLimits | None = None
    clamp: TorqueClamp | None = None

    # Lazily computed caches
    _mass_matrices: np.ndarray | None = field(default=None, repr=False)
    _positions: list | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._validate_trajectory(expected_state_width=4)

    @property
    def theta1(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def phi(self) -> np.ndarray:
        return self.states[:, 1]

    @property
    def dtheta1(self) -> np.ndarray:
        return self.states[:, 2]

    @property
    def dphi(self) -> np.ndarray:
        return self.states[:, 3]

    def mass_matrix_at(self, idx: int) -> dict:
        self._check_idx(idx)
        return mass_matrix_components(self.states[idx, 1], self.params)

    def positions_at(self, idx: int) -> dict:
        self._check_idx(idx)
        return forward_kinematics(self.states[idx, 0], self.states[idx, 1], self.params)

    def torques_at(self, idx: int) -> tuple[float, float]:
        """Get applied torques at time index idx."""
        self._check_idx(idx)
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        self._check_idx(idx)
        state_dot = equations_of_motion(
            self.states[idx],
            self.t[idx],
            self.params,
            self.torque_func,
            self.limits,
            self.clamp,
        )
        return state_dot[2:]

    def joint_forces_at(self, idx: int) -> dict:
        self._check_idx(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(self.states[idx], qddot, self.params)

    def joint_velocities_at(self, idx: int) -> dict:
        """Get linear joint velocities at time index idx."""
        self._check_idx(idx)
        return joint_velocities(self.states[idx], self.params)

    def base_force_at(self, idx: int) -> dict:
        """Get base reaction force at time index idx."""
        self._check_idx(idx)
        qddot = self.accelerations_at(idx)
        return base_force(self.states[idx], qddot, self.params)

    def control_vector_at(self, idx: int) -> dict:
        """Get control vector at time index idx."""
        self._check_idx(idx)
        qddot = self.accelerations_at(idx)
        return control_vector(self.states[idx], qddot, self.params, self.limits)

    def energy_at(self, idx: int) -> dict:
        self._check_idx(idx)
        state = self.states[idx]
        result = {
            "kinetic": kinetic_energy(state, self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }
        self._assert_energy_finite(result, idx)
        return result

    def coriolis_at(self, idx: int) -> np.ndarray:
        self._check_idx(idx)
        s = self.states[idx]
        return coriolis_vector(s[1], s[2], s[3], self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        self._check_idx(idx)
        s = self.states[idx]
        return gravity_vector(s[0], s[1], self.params)

    def friction_torques_at(self, idx: int) -> np.ndarray:
        self._check_idx(idx)
        s = self.states[idx]
        return friction_torque_vector(s[2], s[3], self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        self._check_idx(idx)
        tau_drive = np.array(self.torque_func(self.t[idx]))
        if self.clamp is not None:
            tau_drive = clamp_torque(tau_drive, self.clamp)
        tau_friction = self.friction_torques_at(idx)
        return np.asarray(tau_drive + tau_friction)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: PendulumParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "RK45",
    limits: JointLimits | None = None,
    clamp: TorqueClamp | None = None,
    coeffs: list[float] | None = None,
    n_coeffs_per_joint: int | None = None,
) -> SimulationResult:
    """Integrate the double pendulum equations of motion.

    Pre: initial_state shape (4,), finite. t_end > 0. dt in (0, t_end).
    Post: result has >= 2 time points, all finite.
    """
    require_shape("Initial state", initial_state, (4,))
    require_finite_array("Initial state", initial_state)
    require_positive("t_end", t_end)
    require_dt(dt, t_end)

    t, states = None, None
    if (
        double_native_enabled()
        and coeffs is not None
        and n_coeffs_per_joint is not None
        and limits is None
        and clamp is None
    ):
        q0 = initial_state[:2].tolist()
        qdot0 = initial_state[2:4].tolist()
        t_span = (0.0, t_end)
        max_steps = int(max(t_end / dt * 10, 100000))
        res = simulate_double(
            params, q0, qdot0, coeffs, n_coeffs_per_joint, t_span, max_steps
        )
        if res is not None:
            t_res, states_res = res
            if len(t_res) >= 2:
                # Interpolate to the desired uniform grid if needed
                t_eval = np.arange(0.0, t_end, dt)
                from scipy.interpolate import interp1d

                interp = interp1d(
                    t_res,
                    states_res,
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                states = interp(t_eval)
                t = t_eval

    if t is None or states is None:

        def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
            return equations_of_motion(y, t, params, torque_func, limits, clamp)

        t, states = integrate_ode(
            ode_rhs,
            initial_state,
            t_end,
            dt=dt,
            method=method,
            rtol=1e-8,
            atol=1e-10,
            max_step=dt,
        )

    result = SimulationResult(
        t=t,
        states=states,
        params=params,
        torque_func=torque_func,
        limits=limits,
        clamp=clamp,
    )

    require(result.n_steps >= 2, "Simulation must produce at least 2 time points")
    return result
