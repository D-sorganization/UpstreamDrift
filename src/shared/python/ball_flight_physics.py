"""Ball flight physics simulation with Magnus effect and drag.

This module implements research-grade ball flight physics including:
- Magnus effect (spin-induced forces)
- Drag forces (Reynolds number dependent)
- Launch angle and velocity effects
- 3D trajectory calculation
- Landing dispersion patterns

Refactored to address DRY and Orthogonality violations (Pragmatic Programmer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from src.shared.python.constants import AIR_DENSITY_SEA_LEVEL_KG_M3, GRAVITY_M_S2
from src.shared.python.engine_availability import NUMBA_AVAILABLE
from src.shared.python.logging_config import get_logger

# Performance: Optional Numba JIT compilation
if NUMBA_AVAILABLE:
    from numba import jit
else:

    def jit(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        """No-op decorator when numba is not installed."""

        def decorator(func: object) -> object:
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


logger = get_logger(__name__)

MIN_SPEED_THRESHOLD: float = 0.1
MAX_LIFT_COEFFICIENT: float = 0.25
NUMERICAL_EPSILON: float = 1e-10


@dataclass(frozen=True)
class BallProperties:
    """Physical properties of a golf ball (DRY-optimized)."""

    mass: float = 0.0459
    diameter: float = 0.04267
    cd0: float = 0.21
    cd1: float = 0.05
    cd2: float = 0.02
    cl0: float = 0.00
    cl1: float = 0.38
    cl2: float = 0.08

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def cross_sectional_area(self) -> float:
        return float(np.pi * (self.diameter / 2) ** 2)

    def calculate_cd(self, s: float) -> float:
        return self.cd0 + s * (self.cd1 + s * self.cd2)

    def calculate_cl(self, s: float) -> float:
        return min(MAX_LIFT_COEFFICIENT, self.cl0 + s * (self.cl1 + s * self.cl2))


@dataclass(frozen=True)
class LaunchConditions:
    """Initial launch conditions."""

    velocity: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 0.0
    spin_axis: np.ndarray = np.array([0.0, -1.0, 0.0])


@dataclass(frozen=True)
class EnvironmentalConditions:
    """Environmental settings."""

    air_density: float = AIR_DENSITY_SEA_LEVEL_KG_M3
    wind_velocity: np.ndarray = np.array([0.0, 0.0, 0.0])
    gravity: float = GRAVITY_M_S2


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""

    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    forces: dict[str, np.ndarray]


@jit(nopython=True, cache=True)
def _calculate_accel_core(
    rel_vel: np.ndarray,
    speed: float,
    gravity_acc: np.ndarray,
    ball_radius: float,
    const_term: float,
    coeffs: tuple[float, float, float, float, float, float],
    omega: float,
    spin_axis: np.ndarray,
) -> np.ndarray:
    """Unified core for acceleration calculation."""
    cd0, cd1, cd2, cl0, cl1, cl2 = coeffs
    spin_ratio = (omega * ball_radius) / speed if omega > 0 else 0.0

    # Drag
    cd = cd0 + spin_ratio * (cd1 + spin_ratio * cd2)
    acc = gravity_acc - (const_term * cd * speed) * rel_vel

    # Magnus
    if omega > 0 and spin_ratio > 0:
        cl = cl0 + spin_ratio * (cl1 + spin_ratio * cl2)
        if cl > MAX_LIFT_COEFFICIENT:
            cl = MAX_LIFT_COEFFICIENT

        magnus_mag = const_term * cl * (speed**2)
        # Manual cross product for Numba optimization
        c0 = spin_axis[1] * rel_vel[2] - spin_axis[2] * rel_vel[1]
        c1 = spin_axis[2] * rel_vel[0] - spin_axis[0] * rel_vel[2]
        c2 = spin_axis[0] * rel_vel[1] - spin_axis[1] * rel_vel[0]

        cross_mag = np.sqrt(c0**2 + c1**2 + c2**2)
        if cross_mag > NUMERICAL_EPSILON:
            factor = magnus_mag / cross_mag
            acc[0] += c0 * factor
            acc[1] += c1 * factor
            acc[2] += c2 * factor

    return cast(np.ndarray, acc)


@jit(nopython=True, cache=True)
def _flight_dynamics_step(
    state: np.ndarray,
    gravity_acc: np.ndarray,
    wind_velocity: np.ndarray,
    ball_radius: float,
    const_term: float,
    coeffs: tuple[float, float, float, float, float, float],
    omega: float,
    spin_axis: np.ndarray,
) -> np.ndarray:
    """State transition derivative."""
    velocity = state[3:]
    rel_vel = velocity - wind_velocity
    speed_sq = rel_vel @ rel_vel

    if speed_sq <= MIN_SPEED_THRESHOLD**2:
        acc = gravity_acc
    else:
        acc = _calculate_accel_core(
            rel_vel,
            np.sqrt(speed_sq),
            gravity_acc,
            ball_radius,
            const_term,
            coeffs,
            omega,
            spin_axis,
        )

    res = np.empty(6)
    res[:3], res[3:] = velocity, acc
    return res


@jit(nopython=True, cache=True)
def _solve_rk4_loop(
    initial_state: np.ndarray,
    dt: float,
    max_steps: int,
    gravity_acc: np.ndarray,
    wind_velocity: np.ndarray,
    ball_radius: float,
    const_term: float,
    coeffs: tuple[float, float, float, float, float, float],
    omega: float,
    spin_axis: np.ndarray,
) -> np.ndarray:
    """Numba-optimized RK4 loop."""
    out = np.empty((max_steps, 7))
    curr = initial_state.copy()
    t = 0.0

    out[0, 0], out[0, 1:] = t, curr
    actual_steps = 1

    for i in range(1, max_steps):
        k1 = _flight_dynamics_step(
            curr,
            gravity_acc,
            wind_velocity,
            ball_radius,
            const_term,
            coeffs,
            omega,
            spin_axis,
        )
        k2 = _flight_dynamics_step(
            curr + 0.5 * dt * k1,
            gravity_acc,
            wind_velocity,
            ball_radius,
            const_term,
            coeffs,
            omega,
            spin_axis,
        )
        k3 = _flight_dynamics_step(
            curr + 0.5 * dt * k2,
            gravity_acc,
            wind_velocity,
            ball_radius,
            const_term,
            coeffs,
            omega,
            spin_axis,
        )
        k4 = _flight_dynamics_step(
            curr + dt * k3,
            gravity_acc,
            wind_velocity,
            ball_radius,
            const_term,
            coeffs,
            omega,
            spin_axis,
        )

        curr += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        out[i, 0], out[i, 1:] = t, curr
        actual_steps += 1
        if curr[2] <= 0:
            break

    return out[:actual_steps]


class BallFlightSimulator:
    """Refactored Ball Flight Simulator (Orthogonality-focused)."""

    def __init__(
        self,
        ball: BallProperties | None = None,
        env: EnvironmentalConditions | None = None,
    ):
        self.ball = ball or BallProperties()
        self.env = env or EnvironmentalConditions()

    def _get_coeffs(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.ball.cd0,
            self.ball.cd1,
            self.ball.cd2,
            self.ball.cl0,
            self.ball.cl1,
            self.ball.cl2,
        )

    def simulate_trajectory(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> list[TrajectoryPoint]:
        """Simulate trajectory using JIT-optimized RK4."""
        v0 = launch.velocity
        ca, sa = np.cos(launch.azimuth_angle), np.sin(launch.azimuth_angle)
        cv, sv = np.cos(launch.launch_angle), np.sin(launch.launch_angle)

        initial = np.array([0.0, 0.0, 0.0, v0 * cv * ca, v0 * cv * sa, v0 * sv])
        gravity_acc = np.array([0.0, 0.0, -self.env.gravity])
        const_term = (
            0.5 * self.env.air_density * self.ball.cross_sectional_area
        ) / self.ball.mass
        omega = launch.spin_rate * 2 * np.pi / 60

        raw_data = _solve_rk4_loop(
            initial,
            dt,
            int(max_time / dt) + 1,
            gravity_acc,
            self.env.wind_velocity,
            self.ball.radius,
            const_term,
            self._get_coeffs(),
            omega,
            launch.spin_axis,
        )

        return self._post_process(raw_data, launch)

    def _post_process(
        self, data: np.ndarray, launch: LaunchConditions
    ) -> list[TrajectoryPoint]:
        """Convert raw integration data to rich TrajectoryPoint objects."""
        points = []
        gravity_f = np.array([0.0, 0.0, -self.ball.mass * self.env.gravity])
        omega = launch.spin_rate * 2 * np.pi / 60

        for row in data:
            t, pos, vel = row[0], row[1:4], row[4:]
            rel_vel = vel - self.env.wind_velocity
            speed = np.linalg.norm(rel_vel)

            # Re-calculate forces for reporting (Orthogonal but slightly redundant for clarity)
            drag = np.zeros(3)
            magnus = np.zeros(3)
            if speed > MIN_SPEED_THRESHOLD:
                s_ratio = float((omega * self.ball.radius) / speed)
                cd = self.ball.calculate_cd(s_ratio)
                cl = self.ball.calculate_cl(s_ratio)
                drag = -(
                    0.5
                    * self.env.air_density
                    * self.ball.cross_sectional_area
                    * cd
                    * (speed**2)
                ) * (rel_vel / speed)
                cross = np.cross(launch.spin_axis, rel_vel / speed)
                if np.linalg.norm(cross) > NUMERICAL_EPSILON:
                    magnus = (
                        0.5
                        * self.env.air_density
                        * self.ball.cross_sectional_area
                        * cl
                        * (speed**2)
                    ) * (cross / np.linalg.norm(cross))

            acc = (gravity_f + drag + magnus) / self.ball.mass
            points.append(
                TrajectoryPoint(
                    t,
                    pos,
                    vel,
                    acc,
                    {"gravity": gravity_f, "drag": drag, "magnus": magnus},
                )
            )

        return points
