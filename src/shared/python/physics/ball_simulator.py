"""Core ball flight simulator using the Rust kernel for RK4 integration.

This submodule contains BallFlightSimulator, the base simulator that delegates
trajectory integration to the upstream_physics Rust wheel. Extracted from
ball_flight_physics.py as part of P1 sprint decomposition (issue #2486).

.. deprecated::
    The RK4 integration loop has a Rust kernel equivalent in ``upstream_physics``
    (via ``rust_kernel.create_integrator_config``). New simulation code should use
    the Rust-backed integrator for native performance and WASM parity with the
    React frontend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.core.contracts import invariant, postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)
from src.shared.python.physics.ball_properties import (
    MIN_SPEED_THRESHOLD,
    NUMERICAL_EPSILON,
    BallProperties,
)
from src.shared.python.physics.ball_trajectory_analysis import TrajectoryAnalysisMixin

logger = get_logger(__name__)


@invariant(lambda self: self.ball.mass > 0, "Ball mass must be positive")
@invariant(lambda self: self.environment.gravity > 0, "Gravity must be positive")
class BallFlightSimulator(TrajectoryAnalysisMixin):
    """Refactored Ball Flight Simulator (Orthogonality-focused)."""

    def __init__(
        self,
        ball: BallProperties | None = None,
        env: EnvironmentalConditions | None = None,
        environment: EnvironmentalConditions | None = None,
    ) -> None:
        self.ball = ball or BallProperties()
        self.environment = env or environment or EnvironmentalConditions()

    @precondition(
        lambda self, launch, max_time=10.0, dt=0.01: (
            launch is not None and launch.velocity >= 0
        ),
        "Launch conditions must not be None and velocity must be non-negative",
    )
    @precondition(
        lambda self, launch, max_time=10.0, dt=0.01: max_time > 0 and dt > 0,
        "Max time and time step must be positive",
    )
    @postcondition(
        lambda result: result is not None and isinstance(result, list),
        "Trajectory must be returned as a non-None list",
    )
    def simulate_trajectory(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> list[TrajectoryPoint]:
        """Simulate trajectory using Rust kernel (preferred) or JIT-optimized RK4.

        When the upstream_physics Rust wheel is installed, the RK4 integration
        is delegated to the native Rust implementation for performance.
        Otherwise, falls back to the Python/Numba implementation.
        """
        if launch is None:
            raise ValueError("launch must be provided")
        from src.shared.python.physics.rust_kernel import is_rust_available

        if not is_rust_available():
            raise RuntimeError(
                "upstream-physics Rust kernel not found! Strict Rust Parity Enforced."
            )

        import upstream_physics  # type: ignore[import-untyped]

        v0 = launch.velocity
        ca, sa = np.cos(launch.azimuth_angle), np.sin(launch.azimuth_angle)
        cv, sv = np.cos(launch.launch_angle), np.sin(launch.launch_angle)

        initial = np.array([0.0, 0.0, 0.0, v0 * cv * ca, v0 * cv * sa, v0 * sv])
        omega = launch.spin_rate * 2 * np.pi / 60

        config = upstream_physics.IntegratorConfig(
            dt=dt, max_steps=int(max_time / dt) + 1
        )
        ball_props = upstream_physics.AeroBallProperties(
            mass=self.ball.mass,
            radius=self.ball.radius,
            drag_coefficient=self.ball.cd0,
            spin_decay_rate=self.ball.spin_decay_rate,
        )

        air_props = upstream_physics.AirProperties(
            density=self.environment.air_density,
            viscosity=1.81e-5,
            temperature=self.environment.temperature,
            pressure=101325.0,
        )

        pos0 = [0.0, 0.0, 0.0]
        vel0 = [
            float(initial[3]),
            float(initial[4]),
            float(initial[5]),
        ]
        spin_axis = [
            float(launch.spin_axis[0]),
            float(launch.spin_axis[1]),
            float(launch.spin_axis[2]),
        ]
        gravity = [0.0, 0.0, float(-self.environment.gravity)]
        wind = [
            float(self.environment.wind_velocity[0]),
            float(self.environment.wind_velocity[1]),
            float(self.environment.wind_velocity[2]),
        ]
        logger.debug("Using Rust ball_flight trajectory (dt=%.4f)", dt)
        rust_result = upstream_physics.simulate_ball_trajectory_py(
            pos0,
            vel0,
            spin_axis,
            omega,
            gravity,
            wind,
            ball_props,
            air_props,
            config,
        )
        return self._post_process_rust(rust_result, launch)

    def _post_process_rust(
        self, rust_result: Any, launch: LaunchConditions
    ) -> list[TrajectoryPoint]:
        """Convert a Rust BallTrajectoryResult to a list of TrajectoryPoint objects."""
        if launch is None:
            raise ValueError("launch must be provided")
        points = []
        for p in rust_result.get_points():
            pos = np.array([p.x, p.y, p.z])
            vel = np.array([p.vx, p.vy, p.vz])
            forces = self._calculate_forces(vel, launch)
            acc = (
                forces["gravity"] + forces["drag"] + forces["magnus"]
            ) / self.ball.mass
            points.append(TrajectoryPoint(p.t, pos, vel, acc, forces))
        return points

    @precondition(
        lambda self, trajectory: trajectory is not None,
        "Trajectory must not be None",
    )
    @postcondition(
        lambda result: (
            result is not None and "carry_distance" in result and "max_height" in result
        ),
        "Analysis must include carry_distance and max_height",
    )
    def analyze_trajectory(self, trajectory: list[TrajectoryPoint]) -> dict:
        """Generate comprehensive analysis dictionary."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        # Delegate to mixin implementation (adds DbC decorators on top)
        return super().analyze_trajectory(trajectory)

    def _calculate_forces(
        self, vel: np.ndarray, launch: LaunchConditions
    ) -> dict[str, np.ndarray]:
        """Calculate forces on the ball (supports vectorized input)."""
        if vel is None:
            raise ValueError("vel must be provided")
        is_batch = vel.ndim > 1
        omega = launch.spin_rate * 2 * np.pi / 60

        shape = vel.shape
        gravity = np.zeros(shape)
        gravity[2, ...] = -self.ball.mass * self.environment.gravity

        if is_batch:
            drag, magnus = self._calculate_forces_batch(vel, omega, launch.spin_axis)
        else:
            drag, magnus = self._calculate_forces_single(vel, omega, launch)

        return {"gravity": gravity, "drag": drag, "magnus": magnus}

    def _calculate_forces_batch(
        self,
        vel: np.ndarray,
        omega: float,
        spin_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized force calculation for batch velocity arrays (3, N)."""
        if vel is None:
            raise ValueError("vel must be provided")
        wind = (
            self.environment.wind_velocity.reshape(3, 1)
            if self.environment.wind_velocity.ndim == 1
            else self.environment.wind_velocity
        )
        rel_vel = vel - wind
        speed = np.sqrt(np.einsum("i...,i...->...", rel_vel, rel_vel))

        drag = np.zeros(vel.shape)
        magnus = np.zeros(vel.shape)

        mask = speed > MIN_SPEED_THRESHOLD
        if not np.any(mask):
            return drag, magnus

        valid_speed = speed[mask]
        valid_rel_vel = rel_vel[:, mask]
        s_ratio = (omega * self.ball.radius) / valid_speed
        aero_prefix = (
            0.5 * self.environment.air_density * self.ball.cross_sectional_area
        )

        # Drag
        cd = self.ball.cd0 + s_ratio * (self.ball.cd1 + s_ratio * self.ball.cd2)
        drag_force_mag = aero_prefix * cd * (valid_speed**2)
        drag[:, mask] = -drag_force_mag * (valid_rel_vel / valid_speed)

        # Magnus
        cl = self.ball.cl0 + s_ratio * (self.ball.cl1 + s_ratio * self.ball.cl2)
        magnus_force_mag = aero_prefix * cl * (valid_speed**2)

        axis = spin_axis.reshape(3, 1)
        cross = np.cross(axis, valid_rel_vel / valid_speed, axis=0)
        cross_norm = np.sqrt(np.einsum("i...,i...->...", cross, cross))
        cross_mask = cross_norm > NUMERICAL_EPSILON

        if np.any(cross_mask):
            factor = magnus_force_mag[cross_mask] / cross_norm[cross_mask]
            magnus[:, np.where(mask)[0][cross_mask]] = cross[:, cross_mask] * factor

        return drag, magnus

    def _calculate_forces_single(
        self,
        vel: np.ndarray,
        omega: float,
        launch: LaunchConditions,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scalar force calculation for a single velocity vector (3,)."""
        if vel is None:
            raise ValueError("vel must be provided")
        rel_vel = vel - self.environment.wind_velocity
        speed = float(np.linalg.norm(rel_vel))

        drag = np.zeros(vel.shape)
        magnus = np.zeros(vel.shape)

        if speed <= MIN_SPEED_THRESHOLD:
            return drag, magnus

        s_ratio = (omega * self.ball.radius) / speed
        cd = self.ball.calculate_cd(s_ratio)
        cl = self.ball.calculate_cl(s_ratio)
        aero_prefix = (
            0.5 * self.environment.air_density * self.ball.cross_sectional_area
        )

        drag = -(aero_prefix * cd * speed**2) * (rel_vel / speed)

        cross = np.cross(launch.spin_axis, rel_vel / speed)
        cross_norm = np.linalg.norm(cross)
        if cross_norm > NUMERICAL_EPSILON:
            magnus = (aero_prefix * cl * speed**2) * (cross / cross_norm)

        return drag, magnus
