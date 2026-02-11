"""Ball Rolling Physics for Putting Simulation.

This module implements the physics of a golf ball rolling on a putting surface,
including the transition from sliding to rolling, spin effects, and surface
interaction.

Physics Model:
    1. Initial impact creates sliding (spin ≠ v/r)
    2. Friction converts sliding to pure rolling
    3. Rolling friction decelerates ball
    4. Slopes add gravitational acceleration
    5. Ball eventually stops

Design by Contract:
    - All velocities in m/s
    - Spin in rad/s
    - Positions in meters
    - Time steps should be small (<= 0.01s) for accuracy

References:
    - Cross, R. (2006). Physics of Ball Rolling. American Journal of Physics.
    - Penner, A.R. (2002). The Physics of Putting. Canadian Journal of Physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)


class RollMode(Enum):
    """Rolling mode of the ball."""

    SLIDING = "sliding"  # Ball sliding on surface (spin ≠ v/r)
    ROLLING = "rolling"  # Pure rolling (spin = v/r)
    STOPPED = "stopped"  # Ball at rest


@dataclass
class BallState:
    """Current state of the ball.

    Attributes:
        position: 2D position on green [m, m]
        velocity: 2D velocity [m/s, m/s]
        spin: 3D angular velocity [rad/s] (x=topspin, y=sidespin axis, z=sidespin)
    """

    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        """Ensure arrays are numpy."""
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)
        self.spin = np.array(self.spin, dtype=np.float64)

    @property
    def speed(self) -> float:
        """Ball speed magnitude."""
        return float(np.linalg.norm(self.velocity))

    @property
    def is_moving(self) -> bool:
        """Check if ball is moving (above threshold)."""
        return self.speed > 0.005  # 5mm/s threshold

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector of velocity."""
        if self.speed < 1e-10:
            return np.zeros(2)
        return self.velocity / self.speed

    def copy(self) -> BallState:
        """Create independent copy."""
        return BallState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            spin=self.spin.copy(),
        )


class BallRollPhysics:
    """Physics engine for ball rolling on putting surface.

    Implements realistic ball dynamics including:
    - Sliding-to-rolling transition
    - Spin decay
    - Surface friction
    - Slope effects
    - Grain effects

    Attributes:
        ball_mass: Ball mass [kg]
        ball_radius: Ball radius [m]
        turf: Turf properties
        green: Optional green surface for slopes
    """

    # Sliding friction is typically 1.5-2x rolling friction
    SLIDING_FRICTION_MULTIPLIER = 1.8

    # Velocity threshold for stopping
    STOP_VELOCITY_THRESHOLD = 0.005  # m/s

    # Spin threshold for pure rolling determination
    SPIN_VELOCITY_RATIO_TOLERANCE = 0.05

    def __init__(
        self,
        turf: TurfProperties | None = None,
        green: GreenSurface | None = None,
        ball_mass: float = GOLF_BALL_MASS_KG,
        ball_radius: float = GOLF_BALL_RADIUS_M,
        integrator: str = "euler",
    ) -> None:
        """Initialize ball physics.

        Args:
            turf: Turf properties (uses green's turf if green provided)
            green: Full green surface (optional)
            ball_mass: Ball mass [kg]
            ball_radius: Ball radius [m]
            integrator: Integration method ("euler", "rk4", "verlet")
        """
        self.green = green
        self.turf = turf or (green.turf if green else TurfProperties())
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.integrator = integrator

        # Ball moment of inertia (solid sphere)
        self._moment_of_inertia = (2.0 / 5.0) * ball_mass * ball_radius**2

        # Previous acceleration for Verlet integration
        self._prev_acceleration: np.ndarray | None = None

    def determine_roll_mode(self, state: BallState) -> RollMode:
        """Determine current rolling mode from state.

        Pure rolling occurs when the contact point has zero velocity,
        which means: v = ω × r, or for our 2D case: v = -ω_y * r

        Args:
            state: Current ball state

        Returns:
            Current RollMode
        """
        speed = state.speed

        if speed < self.STOP_VELOCITY_THRESHOLD:
            return RollMode.STOPPED

        # At very low speeds, numerical errors in spin-velocity ratio
        # dominate; treat as rolling to avoid slip-friction feedback loop
        if speed < 0.05:
            return RollMode.ROLLING

        # For pure rolling: spin_y (about axis perpendicular to velocity) = v / r
        # The spin_y should be negative for forward roll (right-hand rule)
        expected_spin = -speed / self.ball_radius

        # Get spin component about axis perpendicular to velocity
        # This is the y-component of spin when velocity is in x-direction
        # For general direction, we need to project
        if speed > 1e-10:
            v_dir = state.velocity / speed
            # Spin axis for pure rolling is perpendicular to velocity and surface normal
            # For 2D ground: spin_axis = [-v_dir[1], v_dir[0], 0]
            # Component of spin about this axis:
            spin_axis = np.array([-v_dir[1], v_dir[0], 0])
            rolling_spin = np.dot(state.spin, spin_axis)

            # Check if close to pure rolling
            spin_error = abs(rolling_spin - expected_spin) / (
                abs(expected_spin) + 1e-10
            )

            if spin_error < self.SPIN_VELOCITY_RATIO_TOLERANCE:
                return RollMode.ROLLING

        return RollMode.SLIDING

    def compute_rolling_friction(self, state: BallState) -> np.ndarray:
        """Compute rolling friction force.

        Args:
            state: Current ball state

        Returns:
            Friction force vector [N]
        """
        if state.speed < 1e-10:
            return np.zeros(2)

        # Base friction from turf
        mu = self.turf.effective_friction

        # Apply grain effect
        grain_effect = self.turf.compute_grain_effect(state.direction)
        effective_mu = mu * (1.0 + grain_effect)

        # Friction force = μ * m * g (opposes motion)
        friction_mag = effective_mu * self.ball_mass * GRAVITY_M_S2
        friction_dir = -state.direction

        return friction_mag * friction_dir

    def compute_sliding_friction(self, state: BallState) -> np.ndarray:
        """Compute sliding friction force.

        During sliding, friction is higher and acts to both slow the ball
        and bring it to pure rolling.

        Args:
            state: Current ball state

        Returns:
            Friction force vector [N]
        """
        if state.speed < 1e-10:
            return np.zeros(2)

        # Sliding friction is higher than rolling
        mu = self.turf.effective_friction * self.SLIDING_FRICTION_MULTIPLIER

        # Friction opposes the slip velocity (contact point velocity)
        # For sliding: slip = v - ω × r
        # In 2D: slip_x = v_x + ω_y * r, slip_y = v_y - ω_x * r
        slip_velocity = np.array(
            [
                state.velocity[0] + state.spin[1] * self.ball_radius,
                state.velocity[1] - state.spin[0] * self.ball_radius,
            ]
        )

        slip_speed = np.linalg.norm(slip_velocity)
        if slip_speed < 1e-10:
            return self.compute_rolling_friction(state)

        slip_dir = slip_velocity / slip_speed

        # Friction force opposes slip
        friction_mag = mu * self.ball_mass * GRAVITY_M_S2
        return -friction_mag * slip_dir

    def compute_slope_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Compute acceleration from slope at position.

        Args:
            position: Ball position [m, m]

        Returns:
            Acceleration vector [m/s²]
        """
        if self.green is None:
            return np.zeros(2)

        return self.green.get_gravitational_acceleration(position)

    def compute_spin_decay(
        self, state: BallState, dt: float, mode: RollMode
    ) -> np.ndarray:
        """Compute spin decay over time step.

        During sliding, spin changes rapidly to approach pure rolling.
        During rolling, spin decays slowly with velocity.

        Args:
            state: Current ball state
            dt: Time step [s]
            mode: Current roll mode

        Returns:
            New spin vector [rad/s]
        """
        speed = state.speed

        if mode == RollMode.STOPPED:
            return np.zeros(3)

        if mode == RollMode.ROLLING:
            # Spin is locked to velocity in pure rolling
            # ω = -v / r (negative for forward roll)
            if speed > 1e-10:
                v_dir = state.velocity / speed
                spin_axis = np.array([-v_dir[1], v_dir[0], 0])
                rolling_spin_mag = speed / self.ball_radius
                return -spin_axis * rolling_spin_mag
            return np.zeros(3)

        # Sliding mode: spin decays toward pure rolling condition
        # The friction torque changes spin
        v_dir = state.velocity / (speed + 1e-10)

        # Target spin for pure rolling
        spin_axis = np.array([-v_dir[1], v_dir[0], 0])
        target_spin = -spin_axis * (speed / self.ball_radius)

        # Exponential approach to target (with friction-dependent rate)
        decay_rate = (
            self.turf.effective_friction * GRAVITY_M_S2 / self.ball_radius * 5.0
        )
        alpha = 1.0 - np.exp(-decay_rate * dt)

        new_spin = state.spin + alpha * (target_spin - state.spin)

        # Also decay sidespin (z-component)
        sidespin_decay = 0.9 ** (dt / 0.1)  # Decay 10% per 0.1s
        new_spin[2] *= sidespin_decay

        return new_spin

    def compute_total_acceleration(self, state: BallState) -> np.ndarray:
        """Compute total acceleration on ball.

        Args:
            state: Current ball state

        Returns:
            Acceleration vector [m/s²]
        """
        mode = self.determine_roll_mode(state)

        if mode == RollMode.STOPPED:
            # Check if on slope (could start moving)
            slope_accel = self.compute_slope_acceleration(state.position)
            return slope_accel

        # Friction force
        if mode == RollMode.ROLLING:
            friction = self.compute_rolling_friction(state)
        else:
            friction = self.compute_sliding_friction(state)

        # Slope acceleration
        slope_accel = self.compute_slope_acceleration(state.position)

        # Total acceleration
        friction_accel = friction / self.ball_mass

        return friction_accel + slope_accel

    def compute_kinetic_energy(self, state: BallState) -> float:
        """Compute total kinetic energy (translational + rotational).

        Args:
            state: Ball state

        Returns:
            Total kinetic energy [J]
        """
        # Translational: 0.5 * m * v²
        translational = 0.5 * self.ball_mass * state.speed**2

        # Rotational: 0.5 * I * ω²
        spin_mag = np.linalg.norm(state.spin)
        rotational = 0.5 * self._moment_of_inertia * spin_mag**2

        return float(translational + rotational)

    def step(self, state: BallState, dt: float) -> BallState:
        """Advance ball state by one time step.

        Args:
            state: Current ball state
            dt: Time step [s]

        Returns:
            New ball state
        """
        if self.integrator == "rk4":
            return self._step_rk4(state, dt)
        elif self.integrator == "verlet":
            return self._step_verlet(state, dt)
        else:
            return self._step_euler(state, dt)

    def _step_euler(self, state: BallState, dt: float) -> BallState:
        """Euler integration step."""
        mode = self.determine_roll_mode(state)

        if mode == RollMode.STOPPED:
            # Check if slope would cause movement
            accel = self.compute_slope_acceleration(state.position)
            if np.linalg.norm(accel) < 0.01:  # Threshold for starting
                return state.copy()

        # Compute acceleration
        accel = self.compute_total_acceleration(state)

        # Update velocity
        new_velocity = state.velocity + accel * dt

        # Apply grain curve effect
        new_velocity = self.turf.apply_grain_to_velocity(new_velocity)

        # Apply sidespin curve effect (Magnus-like on ground)
        if abs(state.spin[2]) > 1.0:  # Significant sidespin
            # Sidespin causes lateral acceleration
            spin_curve_accel = state.spin[2] * 0.001  # Coefficient
            perp_dir = np.array([-state.direction[1], state.direction[0]])
            new_velocity += perp_dir * spin_curve_accel * dt

        # Check for stopping
        new_speed = np.linalg.norm(new_velocity)
        if new_speed < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update position
        new_position = state.position + new_velocity * dt

        # Update spin
        new_spin = self.compute_spin_decay(state, dt, mode)

        return BallState(
            position=new_position,
            velocity=new_velocity,
            spin=new_spin,
        )

    def _step_rk4(self, state: BallState, dt: float) -> BallState:
        """4th-order Runge-Kutta integration."""

        def derivatives(
            pos: np.ndarray, vel: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            temp_state = BallState(pos, vel, state.spin)
            accel = self.compute_total_acceleration(temp_state)
            return vel, accel

        pos, vel = state.position, state.velocity

        # RK4 stages
        k1_v, k1_a = derivatives(pos, vel)
        k2_v, k2_a = derivatives(pos + 0.5 * dt * k1_v, vel + 0.5 * dt * k1_a)
        k3_v, k3_a = derivatives(pos + 0.5 * dt * k2_v, vel + 0.5 * dt * k2_a)
        k4_v, k4_a = derivatives(pos + dt * k3_v, vel + dt * k3_a)

        # Weighted average
        new_position = pos + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        new_velocity = vel + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)

        # Check stopping
        if np.linalg.norm(new_velocity) < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update spin
        mode = self.determine_roll_mode(state)
        new_spin = self.compute_spin_decay(state, dt, mode)

        return BallState(position=new_position, velocity=new_velocity, spin=new_spin)

    def _step_verlet(self, state: BallState, dt: float) -> BallState:
        """Velocity Verlet integration (better energy conservation)."""
        # Current acceleration
        accel = self.compute_total_acceleration(state)

        # Update position
        if self._prev_acceleration is None:
            self._prev_acceleration = accel

        new_position = state.position + state.velocity * dt + 0.5 * accel * dt**2

        # Compute new acceleration at new position
        temp_state = BallState(new_position, state.velocity, state.spin)
        new_accel = self.compute_total_acceleration(temp_state)

        # Update velocity
        new_velocity = state.velocity + 0.5 * (accel + new_accel) * dt

        # Check stopping
        if np.linalg.norm(new_velocity) < self.STOP_VELOCITY_THRESHOLD:
            new_velocity = np.zeros(2)

        # Update spin
        mode = self.determine_roll_mode(state)
        new_spin = self.compute_spin_decay(state, dt, mode)

        self._prev_acceleration = new_accel

        return BallState(position=new_position, velocity=new_velocity, spin=new_spin)

    def simulate_putt(
        self,
        initial_state: BallState,
        max_time: float = 30.0,
        dt: float = 0.001,
    ) -> dict[str, Any]:
        """Simulate complete putt trajectory.

        Args:
            initial_state: Initial ball state
            max_time: Maximum simulation time [s]
            dt: Time step [s]

        Returns:
            Dictionary with trajectory data
        """
        positions = [initial_state.position.copy()]
        velocities = [initial_state.velocity.copy()]
        spins = [initial_state.spin.copy()]
        times = [0.0]
        modes = [self.determine_roll_mode(initial_state)]

        state = initial_state.copy()
        t = 0.0
        holed = False

        while t < max_time and state.is_moving:
            state = self.step(state, dt)
            t += dt

            positions.append(state.position.copy())
            velocities.append(state.velocity.copy())
            spins.append(state.spin.copy())
            times.append(t)
            modes.append(self.determine_roll_mode(state))

            # Check for hole
            if self.green is not None:
                if self.green.is_in_hole(state.position, state.velocity):
                    holed = True
                    break

                # Check if off green
                if not self.green.is_on_green(state.position):
                    break

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "spins": np.array(spins),
            "times": np.array(times),
            "modes": modes,
            "holed": holed,
            "final_position": state.position.copy(),
            "final_velocity": state.velocity.copy(),
        }
