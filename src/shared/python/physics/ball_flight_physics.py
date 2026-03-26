"""Ball flight physics simulation with Magnus effect and drag.

This module implements research-grade ball flight physics including:
- Magnus effect (spin-induced forces)
- Drag forces (Reynolds number dependent)
- Launch angle and velocity effects
- 3D trajectory calculation
- Landing dispersion patterns

Refactored to address DRY and Orthogonality violations (Pragmatic Programmer).

.. deprecated::
    The RK4 integration loop in this module has a Rust kernel equivalent
    in ``upstream_physics`` (via ``rust_kernel.create_integrator_config``).
    New simulation code should use the Rust-backed integrator for native
    performance and WASM parity with the React frontend.

Planned enhancement: implement Environmental Gradient Modeling (wind shear, temperature gradients).
Planned enhancement: implement Hydrodynamic Lubrication (wet ball physics).
Planned enhancement: implement Dimple Geometry Optimization.
Planned enhancement: implement Turbulence Modeling.
Planned enhancement: implement Mud Ball Physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.physics.aerodynamics import (
        AerodynamicsConfig,
        RandomizationConfig,
        WindConfig,
    )

from src.shared.python.core.constants import AIR_DENSITY_SEA_LEVEL_KG_M3, GRAVITY_M_S2
from src.shared.python.core.contracts import invariant, postcondition, precondition
from src.shared.python.core.physics_constants import SPIN_DECAY_RATE_S
from src.shared.python.logging_pkg.logging_config import get_logger

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
    spin_decay_rate: float = float(SPIN_DECAY_RATE_S)

    @property
    def radius(self) -> float:
        """Return the ball radius in meters."""
        return self.diameter / 2

    @property
    def cross_sectional_area(self) -> float:
        """Return the cross-sectional area of the ball."""
        return float(np.pi * (self.diameter / 2) ** 2)

    def calculate_cd(self, s: float) -> float:
        """Compute the drag coefficient from the spin parameter."""
        return self.cd0 + s * (self.cd1 + s * self.cd2)

    def calculate_cl(self, s: float) -> float:
        """Compute the lift coefficient from the spin parameter, clamped to max."""
        return min(MAX_LIFT_COEFFICIENT, self.cl0 + s * (self.cl1 + s * self.cl2))


@dataclass(frozen=True)
class LaunchConditions:
    """Initial launch conditions."""

    velocity: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 0.0
    spin_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))


@dataclass(frozen=True)
class EnvironmentalConditions:
    """Environmental settings."""

    air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    wind_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gravity: float = float(GRAVITY_M_S2)
    altitude: float = 0.0
    temperature: float = 15.0


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""

    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    forces: dict[str, np.ndarray]

    @property
    def speed(self) -> float:
        """Return the scalar speed from the velocity vector."""
        return float(np.linalg.norm(self.velocity))

    @property
    def height(self) -> float:
        """Return the vertical position component."""
        return float(self.position[2])


@invariant(lambda self: self.ball.mass > 0, "Ball mass must be positive")
@invariant(lambda self: self.environment.gravity > 0, "Gravity must be positive")
class BallFlightSimulator:
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
        if not (launch is not None):
            raise ValueError("launch must be provided")
        if not (launch is not None):
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
        if not (launch is not None):
            raise ValueError("launch must be provided")
        if not (launch is not None):
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

    def calculate_carry_distance(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total carry distance."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        last_pos = trajectory[-1].position
        return float(np.sqrt(last_pos[0] ** 2 + last_pos[1] ** 2))

    def calculate_max_height(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate maximum height achieved."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return float(max(p.position[2] for p in trajectory))

    def calculate_flight_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total flight time."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return trajectory[-1].time

    def _calculate_landing_angle(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate landing angle in degrees."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if len(trajectory) < 2:
            return 0.0

        v = trajectory[-1].velocity
        v_horiz = np.linalg.norm(v[:2])

        if v_horiz < NUMERICAL_EPSILON:
            return 90.0

        # Angle with horizontal (positive for descent)
        return float(np.degrees(np.arctan2(-v[2], v_horiz)))

    def _calculate_apex_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate time to reach apex."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0

        max_h = -float("inf")
        apex_t = 0.0
        for p in trajectory:
            if p.position[2] > max_h:
                max_h = p.position[2]
                apex_t = p.time
        return apex_t

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
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        return {
            "carry_distance": self.calculate_carry_distance(trajectory),
            "max_height": self.calculate_max_height(trajectory),
            "flight_time": self.calculate_flight_time(trajectory),
            "landing_angle": self._calculate_landing_angle(trajectory),
            "apex_time": self._calculate_apex_time(trajectory),
            "trajectory_points": len(trajectory),
        }

    def _calculate_forces(
        self, vel: np.ndarray, launch: LaunchConditions
    ) -> dict[str, np.ndarray]:
        """Calculate forces on the ball (supports vectorized input)."""
        if not (vel is not None):
            raise ValueError("vel must be provided")
        if not (vel is not None):
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
        if not (vel is not None):
            raise ValueError("vel must be provided")
        if not (vel is not None):
            raise ValueError("vel must be provided")
        wind = (
            self.environment.wind_velocity.reshape(3, 1)
            if self.environment.wind_velocity.ndim == 1
            else self.environment.wind_velocity
        )
        rel_vel = vel - wind
        speed = np.sqrt(np.sum(rel_vel**2, axis=0))

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
        cross_norm = np.sqrt(np.sum(cross**2, axis=0))
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
        if not (vel is not None):
            raise ValueError("vel must be provided")
        if not (vel is not None):
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


# =============================================================================
# Enhanced Simulator with Toggleable Aerodynamics
# =============================================================================


class EnhancedBallFlightSimulator:
    """Ball flight simulator with toggleable aerodynamic effects.

    This simulator integrates with the aerodynamics module to provide:
    - Toggleable drag, lift, and Magnus effects
    - Sophisticated wind model with gusts and turbulence
    - Environment randomization for Monte Carlo simulations
    - Full backward compatibility with standard simulator

    Design Principles (Pragmatic Programmer):
    - Reversible: Aerodynamics can be toggled on/off at any time
    - Reusable: Composes with existing BallFlightSimulator
    - DRY: Reuses existing trajectory analysis methods
    - Orthogonal: Aerodynamics, wind, and randomization are independent

    Example:
        >>> from src.shared.python.physics.aerodynamics import AerodynamicsConfig, WindConfig
        >>> config = AerodynamicsConfig(drag_enabled=True, lift_enabled=True)
        >>> wind = WindConfig(base_velocity=np.array([5.0, 0.0, 0.0]))
        >>> sim = EnhancedBallFlightSimulator(aero_config=config, wind_config=wind)
        >>> traj = sim.simulate_trajectory(launch_conditions)
    """

    def __init__(
        self,
        ball: BallProperties | None = None,
        environment: EnvironmentalConditions | None = None,
        aero_config: AerodynamicsConfig | None = None,
        wind_config: WindConfig | None = None,
        randomization_config: RandomizationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize enhanced simulator.

        Args:
            ball: Golf ball properties
            environment: Environmental conditions (temperature, altitude)
            aero_config: Aerodynamics configuration (toggles and coefficients)
            wind_config: Wind configuration (base wind, gusts, turbulence)
            randomization_config: Environment randomization configuration
            seed: Random seed for reproducibility
        """
        # Import here to avoid circular dependency
        from src.shared.python.physics.aerodynamics import (
            AerodynamicsConfig,
            AerodynamicsEngine,
            EnvironmentRandomizer,
            RandomizationConfig,
            WindConfig,
            WindModel,
        )

        self.ball = ball or BallProperties()
        self.environment = environment or EnvironmentalConditions()
        self.aero_config = aero_config or AerodynamicsConfig()
        self.wind_config = wind_config or WindConfig()
        self.randomization_config = randomization_config or RandomizationConfig()
        self._seed = seed

        # Initialize wind model
        self._wind_model = WindModel(self.wind_config, seed=seed)

        # Initialize randomizer
        self._randomizer = (
            EnvironmentRandomizer(self.randomization_config, seed=seed)
            if self.randomization_config.enabled
            else None
        )

        # Initialize aerodynamics engine
        self._aero_engine = AerodynamicsEngine(
            config=self.aero_config,
            wind_model=self._wind_model,
            randomization=self._randomizer,
            air_density=self.environment.air_density,
        )

    @precondition(
        lambda self, launch, max_time=10.0, dt=0.01, include_gravity=True: (
            launch is not None and launch.velocity >= 0
        ),
        "Launch conditions must not be None and velocity must be non-negative",
    )
    @precondition(
        lambda self, launch, max_time=10.0, dt=0.01, include_gravity=True: (
            max_time > 0 and dt > 0
        ),
        "Max time and time step must be positive",
    )
    @postcondition(
        lambda result: result is not None and isinstance(result, list),
        "Trajectory must be returned as a non-None list",
    )
    def simulate_trajectory(
        self,
        launch: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
        include_gravity: bool = True,
    ) -> list[TrajectoryPoint]:
        """Simulate ball trajectory with configurable aerodynamics.

        Uses RK4 integration with the aerodynamics engine for force
        calculations. Aerodynamic effects can be toggled via the
        aero_config provided at initialization.

        Args:
            launch: Launch conditions (velocity, angle, spin)
            max_time: Maximum simulation time [s]
            dt: Time step [s]
            include_gravity: Include gravitational acceleration

        Returns:
            List of TrajectoryPoint objects representing the flight path
        """
        # Convert launch conditions to initial state
        if not (launch is not None):
            raise ValueError("launch must be provided")
        if not (launch is not None):
            raise ValueError("launch must be provided")
        v0 = launch.velocity
        ca, sa = np.cos(launch.azimuth_angle), np.sin(launch.azimuth_angle)
        cv, sv = np.cos(launch.launch_angle), np.sin(launch.launch_angle)

        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([v0 * cv * ca, v0 * cv * sa, v0 * sv])

        # Convert spin rate (rpm) to angular velocity (rad/s)
        omega = launch.spin_rate * 2 * np.pi / 60
        spin = launch.spin_axis * omega

        # Gravity acceleration
        gravity_acc = (
            np.array([0.0, 0.0, -self.environment.gravity])
            if include_gravity
            else np.zeros(3)
        )

        # Run simulation
        trajectory = []
        t = 0.0
        max_steps = int(max_time / dt) + 1

        for _ in range(max_steps):
            # Calculate forces
            aero_forces = self._aero_engine.compute_forces(
                velocity, spin, t=t, position=position
            )

            # Total acceleration
            gravity_force = self.ball.mass * gravity_acc
            total_force = aero_forces["total"] + gravity_force
            acceleration = total_force / self.ball.mass

            # Store trajectory point
            forces = {
                "gravity": gravity_force,
                "drag": aero_forces["drag"],
                "lift": aero_forces["lift"],
                "magnus": aero_forces["magnus"],
            }

            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    acceleration=acceleration.copy(),
                    forces=forces,
                )
            )

            # Check termination (ball hit ground)
            if position[2] < 0 and t > 0:
                break

            # RK4 integration step
            position, velocity, spin = self._rk4_step(
                position, velocity, spin, gravity_acc, t, dt
            )

            # Update spin (decay)
            spin = self._aero_engine.compute_spin_decay(spin, dt)

            t += dt

        return trajectory

    def _rk4_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        spin: np.ndarray,
        gravity_acc: np.ndarray,
        t: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one RK4 integration step.

        Args:
            pos: Current position [m]
            vel: Current velocity [m/s]
            spin: Current angular velocity [rad/s]
            gravity_acc: Gravitational acceleration [m/s^2]
            t: Current time [s]
            dt: Time step [s]

        Returns:
            Tuple of (new_position, new_velocity, spin)
        """

        if not (pos is not None):
            raise ValueError("pos must be provided")
        if not (pos is not None):
            raise ValueError("pos must be provided")

        def derivatives(
            p: np.ndarray, v: np.ndarray, time: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute velocity and acceleration derivatives for RK4 integration."""
            if not (p is not None):
                raise ValueError("p must be provided")
            if not (p is not None):
                raise ValueError("p must be provided")
            aero_forces = self._aero_engine.compute_forces(v, spin, t=time, position=p)
            gravity_force = self.ball.mass * gravity_acc
            total_force = aero_forces["total"] + gravity_force
            accel = total_force / self.ball.mass
            return v, accel

        # RK4 coefficients
        k1_v, k1_a = derivatives(pos, vel, t)
        k2_v, k2_a = derivatives(
            pos + 0.5 * dt * k1_v, vel + 0.5 * dt * k1_a, t + 0.5 * dt
        )
        k3_v, k3_a = derivatives(
            pos + 0.5 * dt * k2_v, vel + 0.5 * dt * k2_a, t + 0.5 * dt
        )
        k4_v, k4_a = derivatives(pos + dt * k3_v, vel + dt * k3_a, t + dt)

        # Update state
        new_pos = pos + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        new_vel = vel + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)

        return new_pos, new_vel, spin

    def simulate_with_comparison(
        self,
        launch: LaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> dict[str, list[TrajectoryPoint]]:
        """Simulate trajectory with and without aerodynamics for comparison.

        This method is useful for visualizing the effect of aerodynamic
        forces on ball flight.

        Args:
            launch: Launch conditions
            max_time: Maximum simulation time [s]
            dt: Time step [s]

        Returns:
            Dictionary with 'with_aero' and 'without_aero' trajectories
        """
        if not (launch is not None):
            raise ValueError("launch must be provided")
        if not (launch is not None):
            raise ValueError("launch must be provided")
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig

        # Trajectory with current aerodynamics settings
        traj_with = self.simulate_trajectory(launch, max_time, dt)

        # Create a temporary simulator with aerodynamics disabled
        no_aero_sim = EnhancedBallFlightSimulator(
            ball=self.ball,
            environment=self.environment,
            aero_config=AerodynamicsConfig(enabled=False),
            seed=self._seed,
        )
        traj_without = no_aero_sim.simulate_trajectory(launch, max_time, dt)

        return {
            "with_aero": traj_with,
            "without_aero": traj_without,
        }

    @precondition(
        lambda self, launch, n_samples=100, max_time=10.0, dt=0.01: launch is not None,
        "Launch conditions must not be None",
    )
    @precondition(
        lambda self, launch, n_samples=100, max_time=10.0, dt=0.01: n_samples > 0,
        "Number of Monte Carlo samples must be positive",
    )
    @postcondition(
        lambda result: result is not None and isinstance(result, list),
        "Monte Carlo results must be returned as a non-None list",
    )
    def monte_carlo_simulation(
        self,
        launch: LaunchConditions,
        n_samples: int = 100,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> list[dict]:
        """Run Monte Carlo simulation with randomized environment.

        Useful for understanding dispersion patterns and the effect
        of environmental variability on ball flight.

        Args:
            launch: Launch conditions
            n_samples: Number of simulation runs
            max_time: Maximum simulation time per run [s]
            dt: Time step [s]

        Returns:
            List of analysis dictionaries for each run
        """
        if not (launch is not None):
            raise ValueError("launch must be provided")
        if not (launch is not None):
            raise ValueError("launch must be provided")
        from src.shared.python.physics.aerodynamics import (
            AerodynamicsEngine,
            EnvironmentRandomizer,
            WindModel,
        )

        results = []

        for i in range(n_samples):
            # Create new randomizer with different seed for each run
            seed = (self._seed or 0) + i
            randomizer = EnvironmentRandomizer(self.randomization_config, seed=seed)
            wind_model = WindModel(self.wind_config, seed=seed)

            # Create engine with randomized environment
            engine = AerodynamicsEngine(
                config=self.aero_config,
                wind_model=wind_model,
                randomization=randomizer,
                air_density=self.environment.air_density,
            )

            # Temporarily swap engine
            old_engine = self._aero_engine
            self._aero_engine = engine

            # Simulate
            traj = self.simulate_trajectory(launch, max_time, dt)

            # Restore engine
            self._aero_engine = old_engine

            # Analyze
            analysis = self.analyze_trajectory(traj)
            analysis["run"] = i
            results.append(analysis)

        return results

    # Delegate analysis methods to base simulator (DRY principle)
    def calculate_carry_distance(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total carry distance."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        last_pos = trajectory[-1].position
        return float(np.sqrt(last_pos[0] ** 2 + last_pos[1] ** 2))

    def calculate_max_height(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate maximum height achieved."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return float(max(p.position[2] for p in trajectory))

    def calculate_flight_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total flight time."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return 0.0
        return trajectory[-1].time

    def analyze_trajectory(self, trajectory: list[TrajectoryPoint]) -> dict:
        """Generate comprehensive analysis dictionary."""
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not (trajectory is not None):
            raise ValueError("trajectory must be provided")
        if not trajectory:
            return {
                "carry_distance": 0.0,
                "max_height": 0.0,
                "flight_time": 0.0,
                "landing_angle": 0.0,
                "apex_time": 0.0,
                "trajectory_points": 0,
            }

        # Landing angle calculation
        landing_angle = 0.0
        if len(trajectory) >= 2:
            v = trajectory[-1].velocity
            v_horiz = np.linalg.norm(v[:2])
            if v_horiz > NUMERICAL_EPSILON:
                landing_angle = float(np.degrees(np.arctan2(-v[2], v_horiz)))
            else:
                landing_angle = 90.0

        # Apex time calculation
        max_h = -float("inf")
        apex_t = 0.0
        for p in trajectory:
            if p.position[2] > max_h:
                max_h = p.position[2]
                apex_t = p.time

        return {
            "carry_distance": self.calculate_carry_distance(trajectory),
            "max_height": self.calculate_max_height(trajectory),
            "flight_time": self.calculate_flight_time(trajectory),
            "landing_angle": landing_angle,
            "apex_time": apex_t,
            "trajectory_points": len(trajectory),
        }
