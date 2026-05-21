"""Enhanced ball flight simulator with toggleable aerodynamics and Monte Carlo support.

This submodule contains EnhancedBallFlightSimulator, which integrates with the
aerodynamics module for configurable drag/lift/Magnus effects, wind models, and
environment randomization. Extracted from ball_flight_physics.py as part of P1
sprint decomposition (issue #2486).

Environmental gradient support (altitude-dependent air density) is provided
via the ``track_altitude_density`` flag and uses
:func:`src.shared.python.physics.atmosphere.air_density`. See issue #3504.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.physics.atmosphere import (
    MAX_VALID_ALTITUDE_M,
    MIN_VALID_ALTITUDE_M,
)
from src.shared.python.physics.atmosphere import (
    air_density as _isa_air_density,
)
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)
from src.shared.python.physics.ball_properties import BallProperties
from src.shared.python.physics.ball_trajectory_analysis import TrajectoryAnalysisMixin

if TYPE_CHECKING:
    from src.shared.python.physics.aerodynamics import (
        AerodynamicsConfig,
        RandomizationConfig,
        WindConfig,
    )


class EnhancedBallFlightSimulator(TrajectoryAnalysisMixin):
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
        track_altitude_density: bool = False,
    ) -> None:
        """Initialize enhanced simulator.

        Args:
            ball: Golf ball properties
            environment: Environmental conditions (temperature, altitude)
            aero_config: Aerodynamics configuration (toggles and coefficients)
            wind_config: Wind configuration (base wind, gusts, turbulence)
            randomization_config: Environment randomization configuration
            seed: Random seed for reproducibility
            track_altitude_density: When ``True``, the integrator updates
                ``rho`` each step from the ISA-troposphere model using the
                ball's current altitude (course altitude plus trajectory z)
                and the environment's ground temperature/pressure. Defaults
                to ``False`` to preserve the supplied
                ``environment.air_density`` (e.g. humidity- or
                weather-calibrated values). Pair with
                ``EnvironmentalConditions.from_altitude(...)`` to opt in to
                ISA tracking; any non-``None``
                ``environment.sea_level_pressure_pa`` is propagated to each
                per-step density update.
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
        self.track_altitude_density = bool(track_altitude_density)

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
        if launch is None:
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
            # Update altitude-dependent air density before force evaluation
            # so the integrator sees thinner air at higher Z (issue #3504).
            self._update_air_density_for_position(position)

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

        if pos is None:
            raise ValueError("pos must be provided")

        def derivatives(
            p: np.ndarray, v: np.ndarray, time: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """Compute velocity and acceleration derivatives for RK4 integration."""
            if p is None:
                raise ValueError("p must be provided")
            self._update_air_density_for_position(p)
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

    def _update_air_density_for_position(self, position: np.ndarray) -> None:
        """Update the engine's current air density from ball altitude.

        Combines the course altitude (``environment.altitude``) with the
        instantaneous trajectory ``z`` to obtain altitude above mean sea
        level, then queries
        :func:`src.shared.python.physics.atmosphere.air_density`. The result
        is clamped to the validated altitude range so the integrator never
        crashes on extreme transient values during early RK4 iterations.

        No-op when ``track_altitude_density`` is False.
        """
        if not self.track_altitude_density:
            return
        course_alt = float(getattr(self.environment, "altitude", 0.0))
        ball_z = float(position[2]) if position is not None else 0.0
        amsl = course_alt + ball_z
        # Clamp to the validated atmosphere range; outside it we just hold
        # at the boundary value rather than raising mid-integration.
        amsl = max(MIN_VALID_ALTITUDE_M, min(MAX_VALID_ALTITUDE_M, amsl))
        rho = _isa_air_density(
            altitude_m=amsl,
            temperature_c=float(getattr(self.environment, "temperature", 15.0)),
            pressure_pa=getattr(self.environment, "sea_level_pressure_pa", None),
        )
        self._aero_engine._current_air_density = rho

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
        if launch is None:
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
        if launch is None:
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

    # Analysis methods are inherited from TrajectoryAnalysisMixin (DRY principle).
