from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.engines.physics_engines.putting_green.python._checkpoint import (
    get_checkpoint,
    restore_checkpoint,
)
from src.engines.physics_engines.putting_green.python._dynamics import (
    compute_bias_forces,
    compute_control_acceleration,
    compute_drift_acceleration,
    compute_gravity_forces,
    compute_inverse_dynamics,
    compute_jacobian,
    compute_mass_matrix,
    compute_ztcf,
    compute_zvcf,
)
from src.engines.physics_engines.putting_green.python._green_loader import (
    load_from_data,
    load_from_path,
    load_from_string,
    load_topographical_data,
)
from src.engines.physics_engines.putting_green.python._practice_mode import (
    compute_aim_line,
    export_result,
    read_green,
    simulate_scatter,
    simulate_with_feedback,
)
from src.engines.physics_engines.putting_green.python._sim_config import (
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python._wind_physics import (
    compute_wind_force,
)
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
    RollMode,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint


class PuttingGreenSimulator:
    """Main putting green simulation engine.

    Implements the PhysicsEngine protocol for integration with the
    unified simulation framework.

    Example:
        >>> sim = PuttingGreenSimulator()
        >>> sim.set_ball_position(np.array([5.0, 10.0]))
        >>> stroke = StrokeParameters(speed=2.0, direction=np.array([1.0, 0.0]))
        >>> result = sim.simulate_putt(stroke)
        >>> print(f"Ball stopped at {result.final_position}")
    """

    def __init__(
        self,
        green: GreenSurface | None = None,
        config: SimulationConfig | None = None,
        putter: PutterStroke | None = None,
        rng: np.random.Generator | None = None,
        random_seed: int = 0,
    ) -> None:
        """Initialize simulator.

        Args:
            green: Putting green surface (creates default if None)
            config: Simulation configuration
            putter: Putter model for strokes
            rng: Optional numpy random generator for deterministic scatter
            random_seed: Seed for deterministic randomness (used if rng is None)
        """
        if random_seed is None:
            raise ValueError("random_seed must be provided")
        self.config = config or SimulationConfig()
        self.green = green or GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        self.putter = putter or PutterStroke()

        self._physics = BallRollPhysics(
            green=self.green,
            integrator=self.config.integrator,
        )

        self._ball_state = BallState(
            position=np.array([self.green.width / 2, self.green.height / 2]),
            velocity=np.zeros(2),
            spin=np.zeros(3),
        )
        self._time = 0.0
        self._real_time_mode = False
        self._last_acceleration: np.ndarray | None = None
        self._last_roll_mode: RollMode | None = None

        self._trajectory: dict[str, list[Any]] = {
            "positions": [],
            "velocities": [],
            "times": [],
            "modes": [],
        }

        self._wind_speed = 0.0
        self._wind_direction = np.array([1.0, 0.0])

        self._practice_mode = False

        self._rng = rng or np.random.default_rng(random_seed)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "putting_green"

    @property
    def ball_mass(self) -> float:
        """Ball mass in kg."""
        return self._physics.ball_mass

    def load_from_path(self, path: str) -> None:
        """Load green configuration from file.

        Supports JSON configuration files.

        Args:
            path: Path to configuration file
        """
        load_from_path(self, path)

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load green configuration from string.

        Args:
            content: Configuration content
            extension: Format hint (e.g., "json")
        """
        load_from_string(self, content, extension)

    def _load_from_data(self, data: dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        load_from_data(self, data)

    def load_topographical_data(
        self,
        path: str,
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        """Load topographical/elevation data.

        Args:
            path: Path to topographical data file
            width: Physical width [m] (uses current if None)
            height: Physical height [m] (uses current if None)
        """
        load_topographical_data(self, path, width, height)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._time = 0.0
        self._ball_state = BallState(
            position=np.array([self.green.width / 2, self.green.height / 2]),
            velocity=np.zeros(2),
            spin=np.zeros(3),
        )
        self._last_acceleration = None
        self._last_roll_mode = None
        self._trajectory = {
            "positions": [],
            "velocities": [],
            "times": [],
            "modes": [],
        }

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by one time step.

        Args:
            dt: Time step (uses config default if None)
        """
        dt = dt or self.config.timestep

        if self._wind_speed > 0 and self._ball_state.is_moving:
            wind_force = compute_wind_force(
                self._wind_speed, self._wind_direction, self._ball_state.velocity
            )
            wind_accel = wind_force / self.ball_mass
            self._ball_state.velocity += wind_accel * dt

        self._ball_state = self._physics.step(self._ball_state, dt)
        self._time += dt
        self._last_acceleration = self._physics.compute_total_acceleration(
            self._ball_state
        )
        self._last_roll_mode = self._physics.determine_roll_mode(self._ball_state)

        if self.config.record_trajectory:
            self._trajectory["positions"].append(self._ball_state.position.copy())
            self._trajectory["velocities"].append(self._ball_state.velocity.copy())
            self._trajectory["times"].append(self._time)
            self._trajectory["modes"].append(
                self._physics.determine_roll_mode(self._ball_state)
            )

    def forward(self) -> None:
        """Compute kinematics without advancing time."""
        self._last_acceleration = self._physics.compute_total_acceleration(
            self._ball_state
        )
        self._last_roll_mode = self._physics.determine_roll_mode(self._ball_state)

    def get_last_acceleration(self) -> np.ndarray | None:
        """Get last computed acceleration."""
        if self._last_acceleration is None:
            return None
        return self._last_acceleration.copy()

    def get_last_roll_mode(self) -> RollMode | None:
        """Get last computed roll mode."""
        return self._last_roll_mode

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current state (position, velocity)."""
        return self._ball_state.position.copy(), self._ball_state.velocity.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set current state."""
        if q is None:
            raise ValueError("q must be provided")
        self._ball_state.position = np.array(q)
        self._ball_state.velocity = np.array(v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control input (force on ball)."""
        if u is None:
            raise ValueError("u must be provided")
        accel = u / self.ball_mass
        self._ball_state.velocity += accel * self.config.timestep

    def get_time(self) -> float:
        """Get current simulation time."""
        return self._time

    def get_ball_position(self) -> np.ndarray:
        """Get current ball position."""
        return self._ball_state.position.copy()

    def set_ball_position(self, position: np.ndarray) -> None:
        """Set ball position."""
        self._ball_state.position = np.array(position[:2])

    def get_ball_velocity(self) -> np.ndarray:
        """Get current ball velocity."""
        return self._ball_state.velocity.copy()

    def set_ball_velocity(self, velocity: np.ndarray) -> None:
        """Set ball velocity."""
        self._ball_state.velocity = np.array(velocity[:2])

    def simulate_putt(
        self,
        stroke_params: StrokeParameters,
        ball_position: np.ndarray | None = None,
    ) -> SimulationResult:
        """Simulate a complete putt.

        Args:
            stroke_params: Parameters of the putting stroke
            ball_position: Starting position (uses current if None)

        Returns:
            SimulationResult with trajectory and outcome
        """
        if stroke_params is None:
            raise ValueError("stroke_params must be provided")
        if ball_position is not None:
            self.set_ball_position(ball_position)

        self._ball_state = self.putter.execute_stroke(
            self._ball_state.position, stroke_params
        )

        self._trajectory = {
            "positions": [self._ball_state.position.copy()],
            "velocities": [self._ball_state.velocity.copy()],
            "times": [0.0],
            "modes": [self._physics.determine_roll_mode(self._ball_state)],
        }
        self._time = 0.0

        holed = False
        while (
            self._time < self.config.max_simulation_time and self._ball_state.is_moving
        ):
            self.step()

            if self.green.is_in_hole(
                self._ball_state.position, self._ball_state.velocity
            ):
                holed = True
                self._ball_state.velocity = np.zeros(2)
                if self.config.record_trajectory:
                    self._record_terminal_step()
                break

            if not self.green.is_on_green(self._ball_state.position):
                self._ball_state.velocity = np.zeros(2)
                if self.config.record_trajectory:
                    self._record_terminal_step()
                break

        return SimulationResult(
            positions=np.array(self._trajectory["positions"]),
            velocities=np.array(self._trajectory["velocities"]),
            times=np.array(self._trajectory["times"]),
            holed=holed,
            final_position=self._ball_state.position.copy(),
            modes=self._trajectory["modes"],
        )

    def _record_terminal_step(self) -> None:
        """Record the final step to the trajectory."""
        self._trajectory["positions"].append(self._ball_state.position.copy())
        self._trajectory["velocities"].append(self._ball_state.velocity.copy())
        self._trajectory["times"].append(self._time)
        self._trajectory["modes"].append(
            self._physics.determine_roll_mode(self._ball_state)
        )

    def get_current_trajectory(self) -> dict[str, Any]:
        """Get trajectory recorded so far."""
        return {
            "positions": np.array(self._trajectory["positions"]),
            "velocities": np.array(self._trajectory["velocities"]),
            "times": np.array(self._trajectory["times"]),
        }

    def get_checkpoint(self) -> StateCheckpoint:
        """Save current state to checkpoint."""
        return get_checkpoint(self)

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from checkpoint."""
        restore_checkpoint(self, checkpoint)

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute mass matrix (scalar mass for single ball)."""
        return compute_mass_matrix(self)

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces (friction + slope)."""
        return compute_bias_forces(self)

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravitational forces from slope."""
        return compute_gravity_forces(self)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute forces required for given acceleration."""
        return compute_inverse_dynamics(self, qacc)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute Jacobian (identity for ball)."""
        return compute_jacobian(self, body_name)

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive drift acceleration."""
        return compute_drift_acceleration(self)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute acceleration from applied force."""
        return compute_control_acceleration(self, tau)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-torque counterfactual (drift only)."""
        return compute_ztcf(self, q, v)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-velocity counterfactual."""
        return compute_zvcf(self, q)

    def set_real_time_mode(self, enabled: bool) -> None:
        """Enable or disable real-time simulation mode."""
        self._real_time_mode = enabled

    def set_wind(self, speed: float, direction: np.ndarray) -> None:
        """Set wind conditions.

        Args:
            speed: Wind speed [m/s]
            direction: Wind direction (unit vector)
        """
        if speed is None:
            raise ValueError("speed must be provided")
        self._wind_speed = speed
        mag = math.hypot(*direction)
        if mag > 0:
            self._wind_direction = direction / mag

    def _compute_wind_force(self) -> np.ndarray:
        """Compute wind force on ball."""
        return compute_wind_force(
            self._wind_speed, self._wind_direction, self._ball_state.velocity
        )

    def enable_practice_mode(self) -> None:
        """Enable practice mode with feedback."""
        self._practice_mode = True

    def simulate_with_feedback(self, stroke_params: StrokeParameters) -> dict[str, Any]:
        """Simulate putt with practice feedback."""
        return simulate_with_feedback(self, stroke_params)

    def simulate_scatter(
        self,
        start_position: np.ndarray,
        stroke_params: StrokeParameters,
        n_simulations: int = 10,
        speed_variance: float = 0.1,
        direction_variance_deg: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> list[SimulationResult]:
        """Simulate multiple putts with variance for scatter analysis."""
        return simulate_scatter(
            self,
            start_position,
            stroke_params,
            n_simulations,
            speed_variance,
            direction_variance_deg,
            rng,
        )

    def compute_aim_line(self, ball_position: np.ndarray) -> dict[str, Any]:
        """Compute aim line accounting for break."""
        return compute_aim_line(self, ball_position)

    def read_green(
        self, ball_position: np.ndarray, target: np.ndarray
    ) -> dict[str, Any]:
        """Read green between ball and target."""
        return read_green(self, ball_position, target)

    def export_result(self, result: SimulationResult, path: str) -> None:
        """Export simulation result to file."""
        export_result(result, path)
