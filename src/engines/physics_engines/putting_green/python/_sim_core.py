from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

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
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    GrassType,
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
        if not (random_seed is not None):
            raise ValueError("random_seed must be provided")
        if not (random_seed is not None):
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
        if not (path is not None):
            raise ValueError("path must be provided")
        if not (path is not None):
            raise ValueError("path must be provided")
        filepath = Path(path)

        with open(filepath) as f:
            data = json.load(f)

        self._load_from_data(data)

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load green configuration from string.

        Args:
            content: Configuration content
            extension: Format hint (e.g., "json")
        """
        if not (content is not None):
            raise ValueError("content must be provided")
        if not (content is not None):
            raise ValueError("content must be provided")
        data = json.loads(content)
        self._load_from_data(data)

    def _load_from_data(self, data: dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        if not (data is not None):
            raise ValueError("data must be provided")
        if not (data is not None):
            raise ValueError("data must be provided")
        if "green" in data:
            green_data = data["green"]

            turf_data = green_data.get("turf", {})
            if "stimp_rating" in turf_data:
                grass_type = GrassType(turf_data.get("grass_type", "bent_grass"))
                turf = TurfProperties(
                    stimp_rating=turf_data["stimp_rating"],
                    grass_type=grass_type,
                )
            else:
                turf = TurfProperties()

            self.green = GreenSurface(
                width=green_data.get("width", 20.0),
                height=green_data.get("height", 20.0),
                turf=turf,
            )

            if "hole_position" in green_data:
                self.green.set_hole_position(np.array(green_data["hole_position"]))

            if "slopes" in green_data:
                for s in green_data["slopes"]:
                    self.green.add_slope_region(
                        SlopeRegion(
                            center=np.array(s["center"]),
                            radius=s["radius"],
                            slope_direction=np.array(s["direction"]),
                            slope_magnitude=s["magnitude"],
                        )
                    )

        self._physics = BallRollPhysics(
            green=self.green,
            integrator=self.config.integrator,
        )

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
        if not (path is not None):
            raise ValueError("path must be provided")
        if not (path is not None):
            raise ValueError("path must be provided")
        filepath = Path(path)
        suffix = filepath.suffix.lower()

        if width is not None:
            self.green.width = width
        if height is not None:
            self.green.height = height

        if suffix == ".npy":
            heightmap = np.load(filepath)
            self.green.set_heightmap(heightmap)
        elif suffix == ".csv" or suffix in (".tif", ".tiff"):
            self.green.load_from_file(filepath)
        else:
            self.green.load_from_file(filepath)

        self._physics = BallRollPhysics(
            green=self.green,
            integrator=self.config.integrator,
        )

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
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        self._ball_state.position = np.array(q)
        self._ball_state.velocity = np.array(v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control input (force on ball)."""
        if not (u is not None):
            raise ValueError("u must be provided")
        if not (u is not None):
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
        if not (stroke_params is not None):
            raise ValueError("stroke_params must be provided")
        if not (stroke_params is not None):
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
                    self._trajectory["positions"].append(
                        self._ball_state.position.copy()
                    )
                    self._trajectory["velocities"].append(
                        self._ball_state.velocity.copy()
                    )
                    self._trajectory["times"].append(self._time)
                    self._trajectory["modes"].append(
                        self._physics.determine_roll_mode(self._ball_state)
                    )
                break

            if not self.green.is_on_green(self._ball_state.position):
                self._ball_state.velocity = np.zeros(2)
                if self.config.record_trajectory:
                    self._trajectory["positions"].append(
                        self._ball_state.position.copy()
                    )
                    self._trajectory["velocities"].append(
                        self._ball_state.velocity.copy()
                    )
                    self._trajectory["times"].append(self._time)
                    self._trajectory["modes"].append(
                        self._physics.determine_roll_mode(self._ball_state)
                    )
                break

        return SimulationResult(
            positions=np.array(self._trajectory["positions"]),
            velocities=np.array(self._trajectory["velocities"]),
            times=np.array(self._trajectory["times"]),
            holed=holed,
            final_position=self._ball_state.position.copy(),
            modes=self._trajectory["modes"],
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
        return StateCheckpoint.create(
            engine_type="putting_green",
            engine_state={
                "spin": self._ball_state.spin.tolist(),
            },
            q=self._ball_state.position,
            v=self._ball_state.velocity,
            timestamp=self._time,
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from checkpoint."""
        if not (checkpoint is not None):
            raise ValueError("checkpoint must be provided")
        if not (checkpoint is not None):
            raise ValueError("checkpoint must be provided")
        self._ball_state.position = checkpoint.get_q()
        self._ball_state.velocity = checkpoint.get_v()
        self._time = checkpoint.timestamp
        if "spin" in checkpoint.engine_state:
            self._ball_state.spin = np.array(checkpoint.engine_state["spin"])

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute mass matrix (scalar mass for single ball)."""
        return np.eye(2) * self.ball_mass

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces (friction + slope)."""
        accel = self._physics.compute_total_acceleration(self._ball_state)
        return self.ball_mass * accel

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravitational forces from slope."""
        g_accel = self._physics.compute_slope_acceleration(self._ball_state.position)
        return self.ball_mass * g_accel

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute forces required for given acceleration."""
        return self.ball_mass * qacc

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute Jacobian (identity for ball)."""
        if not (body_name is not None):
            raise ValueError("body_name must be provided")
        if not (body_name is not None):
            raise ValueError("body_name must be provided")
        if body_name == "ball":
            return {
                "linear": np.eye(2),
                "angular": np.zeros((1, 2)),
            }
        return None

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive drift acceleration."""
        return self._physics.compute_total_acceleration(self._ball_state)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute acceleration from applied force."""
        return tau / self.ball_mass

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-torque counterfactual (drift only)."""
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        temp_state = BallState(q, v, self._ball_state.spin)
        return self._physics.compute_total_acceleration(temp_state)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-velocity counterfactual."""
        return self._physics.compute_slope_acceleration(q)

    def set_real_time_mode(self, enabled: bool) -> None:
        """Enable or disable real-time simulation mode."""
        self._real_time_mode = enabled

    def set_wind(self, speed: float, direction: np.ndarray) -> None:
        """Set wind conditions.

        Args:
            speed: Wind speed [m/s]
            direction: Wind direction (unit vector)
        """
        if not (speed is not None):
            raise ValueError("speed must be provided")
        if not (speed is not None):
            raise ValueError("speed must be provided")
        self._wind_speed = speed
        mag = np.linalg.norm(direction)
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
        """Simulate putt with practice feedback.

        Args:
            stroke_params: Stroke parameters

        Returns:
            Dictionary with result and feedback
        """
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
        """Simulate multiple putts with variance for scatter analysis.

        Args:
            start_position: Starting ball position
            stroke_params: Base stroke parameters
            n_simulations: Number of simulations
            speed_variance: Standard deviation of speed [m/s]
            direction_variance_deg: Standard deviation of direction [degrees]
            rng: Optional random generator (defaults to simulator RNG)

        Returns:
            List of simulation results
        """
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
        """Compute aim line accounting for break.

        Args:
            ball_position: Current ball position

        Returns:
            Dictionary with aim information
        """
        return compute_aim_line(self, ball_position)

    def read_green(
        self, ball_position: np.ndarray, target: np.ndarray
    ) -> dict[str, Any]:
        """Read green between ball and target.

        Args:
            ball_position: Ball position
            target: Target position

        Returns:
            Green reading with slopes and recommendations
        """
        return read_green(self, ball_position, target)

    def export_result(self, result: SimulationResult, path: str) -> None:
        """Export simulation result to file.

        Args:
            result: Simulation result
            path: Output file path
        """
        export_result(result, path)
