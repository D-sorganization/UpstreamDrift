"""Putting Green Simulator - Main Physics Engine.

This module implements the main PuttingGreenSimulator class that provides
a complete putting simulation conforming to the PhysicsEngine protocol.

Features:
    - Real-time ball rolling simulation
    - Configurable turf and surface properties
    - Support for topographical data loading
    - Putter stroke simulation
    - Trajectory recording and replay
    - Wind effects (optional)
    - Practice mode with feedback

Design by Contract:
    - Follows PhysicsEngine protocol
    - Thread-safe state management
    - Deterministic simulation (same inputs = same outputs)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    PUTTING_WIND_DRAG_COEFFICIENT,
    PUTTING_WIND_FORCE_SCALING,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint


@dataclass
class SimulationConfig:
    """Configuration for putting simulation.

    Attributes:
        timestep: Simulation time step [s]
        max_simulation_time: Maximum simulation duration [s]
        stopping_velocity_threshold: Speed below which ball stops [m/s]
        record_trajectory: Whether to record full trajectory
        integrator: Integration method ("euler", "rk4", "verlet")
    """

    timestep: float = 0.001
    max_simulation_time: float = 30.0
    stopping_velocity_threshold: float = 0.005
    record_trajectory: bool = True
    integrator: str = "euler"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")
        if self.max_simulation_time <= 0:
            raise ValueError(
                f"max_simulation_time must be positive, got {self.max_simulation_time}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestep": self.timestep,
            "max_simulation_time": self.max_simulation_time,
            "stopping_velocity_threshold": self.stopping_velocity_threshold,
            "record_trajectory": self.record_trajectory,
            "integrator": self.integrator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class SimulationResult:
    """Result of a putting simulation.

    Attributes:
        positions: Array of ball positions [[x, y], ...]
        velocities: Array of ball velocities [[vx, vy], ...]
        times: Array of time stamps [t0, t1, ...]
        holed: Whether ball went in hole
        final_position: Final ball position
        spins: Optional array of spin vectors
        modes: Optional list of roll modes at each step
    """

    positions: np.ndarray
    velocities: np.ndarray
    times: np.ndarray
    holed: bool
    final_position: np.ndarray
    spins: np.ndarray | None = None
    modes: list[RollMode] | None = None

    @property
    def total_distance(self) -> float:
        """Compute total distance rolled."""
        if len(self.positions) < 2:
            return 0.0

        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return float(np.sum(distances))

    @property
    def duration(self) -> float:
        """Total simulation duration."""
        return float(self.times[-1] - self.times[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "times": self.times.tolist(),
            "holed": self.holed,
            "final_position": self.final_position.tolist(),
            "total_distance": self.total_distance,
            "duration": self.duration,
        }


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
        self.config = config or SimulationConfig()
        self.green = green or GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        self.putter = putter or PutterStroke()

        # Physics engine
        self._physics = BallRollPhysics(
            green=self.green,
            integrator=self.config.integrator,
        )

        # State
        self._ball_state = BallState(
            position=np.array([self.green.width / 2, self.green.height / 2]),
            velocity=np.zeros(2),
            spin=np.zeros(3),
        )
        self._time = 0.0
        self._real_time_mode = False
        self._last_acceleration: np.ndarray | None = None
        self._last_roll_mode: RollMode | None = None

        # Trajectory recording
        self._trajectory: dict[str, list[Any]] = {
            "positions": [],
            "velocities": [],
            "times": [],
            "modes": [],
        }

        # Optional wind
        self._wind_speed = 0.0
        self._wind_direction = np.array([1.0, 0.0])

        # Practice mode
        self._practice_mode = False

        # Randomness (seeded for determinism)
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
        data = json.loads(content)
        self._load_from_data(data)

    def _load_from_data(self, data: dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        if "green" in data:
            green_data = data["green"]

            # Create turf
            turf_data = green_data.get("turf", {})
            if "stimp_rating" in turf_data:
                grass_type = GrassType(turf_data.get("grass_type", "bent_grass"))
                turf = TurfProperties(
                    stimp_rating=turf_data["stimp_rating"],
                    grass_type=grass_type,
                )
            else:
                turf = TurfProperties()

            # Create green
            self.green = GreenSurface(
                width=green_data.get("width", 20.0),
                height=green_data.get("height", 20.0),
                turf=turf,
            )

            # Set hole position
            if "hole_position" in green_data:
                self.green.set_hole_position(np.array(green_data["hole_position"]))

            # Load slope regions
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

        # Update physics
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

        # Rebuild physics with updated green
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

        # Apply wind effect
        if self._wind_speed > 0 and self._ball_state.is_moving:
            wind_force = self._compute_wind_force()
            wind_accel = wind_force / self.ball_mass
            self._ball_state.velocity += wind_accel * dt

        # Physics step
        self._ball_state = self._physics.step(self._ball_state, dt)
        self._time += dt
        self._last_acceleration = self._physics.compute_total_acceleration(
            self._ball_state
        )
        self._last_roll_mode = self._physics.determine_roll_mode(self._ball_state)

        # Record trajectory
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
        self._ball_state.position = np.array(q)
        self._ball_state.velocity = np.array(v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply control input (force on ball)."""
        # Not typically used for putting, but implemented for protocol
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
        # Set ball position
        if ball_position is not None:
            self.set_ball_position(ball_position)

        # Execute stroke
        self._ball_state = self.putter.execute_stroke(
            self._ball_state.position, stroke_params
        )

        # Clear trajectory
        self._trajectory = {
            "positions": [self._ball_state.position.copy()],
            "velocities": [self._ball_state.velocity.copy()],
            "times": [0.0],
            "modes": [self._physics.determine_roll_mode(self._ball_state)],
        }
        self._time = 0.0

        # Simulate
        holed = False
        while (
            self._time < self.config.max_simulation_time and self._ball_state.is_moving
        ):
            self.step()

            # Check for hole
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

            # Check for off-green
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

    # Checkpoint methods
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
        self._ball_state.position = checkpoint.get_q()
        self._ball_state.velocity = checkpoint.get_v()
        self._time = checkpoint.timestamp
        if "spin" in checkpoint.engine_state:
            self._ball_state.spin = np.array(checkpoint.engine_state["spin"])

    # PhysicsEngine dynamics methods
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
        temp_state = BallState(q, v, self._ball_state.spin)
        return self._physics.compute_total_acceleration(temp_state)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-velocity counterfactual."""
        return self._physics.compute_slope_acceleration(q)

    # Real-time mode
    def set_real_time_mode(self, enabled: bool) -> None:
        """Enable or disable real-time simulation mode."""
        self._real_time_mode = enabled

    # Wind
    def set_wind(self, speed: float, direction: np.ndarray) -> None:
        """Set wind conditions.

        Args:
            speed: Wind speed [m/s]
            direction: Wind direction (unit vector)
        """
        self._wind_speed = speed
        mag = np.linalg.norm(direction)
        if mag > 0:
            self._wind_direction = direction / mag

    def _compute_wind_force(self) -> np.ndarray:
        """Compute wind force on ball."""
        if self._wind_speed <= 0:
            return np.zeros(2)

        # Simplified aerodynamic drag from wind
        # F = 0.5 * rho * Cd * A * v^2
        rho = AIR_DENSITY_SEA_LEVEL_KG_M3
        Cd = PUTTING_WIND_DRAG_COEFFICIENT
        A = GOLF_BALL_CROSS_SECTIONAL_AREA_M2

        # Relative velocity
        relative_v = (
            self._wind_direction * self._wind_speed - self._ball_state.velocity[:2]
        )
        rel_speed = np.linalg.norm(relative_v)

        if rel_speed < 0.1:
            return np.zeros(2)

        force_mag = 0.5 * rho * Cd * A * rel_speed**2
        force_dir = relative_v / rel_speed

        return force_mag * force_dir * PUTTING_WIND_FORCE_SCALING

    # Practice mode
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
        result = self.simulate_putt(stroke_params)

        distance_from_hole = np.linalg.norm(
            result.final_position - self.green.hole_position
        )

        # Generate feedback
        feedback = {
            "distance_from_hole": distance_from_hole,
            "holed": result.holed,
            "total_distance": result.total_distance,
        }

        if result.holed:
            feedback["suggested_adjustment"] = "Great putt"
        else:
            # Suggest adjustment
            if distance_from_hole < 0.5:
                if result.final_position[0] < self.green.hole_position[0]:
                    feedback["suggested_adjustment"] = "Hit slightly firmer"
                else:
                    feedback["suggested_adjustment"] = "Hit slightly softer"
            else:
                feedback["suggested_adjustment"] = "Check your aim line"

        return feedback

    # Scatter analysis
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
        results = []
        rng = rng or self._rng

        for _ in range(n_simulations):
            # Add variance
            speed = stroke_params.speed + rng.normal(0, speed_variance)
            speed = max(0.1, speed)  # Ensure positive

            angle_var = rng.normal(0, direction_variance_deg * np.pi / 180)
            cos_a, sin_a = np.cos(angle_var), np.sin(angle_var)
            direction = np.array(
                [
                    cos_a * stroke_params.direction[0]
                    - sin_a * stroke_params.direction[1],
                    sin_a * stroke_params.direction[0]
                    + cos_a * stroke_params.direction[1],
                ]
            )

            varied_params = StrokeParameters(
                speed=speed,
                direction=direction,
                face_angle=stroke_params.face_angle + rng.normal(0, 1.0),
                attack_angle=stroke_params.attack_angle,
            )

            result = self.simulate_putt(varied_params, ball_position=start_position)
            results.append(result)

        return results

    # Aim assist
    def compute_aim_line(self, ball_position: np.ndarray) -> dict[str, Any]:
        """Compute aim line accounting for break.

        Args:
            ball_position: Current ball position

        Returns:
            Dictionary with aim information
        """
        target = self.green.hole_position

        # Calculate break
        break_info = self.green.calculate_break(ball_position, target)

        # Aim point compensates for break
        aim_point = target - break_info["break_direction"] * break_info["total_break"]

        # Recommended speed
        distance = float(np.linalg.norm(target - ball_position))
        avg_slope = np.dot(
            break_info["average_slope"], (target - ball_position) / (distance + 1e-10)
        )
        recommended_speed = self.putter.estimate_required_speed(
            distance, self.green.turf.stimp_rating, slope_percent=avg_slope * 100
        )

        return {
            "aim_point": aim_point,
            "break": break_info["total_break"],
            "break_direction": break_info["break_direction"],
            "recommended_speed": recommended_speed,
            "distance": distance,
        }

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
        reading = self.green.read_putt_line(ball_position, target)
        break_info = self.green.calculate_break(ball_position, target)
        aim_info = self.compute_aim_line(ball_position)

        return {
            "positions": reading["positions"],
            "elevations": reading["elevations"],
            "slopes": reading["slopes"],
            "distance": reading["distance"],
            "total_break": break_info["total_break"],
            "recommended_speed": aim_info["recommended_speed"],
            "aim_point": aim_info["aim_point"],
        }

    def export_result(self, result: SimulationResult, path: str) -> None:
        """Export simulation result to file.

        Args:
            result: Simulation result
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
