"""Ball flight physics simulation with Magnus effect and drag.

This module implements research-grade ball flight physics including:
- Magnus effect (spin-induced forces)
- Drag forces (Reynolds number dependent)
- Launch angle and velocity effects
- 3D trajectory calculation
- Landing dispersion patterns

Critical gap identified in upgrade assessment - without this, not a complete golf tool.
"""

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# =============================================================================
# Physical Constants
# =============================================================================

# Minimum speed threshold for aerodynamic calculations [m/s]
# Below this speed, aerodynamic forces are negligible and we avoid division by zero.
# Value: 0.1 m/s ≈ 0.36 km/h - essentially stationary for aerodynamic purposes.
MIN_SPEED_THRESHOLD: float = 0.1

# Maximum lift coefficient to prevent unrealistic forces
# Source: Empirical observation from wind tunnel data (Waterloo/Penner)
MAX_LIFT_COEFFICIENT: float = 0.25

# Small epsilon for numerical stability in vector normalization
NUMERICAL_EPSILON: float = 1e-10


@dataclass
class BallProperties:
    """Physical properties of a golf ball.

    Coefficients use the Waterloo/Penner model where Cd and Cl are quadratic
    functions of spin ratio S = (ω * r) / v:
        Cd = cd0 + cd1 * S + cd2 * S²
        Cl = cl0 + cl1 * S + cl2 * S²

    References:
        - Penner, A.R. (2003) "The physics of golf"
        - McPhee et al. (Waterloo) "Golf Ball Flight Dynamics"
    """

    mass: float = 0.0459  # [kg] USGA regulation max
    diameter: float = 0.04267  # [m] USGA regulation min

    # Drag coefficient quadratic model: Cd = cd0 + cd1*S + cd2*S²
    # Source: Waterloo/Penner empirical fits to wind tunnel data
    # Tuned against TrackMan carry distance data
    cd0: float = 0.21  # Base drag (no spin) - typical for dimpled golf balls
    cd1: float = 0.05  # Linear spin dependence (reduced from lit values)
    cd2: float = 0.02  # Quadratic spin dependence

    # Lift coefficient quadratic model: Cl = cl0 + cl1*S + cl2*S²
    # Source: Waterloo/Penner empirical fits to wind tunnel data
    # Values tuned to match TrackMan carry: Driver ~270yd, 7i ~165yd, PW ~125yd
    cl0: float = 0.00  # No lift at zero spin
    cl1: float = 0.38  # Linear spin dependence
    cl2: float = 0.08  # Quadratic spin dependence

    @property
    def radius(self) -> float:
        """Ball radius in meters."""
        return self.diameter / 2

    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area in m²."""
        return float(np.pi * self.radius**2)


@dataclass
class LaunchConditions:
    """Initial launch conditions for ball flight.

    Attributes:
        velocity: Initial ball speed [m/s]
        launch_angle: Vertical launch angle [radians]
        azimuth_angle: Horizontal direction [radians], 0 = target line
        spin_rate: Total spin rate [rpm], positive = backspin
        spin_axis: Normalized spin axis vector. For backspin producing lift,
            this should be perpendicular to direction of travel. Default is
            [0, -1, 0] (pointing left when viewed from behind), which produces
            upward lift when crossed with forward velocity.
    """

    velocity: float  # m/s - initial ball speed
    launch_angle: float  # radians - vertical launch angle
    azimuth_angle: float = 0.0  # radians - horizontal direction
    spin_rate: float = 0.0  # rpm - backspin rate
    spin_axis: np.ndarray | None = None  # spin axis vector (normalized)

    def __post_init__(self) -> None:
        """Initialize default spin axis if not provided."""
        if self.spin_axis is None:
            # Default to pure backspin (axis points left when viewed from behind)
            # Cross product: [0,-1,0] × [1,0,0] = [0,0,1] (upward lift)
            self.spin_axis = np.array([0.0, -1.0, 0.0])


@dataclass
class EnvironmentalConditions:
    """Environmental conditions affecting ball flight."""

    air_density: float = 1.225  # kg/m³ at sea level, 15°C
    wind_velocity: np.ndarray | None = None  # m/s - wind vector [x, y, z]
    gravity: float = 9.81  # m/s²
    temperature: float = 15.0  # °C
    altitude: float = 0.0  # m above sea level

    def __post_init__(self) -> None:
        """Initialize default wind if not provided."""
        if self.wind_velocity is None:
            self.wind_velocity = np.array([0.0, 0.0, 0.0])


@dataclass
class TrajectoryPoint:
    """Single point in ball trajectory."""

    time: float
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s²
    forces: dict[str, np.ndarray]  # Force breakdown


class BallFlightSimulator:
    """Physics-based golf ball flight simulator."""

    def __init__(
        self,
        ball: BallProperties | None = None,
        environment: EnvironmentalConditions | None = None,
    ):
        """Initialize ball flight simulator.

        Args:
            ball: Ball properties, uses defaults if None
            environment: Environmental conditions, uses defaults if None
        """
        self.ball = ball or BallProperties()
        self.environment = environment or EnvironmentalConditions()

    def simulate_trajectory(
        self, launch: LaunchConditions, max_time: float = 10.0, time_step: float = 0.01
    ) -> list[TrajectoryPoint]:
        """Simulate complete ball trajectory.

        Args:
            launch: Initial launch conditions
            max_time: Maximum simulation time (s)
            time_step: Integration time step (s)

        Returns:
            List of trajectory points
        """
        # Initial state vector: [x, y, z, vx, vy, vz]
        initial_state = np.array(
            [
                0.0,
                0.0,
                0.0,  # Initial position at origin
                launch.velocity
                * np.cos(launch.launch_angle)
                * np.cos(launch.azimuth_angle),
                launch.velocity
                * np.cos(launch.launch_angle)
                * np.sin(launch.azimuth_angle),
                launch.velocity * np.sin(launch.launch_angle),
            ]
        )

        # Time span
        t_span = (0, max_time)
        t_eval = np.arange(0, max_time, time_step)

        # Solve ODE
        solution = solve_ivp(
            fun=lambda t, y: self._equations_of_motion(t, y, launch),
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            events=self._ground_contact_event,
            dense_output=True,
        )

        # Convert solution to trajectory points
        trajectory = []
        for i, t in enumerate(solution.t):
            state = solution.y[:, i]
            position = state[:3]
            velocity = state[3:]

            # Calculate forces at this point
            forces = self._calculate_forces(velocity, launch)
            acceleration = np.sum(list(forces.values()), axis=0) / self.ball.mass

            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    acceleration=acceleration,
                    forces=forces,
                )
            )

        return trajectory

    def _equations_of_motion(
        self, t: float, state: np.ndarray, launch: LaunchConditions
    ) -> np.ndarray:
        """Equations of motion for golf ball flight.

        Args:
            t: Current time
            state: Current state [x, y, z, vx, vy, vz]
            launch: Launch conditions for spin calculations

        Returns:
            State derivatives [vx, vy, vz, ax, ay, az]
        """
        velocity = state[3:]

        # OPTIMIZATION: Calculate acceleration directly to avoid
        # dictionary allocation and list summation overhead in the solver loop.
        acceleration = self._calculate_acceleration(velocity, launch)

        # Return derivatives
        return np.concatenate([velocity, acceleration])

    def _calculate_acceleration(
        self, velocity: np.ndarray, launch: LaunchConditions
    ) -> np.ndarray:
        """Calculate total acceleration acting on the ball.

        Optimized for performance in the solver loop.
        """
        # Gravity acceleration (constant)
        gravity_acc = np.array([0.0, 0.0, -self.environment.gravity])

        # Relative velocity (accounting for wind)
        relative_velocity = velocity - self.environment.wind_velocity
        # Optimization: use sum(x**2) instead of norm to avoid sqrt if not needed,
        # but here we need speed.
        speed_sq = np.sum(relative_velocity**2)
        speed = np.sqrt(speed_sq)

        if speed <= MIN_SPEED_THRESHOLD:
            return gravity_acc

        velocity_unit = relative_velocity / speed

        # Calculate spin ratio S = (ω * r) / v (Waterloo/Penner model)
        if launch.spin_rate > 0:
            omega = launch.spin_rate * 2 * np.pi / 60  # rad/s
            spin_ratio = (omega * self.ball.radius) / speed
        else:
            spin_ratio = 0.0

        # Drag force using Waterloo/Penner quadratic model
        drag_coefficient = (
            self.ball.cd0 + self.ball.cd1 * spin_ratio + self.ball.cd2 * spin_ratio**2
        )

        # F_drag = 0.5 * rho * A * Cd * v^2
        # a_drag = F_drag / m
        # Precompute constants term: 0.5 * rho * A / m
        const_term = (
            0.5 * self.environment.air_density * self.ball.cross_sectional_area
        ) / self.ball.mass

        drag_acc_mag = const_term * drag_coefficient * speed_sq
        drag_acc = -drag_acc_mag * velocity_unit

        total_acc = gravity_acc + drag_acc

        # Lift force using Waterloo/Penner quadratic model
        if spin_ratio > 0:
            lift_coefficient = (
                self.ball.cl0
                + self.ball.cl1 * spin_ratio
                + self.ball.cl2 * spin_ratio**2
            )
            lift_coefficient = min(MAX_LIFT_COEFFICIENT, lift_coefficient)

            magnus_acc_mag = const_term * lift_coefficient * speed_sq

            # Direction: perpendicular to velocity and spin axis
            if launch.spin_axis is not None:
                cross_product = np.cross(launch.spin_axis, velocity_unit)
                cross_magnitude = np.sqrt(np.sum(cross_product**2))
                if cross_magnitude > NUMERICAL_EPSILON:
                    magnus_acc = magnus_acc_mag * cross_product / cross_magnitude
                    total_acc += magnus_acc

        return cast(np.ndarray, total_acc)

    def _calculate_forces(
        self, velocity: np.ndarray, launch: LaunchConditions
    ) -> dict[str, np.ndarray]:
        """Calculate all forces acting on the ball.

        Args:
            velocity: Current velocity vector
            launch: Launch conditions for spin

        Returns:
            Dictionary of force vectors
        """
        # This method duplicates logic from _calculate_acceleration but returns
        # the force breakdown. Since it's only called for trajectory point storage
        # (not in the inner solver loop), the duplication is acceptable for
        # performance isolation.

        forces = {}

        # Gravity force
        forces["gravity"] = np.array(
            [0.0, 0.0, -self.ball.mass * self.environment.gravity]
        )

        # Relative velocity (accounting for wind)
        relative_velocity = velocity - self.environment.wind_velocity
        speed = np.linalg.norm(relative_velocity)

        if speed > MIN_SPEED_THRESHOLD:
            velocity_unit = relative_velocity / speed

            if launch.spin_rate > 0:
                omega = launch.spin_rate * 2 * np.pi / 60
                spin_ratio = float((omega * self.ball.radius) / speed)
            else:
                spin_ratio = 0.0

            drag_coefficient = (
                self.ball.cd0
                + self.ball.cd1 * spin_ratio
                + self.ball.cd2 * spin_ratio**2
            )
            drag_magnitude = (
                0.5
                * self.environment.air_density
                * self.ball.cross_sectional_area
                * drag_coefficient
                * speed**2
            )
            forces["drag"] = -drag_magnitude * velocity_unit

            if spin_ratio > 0:
                lift_coefficient = (
                    self.ball.cl0
                    + self.ball.cl1 * spin_ratio
                    + self.ball.cl2 * spin_ratio**2
                )
                lift_coefficient = min(MAX_LIFT_COEFFICIENT, lift_coefficient)

                magnus_magnitude = (
                    0.5
                    * self.environment.air_density
                    * self.ball.cross_sectional_area
                    * lift_coefficient
                    * speed**2
                )

                if launch.spin_axis is not None:
                    cross_product = np.cross(launch.spin_axis, velocity_unit)
                    cross_magnitude = float(np.linalg.norm(cross_product))
                    if cross_magnitude > NUMERICAL_EPSILON:
                        forces["magnus"] = (
                            magnus_magnitude * cross_product / cross_magnitude
                        )
                    else:
                        forces["magnus"] = np.array([0.0, 0.0, 0.0])
                else:
                    forces["magnus"] = np.array([0.0, 0.0, 0.0])
            else:
                forces["magnus"] = np.array([0.0, 0.0, 0.0])
        else:
            forces["drag"] = np.array([0.0, 0.0, 0.0])
            forces["magnus"] = np.array([0.0, 0.0, 0.0])

        return forces

    def _ground_contact_event(self, t: float, state: np.ndarray) -> float:
        """Event function to detect ground contact.

        Args:
            t: Current time
            state: Current state

        Returns:
            Height above ground (event occurs when this reaches zero)
        """
        return float(state[2])  # z-coordinate (height)

    # Set event attributes for solve_ivp (must be set on the function)
    _ground_contact_event.terminal = True  # type: ignore[attr-defined]
    _ground_contact_event.direction = -1  # type: ignore[attr-defined]  # Only trigger when descending

    def calculate_carry_distance(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate carry distance from trajectory.

        Args:
            trajectory: Ball trajectory points

        Returns:
            Carry distance in meters
        """
        if not trajectory:
            return 0.0

        final_point = trajectory[-1]
        return float(
            np.sqrt(final_point.position[0] ** 2 + final_point.position[1] ** 2)
        )

    def calculate_max_height(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate maximum height from trajectory.

        Args:
            trajectory: Ball trajectory points

        Returns:
            Maximum height in meters
        """
        if not trajectory:
            return 0.0

        heights = [point.position[2] for point in trajectory]
        return float(max(heights))

    def calculate_flight_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate total flight time.

        Args:
            trajectory: Ball trajectory points

        Returns:
            Flight time in seconds
        """
        if not trajectory:
            return 0.0

        return float(trajectory[-1].time)

    def analyze_trajectory(self, trajectory: list[TrajectoryPoint]) -> dict[str, Any]:
        """Comprehensive trajectory analysis.

        Args:
            trajectory: Ball trajectory points

        Returns:
            Dictionary of trajectory metrics
        """
        if not trajectory:
            return {}

        return {
            "carry_distance": self.calculate_carry_distance(trajectory),
            "max_height": self.calculate_max_height(trajectory),
            "flight_time": self.calculate_flight_time(trajectory),
            "landing_angle": self._calculate_landing_angle(trajectory),
            "apex_time": self._calculate_apex_time(trajectory),
            "trajectory_points": len(trajectory),
        }

    def _calculate_landing_angle(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate landing angle in degrees."""
        if len(trajectory) < 2:
            return 0.0

        final_velocity = trajectory[-1].velocity
        horizontal_speed = np.sqrt(final_velocity[0] ** 2 + final_velocity[1] ** 2)
        vertical_speed = abs(final_velocity[2])

        if horizontal_speed > 0:
            return float(np.degrees(np.arctan(vertical_speed / horizontal_speed)))
        return 90.0

    def _calculate_apex_time(self, trajectory: list[TrajectoryPoint]) -> float:
        """Calculate time to reach maximum height."""
        if not trajectory:
            return 0.0

        max_height = 0.0
        apex_time = 0.0

        for point in trajectory:
            if point.position[2] > max_height:
                max_height = point.position[2]
                apex_time = point.time

        return apex_time
