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

        # Precompute constants to avoid attribute access in loop
        gravity_acc = np.array([0.0, 0.0, -self.environment.gravity])
        wind_velocity = self.environment.wind_velocity
        ball_radius = self.ball.radius

        # Aerodynamic constant: 0.5 * rho * A / m
        const_term = (
            0.5 * self.environment.air_density * self.ball.cross_sectional_area
        ) / self.ball.mass

        # Coefficients
        cd0, cd1, cd2 = self.ball.cd0, self.ball.cd1, self.ball.cd2
        cl0, cl1, cl2 = self.ball.cl0, self.ball.cl1, self.ball.cl2

        # Spin parameters
        has_spin = launch.spin_rate > 0
        if has_spin:
            omega = launch.spin_rate * 2 * np.pi / 60  # rad/s
            spin_axis = launch.spin_axis
        else:
            omega = 0.0
            spin_axis = None

        min_speed_sq = MIN_SPEED_THRESHOLD**2

        # Optimized ODE function closure
        # Contains the authoritative physics model for flight dynamics.
        def ode_func(t: float, state: np.ndarray) -> np.ndarray:
            # state is [x, y, z, vx, vy, vz]
            # Velocity view (no copy)
            velocity = state[3:]

            # Relative velocity
            rel_vel = velocity - wind_velocity
            # Optimization: use dot product instead of sum(x**2) to avoid allocation
            speed_sq = rel_vel @ rel_vel

            if speed_sq <= min_speed_sq:
                # If stopped, only gravity acts (or nothing if resting, but here flight)
                acc = gravity_acc
                result = np.empty(6, dtype=state.dtype)
                result[:3] = velocity
                result[3:] = acc
                return result

            speed = np.sqrt(speed_sq)
            vel_unit = rel_vel / speed

            # Spin ratio S = (omega * r) / v
            if has_spin:
                spin_ratio = (omega * ball_radius) / speed
            else:
                spin_ratio = 0.0

            # Drag
            cd = cd0 + spin_ratio * (cd1 + spin_ratio * cd2)
            drag_mag = const_term * cd * speed_sq

            # acc = gravity - drag * unit
            # Initialize acc with gravity
            acc_x = gravity_acc[0] - drag_mag * vel_unit[0]
            acc_y = gravity_acc[1] - drag_mag * vel_unit[1]
            acc_z = gravity_acc[2] - drag_mag * vel_unit[2]

            # Lift (Magnus)
            if has_spin and spin_ratio > 0:
                cl = cl0 + spin_ratio * (cl1 + spin_ratio * cl2)
                if cl > MAX_LIFT_COEFFICIENT:
                    cl = MAX_LIFT_COEFFICIENT

                magnus_mag = const_term * cl * speed_sq

                # Cross product: spin_axis x vel_unit
                if spin_axis is not None:
                    # Manual cross product is faster for small arrays
                    c0 = spin_axis[1] * vel_unit[2] - spin_axis[2] * vel_unit[1]
                    c1 = spin_axis[2] * vel_unit[0] - spin_axis[0] * vel_unit[2]
                    c2 = spin_axis[0] * vel_unit[1] - spin_axis[1] * vel_unit[0]

                    cross_mag_sq = c0 * c0 + c1 * c1 + c2 * c2

                    if cross_mag_sq > NUMERICAL_EPSILON**2:
                        cross_mag = np.sqrt(cross_mag_sq)
                        factor = magnus_mag / cross_mag
                        acc_x += c0 * factor
                        acc_y += c1 * factor
                        acc_z += c2 * factor

            # Construct result
            result = np.empty(6, dtype=state.dtype)
            result[0] = velocity[0]
            result[1] = velocity[1]
            result[2] = velocity[2]
            result[3] = acc_x
            result[4] = acc_y
            result[5] = acc_z
            return result

        # Time span
        t_span = (0, max_time)
        t_eval = np.arange(0, max_time, time_step)

        # Solve ODE
        solution = solve_ivp(
            fun=ode_func,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            events=self._ground_contact_event,
            dense_output=True,
        )

        # Convert solution to trajectory points
        trajectory = []
        # Pre-fetching attributes to local vars for loop speed
        sol_t = solution.t
        sol_y = solution.y
        ball_mass = self.ball.mass

        # Calculate forces for all points at once (vectorized)
        velocities = sol_y[3:, :]
        all_forces = self._calculate_forces(velocities, launch)

        # Pre-calculate accelerations from forces
        # Use np.sum with axis=0 to ensure we sum the arrays correctly
        total_forces = np.sum(list(all_forces.values()), axis=0)  # type: ignore[arg-type]
        all_accelerations = total_forces / ball_mass

        for i in range(len(sol_t)):
            t = sol_t[i]
            state = sol_y[:, i]
            position = state[:3]
            velocity = state[3:]

            # Extract forces for this point from vectorized result
            forces = {k: v[:, i] for k, v in all_forces.items()}
            acceleration = all_accelerations[:, i]

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

    def _calculate_forces(
        self, velocity: np.ndarray, launch: LaunchConditions
    ) -> dict[str, np.ndarray]:
        """Calculate all forces acting on the ball.

        Args:
            velocity: Current velocity vector (3,) or batch (3, N)
            launch: Launch conditions for spin

        Returns:
            Dictionary of force vectors
        """
        # Note: This logic must match ode_func in simulate_trajectory.
        # It is used for diagnostic reporting (TrajectoryPoint.forces).

        is_batch = velocity.ndim == 2
        forces = {}

        # Gravity force
        gravity = np.array([0.0, 0.0, -self.ball.mass * self.environment.gravity])
        if is_batch:
            forces["gravity"] = np.tile(gravity[:, np.newaxis], (1, velocity.shape[1]))
        else:
            forces["gravity"] = gravity

        # Relative velocity (accounting for wind)
        wind = self.environment.wind_velocity
        assert wind is not None
        if is_batch:
            wind = wind[:, np.newaxis]
            axis: int | None = 0
        else:
            axis = None

        relative_velocity = velocity - wind
        speed = np.linalg.norm(relative_velocity, axis=axis)

        # Prepare outputs
        if is_batch:
            drag_force = np.zeros_like(velocity)
            magnus_force = np.zeros_like(velocity)
        else:
            drag_force = np.zeros(3)
            magnus_force = np.zeros(3)

        # Vectorized calculation
        mask = speed > MIN_SPEED_THRESHOLD

        # Handle scalar (single vector) case for mask
        if not is_batch:
            should_compute = mask
        else:
            should_compute = np.any(mask)

        if should_compute:
            if is_batch:
                # Batch processing
                speed_array = cast(np.ndarray, speed)
                speed_masked = speed_array[mask]
                vel_masked = relative_velocity[:, mask]
                velocity_unit = vel_masked / speed_masked

                if launch.spin_rate > 0:
                    omega = launch.spin_rate * 2 * np.pi / 60
                    spin_ratio = (omega * self.ball.radius) / speed_masked
                else:
                    spin_ratio = np.zeros_like(speed_masked)

                # Drag
                drag_coef = (
                    self.ball.cd0
                    + self.ball.cd1 * spin_ratio
                    + self.ball.cd2 * spin_ratio**2
                )
                drag_mag = (
                    0.5
                    * self.environment.air_density
                    * self.ball.cross_sectional_area
                    * drag_coef
                    * speed_masked**2
                )
                drag_force[:, mask] = -drag_mag * velocity_unit

                # Magnus
                if launch.spin_rate > 0:
                    lift_coef = (
                        self.ball.cl0
                        + self.ball.cl1 * spin_ratio
                        + self.ball.cl2 * spin_ratio**2
                    )
                    lift_coef = np.minimum(MAX_LIFT_COEFFICIENT, lift_coef)

                    magnus_mag = (
                        0.5
                        * self.environment.air_density
                        * self.ball.cross_sectional_area
                        * lift_coef
                        * speed_masked**2
                    )

                    if launch.spin_axis is not None:
                        # Cross product of (3,) and (3, M) -> (3, M) with axisa=0, axisb=0, axisc=0
                        cross_prod = np.cross(
                            launch.spin_axis, velocity_unit, axisa=0, axisb=0, axisc=0
                        )
                        cross_mag = np.linalg.norm(cross_prod, axis=0)

                        # Avoid division by zero
                        valid_cross = cross_mag > NUMERICAL_EPSILON
                        factor = np.zeros_like(magnus_mag)
                        factor[valid_cross] = (
                            magnus_mag[valid_cross] / cross_mag[valid_cross]
                        )

                        magnus_force[:, mask] = factor * cross_prod
            else:
                # Single vector processing (original logic)
                velocity_unit = relative_velocity / speed

                if launch.spin_rate > 0:
                    omega = launch.spin_rate * 2 * np.pi / 60
                    s_ratio = float((omega * self.ball.radius) / speed)
                else:
                    s_ratio = 0.0

                drag_coef_scalar = (
                    self.ball.cd0 + self.ball.cd1 * s_ratio + self.ball.cd2 * s_ratio**2
                )
                drag_magnitude = (
                    0.5
                    * self.environment.air_density
                    * self.ball.cross_sectional_area
                    * drag_coef_scalar
                    * speed**2
                )
                drag_force = -drag_magnitude * velocity_unit

                if s_ratio > 0:
                    lift_coef_scalar = (
                        self.ball.cl0
                        + self.ball.cl1 * s_ratio
                        + self.ball.cl2 * s_ratio**2
                    )
                    lift_coef_scalar = min(MAX_LIFT_COEFFICIENT, lift_coef_scalar)

                    magnus_magnitude = (
                        0.5
                        * self.environment.air_density
                        * self.ball.cross_sectional_area
                        * lift_coef_scalar
                        * speed**2
                    )

                    if launch.spin_axis is not None:
                        cross_product = np.cross(launch.spin_axis, velocity_unit)
                        cross_magnitude = float(np.linalg.norm(cross_product))
                        if cross_magnitude > NUMERICAL_EPSILON:
                            magnus_force = (
                                magnus_magnitude * cross_product / cross_magnitude
                            )

        forces["drag"] = drag_force
        forces["magnus"] = magnus_force

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
