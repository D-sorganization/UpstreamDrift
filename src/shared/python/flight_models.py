"""Multi-Model Ball Flight Physics Framework.

This module provides a unified interface for multiple golf ball flight models,
enabling comparison and validation across different physics implementations.

Supported Models:
- Waterloo/Penner: Quadratic coefficient model (Cd, Cl as functions of spin ratio)
- MacDonald-Hanzely (1991): ODE-based model from "The physics of the drive in golf"
- Nathan: Model based on Prof. Alan Nathan's trajectory calculations

All models implement the BallFlightModel protocol for interoperability.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# =============================================================================
# Physical Constants
# =============================================================================

# Golf ball physical properties (USGA regulations)
GOLF_BALL_MASS = 0.0459  # [kg] Maximum mass
GOLF_BALL_DIAMETER = 0.04267  # [m] Minimum diameter
GOLF_BALL_RADIUS = GOLF_BALL_DIAMETER / 2  # [m]
GOLF_BALL_AREA = math.pi * GOLF_BALL_RADIUS**2  # [m²] Cross-sectional area

# Standard atmospheric conditions
STD_AIR_DENSITY = 1.225  # [kg/m³] At sea level, 15°C
STD_GRAVITY = 9.81  # [m/s²]

# Numerical constants
# Minimum speed threshold for aerodynamic calculations [m/s]
# Below this speed, aerodynamic forces are negligible and we avoid division by zero.
MIN_SPEED_THRESHOLD: float = 0.1  # [m/s]

# Small epsilon for numerical stability in vector normalization
NUMERICAL_EPSILON: float = 1e-10


class FlightModelType(Enum):
    """Available ball flight physics models."""

    WATERLOO_PENNER = "waterloo_penner"
    MACDONALD_HANZELY = "macdonald_hanzely"
    NATHAN = "nathan"
    BALLANTYNE = "ballantyne"  # sb362/golf-ball-simulator
    JCOLE = "jcole"  # jcole/golf-shot-simulation
    ROSPIE_DL = "rospie_dl"  # asrospie/golf-flight-model (Physics + DL paper)
    CHARRY_L3 = "charry_l3"  # angelocharry/Projet-L3


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class UnifiedLaunchConditions:
    """Launch conditions for ball flight simulation.

    This is a standardized interface that all flight models accept.
    Uses SI units internally with conversion methods for convenience.
    """

    ball_speed: float  # [m/s] Initial ball speed
    launch_angle: float  # [rad] Vertical launch angle
    azimuth_angle: float = 0.0  # [rad] Horizontal direction (0 = target)
    spin_rate: float = 2500.0  # [rpm] Total spin rate
    spin_axis_angle: float = 0.0  # [rad] Tilt of spin axis (0 = pure backspin)

    # Optional physical properties
    ball_mass: float = GOLF_BALL_MASS  # [kg]
    ball_radius: float = GOLF_BALL_RADIUS  # [m]

    # Environmental conditions
    air_density: float = STD_AIR_DENSITY  # [kg/m³]
    gravity: float = STD_GRAVITY  # [m/s²]
    wind_speed: float = 0.0  # [m/s]
    wind_direction: float = 0.0  # [rad] Direction wind is coming FROM

    @classmethod
    def from_imperial(
        cls,
        ball_speed_mph: float,
        launch_angle_deg: float,
        azimuth_angle_deg: float = 0.0,
        spin_rate_rpm: float = 2500.0,
        spin_axis_angle_deg: float = 0.0,
        wind_speed_mph: float = 0.0,
        wind_direction_deg: float = 0.0,
    ) -> UnifiedLaunchConditions:
        """Create launch conditions from imperial units.

        Args:
            ball_speed_mph: Ball speed in mph
            launch_angle_deg: Launch angle in degrees
            azimuth_angle_deg: Direction in degrees
            spin_rate_rpm: Spin rate in rpm
            spin_axis_angle_deg: Spin axis tilt in degrees
            wind_speed_mph: Wind speed in mph
            wind_direction_deg: Wind direction in degrees

        Returns:
            UnifiedLaunchConditions in SI units
        """
        return cls(
            ball_speed=ball_speed_mph * 0.44704,  # mph to m/s
            launch_angle=math.radians(launch_angle_deg),
            azimuth_angle=math.radians(azimuth_angle_deg),
            spin_rate=spin_rate_rpm,
            spin_axis_angle=math.radians(spin_axis_angle_deg),
            wind_speed=wind_speed_mph * 0.44704,
            wind_direction=math.radians(wind_direction_deg),
        )

    def get_initial_velocity(self) -> np.ndarray:
        """Get initial velocity vector [vx, vy, vz] in m/s."""
        vx = (
            self.ball_speed * math.cos(self.launch_angle) * math.cos(self.azimuth_angle)
        )
        vy = (
            self.ball_speed * math.cos(self.launch_angle) * math.sin(self.azimuth_angle)
        )
        vz = self.ball_speed * math.sin(self.launch_angle)
        return np.array([vx, vy, vz])

    def get_spin_vector(self) -> np.ndarray:
        """Get spin vector [ωx, ωy, ωz] in rad/s.

        Pure backspin has axis along -Y (pointing left when viewed from behind).
        Spin axis angle tilts this toward vertical for sidespin.
        """
        omega = self.spin_rate * 2 * math.pi / 60  # rpm to rad/s
        # Backspin component (axis perpendicular to flight, horizontal)
        backspin = omega * math.cos(self.spin_axis_angle)
        # Sidespin component (axis vertical)
        sidespin = omega * math.sin(self.spin_axis_angle)
        # Spin axis: [-sidespin * sin(azimuth), -backspin, sidespin * cos(azimuth)]
        return np.array(
            [
                sidespin * math.sin(self.azimuth_angle),
                -backspin,
                sidespin * math.cos(self.azimuth_angle),
            ]
        )

    def get_wind_vector(self) -> np.ndarray:
        """Get wind velocity vector [wx, wy, wz] in m/s.

        Wind direction is where the wind is coming FROM.
        A headwind (0°) blows against the target direction (-x).
        A right-to-left crosswind (90°) blows in -y direction.

        Returns:
            Wind velocity vector in m/s [wx, wy, wz]
        """
        if self.wind_speed < 0.01:
            return np.zeros(3)

        # Wind blows FROM the specified direction
        # 0° = headwind (against positive X)
        # 90° = right-to-left crosswind (against positive Y)
        wx = -self.wind_speed * math.cos(self.wind_direction)
        wy = -self.wind_speed * math.sin(self.wind_direction)
        wz = 0.0  # No vertical wind component

        return np.array([wx, wy, wz])


@dataclass
class TrajectoryPoint:
    """Single point in the ball trajectory."""

    time: float  # [s]
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s

    @property
    def speed(self) -> float:
        """Ball speed in m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def height(self) -> float:
        """Height above ground in meters."""
        return float(self.position[2])


@dataclass
class FlightResult:
    """Complete result of a ball flight simulation."""

    trajectory: list[TrajectoryPoint]
    model_name: str

    # Computed metrics (populated after simulation)
    carry_distance: float = 0.0  # [m]
    max_height: float = 0.0  # [m]
    flight_time: float = 0.0  # [s]
    landing_angle: float = 0.0  # [deg]
    lateral_deviation: float = 0.0  # [m] Positive = right

    def to_position_array(self) -> np.ndarray:
        """Get trajectory as Nx3 array of positions."""
        return np.array([p.position for p in self.trajectory])


# =============================================================================
# Ball Flight Model Protocol
# =============================================================================


class BallFlightModel(ABC):
    """Abstract base class for ball flight models.

    All flight models must implement this interface to be used
    in the shot tracer and comparison tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the model."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the model's physics basis."""

    @property
    @abstractmethod
    def reference(self) -> str:
        """Academic reference for the model."""

    @abstractmethod
    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate ball flight trajectory.

        Args:
            launch: Initial launch conditions
            max_time: Maximum simulation time [s]
            dt: Time step for output [s]

        Returns:
            FlightResult containing trajectory and metrics
        """


# =============================================================================
# Waterloo/Penner Model Implementation
# =============================================================================


@dataclass
class WaterlooPennerCoefficients:
    """Coefficients for Waterloo/Penner quadratic model.

    Cd = cd0 + cd1 * S + cd2 * S²
    Cl = cl0 + cl1 * S + cl2 * S²

    where S = (ω * r) / v is the spin ratio.
    """

    cd0: float = 0.21  # Base drag coefficient
    cd1: float = 0.05  # Linear spin dependence
    cd2: float = 0.02  # Quadratic spin dependence
    cl0: float = 0.00  # Base lift (zero at no spin)
    cl1: float = 0.38  # Linear spin dependence
    cl2: float = 0.08  # Quadratic spin dependence
    cl_max: float = 0.25  # Maximum lift coefficient cap


class WaterlooPennerModel(BallFlightModel):
    """Waterloo/Penner quadratic coefficient model.

    Uses Cd and Cl as quadratic functions of spin ratio,
    based on University of Waterloo wind tunnel data.

    References:
        - McPhee et al., University of Waterloo Motion Research Group
        - Penner, A.R. (2003) "The physics of golf"
    """

    def __init__(
        self,
        coefficients: WaterlooPennerCoefficients | None = None,
        enable_wind: bool = True,
    ) -> None:
        """Initialize model with optional custom coefficients.

        Args:
            coefficients: Custom aerodynamic coefficients
            enable_wind: If True, include wind effects in trajectory (default: True)
        """
        self.coefficients = coefficients or WaterlooPennerCoefficients()
        self.enable_wind = enable_wind

    @property
    def name(self) -> str:
        return "Waterloo/Penner"

    @property
    def description(self) -> str:
        return "Quadratic Cd/Cl model from Waterloo wind tunnel data"

    @property
    def reference(self) -> str:
        return "McPhee et al. (Waterloo); Penner (2003)"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using Waterloo/Penner model."""
        # Initial state: [x, y, z, vx, vy, vz]
        v0 = launch.get_initial_velocity()
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

        # Spin vector (constant during flight)
        omega_vec = launch.get_spin_vector()
        omega_mag = np.linalg.norm(omega_vec)

        # Wind vector (constant)
        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            """Compute derivatives for ODE solver."""
            vel = y[3:6]

            # Relative velocity (velocity minus wind) for aerodynamic forces
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            # Use relative velocity for aerodynamic forces
            vel_unit = vel_rel / speed

            # Spin ratio S = (ω * r) / v
            spin_ratio = (omega_mag * launch.ball_radius) / speed

            # Drag coefficient: Cd = cd0 + cd1*S + cd2*S²
            c = self.coefficients
            cd = c.cd0 + c.cd1 * spin_ratio + c.cd2 * spin_ratio**2

            # Lift coefficient: Cl = cl0 + cl1*S + cl2*S² (capped)
            cl = min(
                c.cl_max, float(c.cl0 + c.cl1 * spin_ratio + c.cl2 * spin_ratio**2)
            )

            # Drag force: F_d = -0.5 * ρ * v² * Cd * A * v_hat
            area = math.pi * launch.ball_radius**2
            drag_mag = 0.5 * launch.air_density * speed**2 * cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # Magnus force: F_m = 0.5 * ρ * v² * Cl * A * (ω × v) / |ω × v|
            if omega_mag > 0:
                cross = np.cross(omega_vec / omega_mag, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    magnus_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = magnus_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            # Total acceleration
            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])

            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])  # z position

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        # Solve ODE
        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        # Sample trajectory at regular intervals
        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        # Add final point
        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])

        # Carry distance (horizontal)
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)

        # Max height
        max_height = float(np.max(positions[:, 2]))

        # Flight time
        flight_time = trajectory[-1].time

        # Landing angle (descent angle)
        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        # Lateral deviation
        lateral = float(positions[-1, 1])

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = lateral

        return result


# =============================================================================
# MacDonald-Hanzely Model Implementation
# =============================================================================


class MacDonaldHanzelyModel(BallFlightModel):
    """MacDonald-Hanzely (1991) golf ball flight model.

    Based on the paper "The physics of the drive in golf" from
    American Journal of Physics 59(3):213-218.

    This model uses fixed drag and lift coefficients with
    an exponential decay model for spin.

    References:
        - MacDonald & Hanzely (1991) Am. J. Phys. 59(3):213-218
    """

    def __init__(
        self,
        cd: float = 0.225,  # Drag coefficient
        cl: float = 0.20,  # Lift coefficient (per unit spin parameter)
        spin_decay_rate: float = 0.05,  # Spin decay constant [1/s]
        enable_wind: bool = True,  # Apply wind effects
    ) -> None:
        """Initialize MacDonald-Hanzely model.

        Args:
            cd: Drag coefficient (constant)
            cl: Lift coefficient scaling
            spin_decay_rate: Exponential spin decay rate
            enable_wind: If True, include wind in calculations
        """
        self.cd = cd
        self.cl = cl
        self.spin_decay_rate = spin_decay_rate
        self.enable_wind = enable_wind

    @property
    def name(self) -> str:
        return "MacDonald-Hanzely"

    @property
    def description(self) -> str:
        return "ODE model from 'The physics of the drive in golf' (1991)"

    @property
    def reference(self) -> str:
        return "MacDonald & Hanzely (1991) Am. J. Phys. 59(3):213-218"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using MacDonald-Hanzely model."""
        v0 = launch.get_initial_velocity()
        omega_initial = launch.spin_rate * 2 * math.pi / 60  # rad/s
        spin_axis = launch.get_spin_vector()
        if np.linalg.norm(spin_axis) > 0:
            spin_axis = spin_axis / np.linalg.norm(spin_axis)
        else:
            spin_axis = np.array([0, -1, 0])

        # State: [x, y, z, vx, vy, vz]
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

        area = math.pi * launch.ball_radius**2
        k_drag = 0.5 * launch.air_density * area * self.cd / launch.ball_mass

        # Wind vector
        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec  # Relative velocity for aero forces
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Drag acceleration: a_d = -k * v² * v_hat
            drag_accel = -k_drag * speed**2 * vel_unit

            # Spin decays exponentially
            omega = omega_initial * math.exp(-self.spin_decay_rate * t)

            # Magnus acceleration: proportional to ω × v
            if omega > 0:
                # Spin ratio
                spin_ratio = (omega * launch.ball_radius) / speed
                cl_effective = self.cl * spin_ratio

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    k_lift = 0.5 * launch.air_density * area * cl_effective
                    magnus_accel = (
                        k_lift * speed**2 / launch.ball_mass * cross / cross_mag
                    )
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        # Sample trajectory
        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# Nathan Model Implementation (libgolf-inspired)
# =============================================================================


class NathanModel(BallFlightModel):
    """Nathan model for golf ball trajectory.

    Based on Prof. Alan Nathan's trajectory calculation methodology
    (University of Illinois Urbana-Champaign), as used in libgolf.

    This model uses Reynolds-number dependent drag and a linear
    lift coefficient model.

    References:
        - Prof. Alan Nathan, UIUC
        - gdifiore/libgolf implementation
    """

    def __init__(
        self,
        cd_base: float = 0.23,  # Base drag at critical Re
        cd_low_re: float = 0.50,  # Drag below critical Re
        re_critical: float = 1.5e5,  # Critical Reynolds number
        cl_slope: float = 0.35,  # Cl slope with spin parameter
        enable_wind: bool = True,  # Apply wind effects
        spin_decay_rate: float | None = None,  # Optional spin decay [1/s]
    ) -> None:
        """Initialize Nathan model with Reynolds-dependent drag.

        Args:
            cd_base: Drag coefficient above critical Reynolds number
            cd_low_re: Drag coefficient below critical Reynolds number
            re_critical: Critical Reynolds number for drag transition
            cl_slope: Linear slope of lift coefficient with spin parameter
            enable_wind: If True, include wind in calculations
            spin_decay_rate: If provided, apply exponential spin decay
        """
        self.cd_base = cd_base
        self.cd_low_re = cd_low_re
        self.re_critical = re_critical
        self.cl_slope = cl_slope
        self.enable_wind = enable_wind
        self.spin_decay_rate = spin_decay_rate
        # Air kinematic viscosity at 15°C [m²/s]
        self.nu = 1.48e-5

    @property
    def name(self) -> str:
        return "Nathan"

    @property
    def description(self) -> str:
        return "Reynolds-dependent model (Prof. Nathan, UIUC)"

    @property
    def reference(self) -> str:
        return "Prof. Alan Nathan, UIUC; libgolf implementation"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using Nathan model."""
        v0 = launch.get_initial_velocity()
        omega_vec = launch.get_spin_vector()
        omega_mag_initial = np.linalg.norm(omega_vec)
        spin_axis = (
            omega_vec / omega_mag_initial
            if omega_mag_initial > 0
            else np.array([0, -1, 0])
        )

        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        area = math.pi * launch.ball_radius**2
        diameter = 2 * launch.ball_radius

        # Wind vector
        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Reynolds number
            re = speed * diameter / self.nu

            # Drag coefficient: transitions at critical Re
            if re > self.re_critical:
                cd = self.cd_base
            else:
                # Smooth transition
                cd = float(
                    self.cd_low_re
                    - (self.cd_low_re - self.cd_base) * (re / self.re_critical)
                )

            # Drag acceleration
            drag_mag = 0.5 * launch.air_density * speed**2 * cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # Spin decay (optional)
            if self.spin_decay_rate is not None:
                omega_mag = omega_mag_initial * math.exp(-self.spin_decay_rate * t)
            else:
                omega_mag = omega_mag_initial

            # Lift coefficient: linear with spin parameter
            if omega_mag > 0:
                spin_ratio = (omega_mag * launch.ball_radius) / speed
                cl = self.cl_slope * spin_ratio

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    lift_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = lift_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# Ballantyne Model (sb362/golf-ball-simulator)
# =============================================================================


class BallantyneModel(BallFlightModel):
    """Ballantyne model from sb362/golf-ball-simulator.

    Based on a Physics group discovery project. Uses constant drag
    coefficient with spin-dependent Magnus effect.

    References:
        - sb362/golf-ball-simulator (GitHub)
        - s-ballantyne/gdp original project
    """

    def __init__(
        self,
        cd: float = 0.24,  # Fixed drag coefficient
        cl_factor: float = 0.22,  # Lift coefficient factor
        enable_wind: bool = True,
        spin_decay_rate: float | None = None,
    ) -> None:
        """Initialize Ballantyne model.

        Args:
            cd: Constant drag coefficient
            cl_factor: Lift coefficient per unit spin ratio
            enable_wind: If True, include wind in calculations
            spin_decay_rate: If provided, apply exponential spin decay
        """
        self.cd = cd
        self.cl_factor = cl_factor
        self.enable_wind = enable_wind
        self.spin_decay_rate = spin_decay_rate

    @property
    def name(self) -> str:
        return "Ballantyne"

    @property
    def description(self) -> str:
        return "Physics discovery project model (constant Cd)"

    @property
    def reference(self) -> str:
        return "sb362/golf-ball-simulator"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using Ballantyne model."""
        v0 = launch.get_initial_velocity()
        omega_vec = launch.get_spin_vector()
        omega_mag_initial = np.linalg.norm(omega_vec)
        spin_axis = (
            omega_vec / omega_mag_initial
            if omega_mag_initial > 0
            else np.array([0, -1, 0])
        )

        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        area = math.pi * launch.ball_radius**2

        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Constant drag coefficient
            drag_mag = 0.5 * launch.air_density * speed**2 * self.cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # Spin decay
            if self.spin_decay_rate is not None:
                omega_mag = omega_mag_initial * math.exp(-self.spin_decay_rate * t)
            else:
                omega_mag = omega_mag_initial

            # Magnus effect
            if omega_mag > 0:
                spin_ratio = (omega_mag * launch.ball_radius) / speed
                cl = self.cl_factor * spin_ratio

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    lift_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = lift_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# JCole Model (jcole/golf-shot-simulation)
# =============================================================================


class JColeModel(BallFlightModel):
    """JCole model from jcole/golf-shot-simulation.

    JavaScript three.js visualization converted to Python.
    Uses simple drag and lift with Euler-like integration.
    Inspired by TrackMan data validation approach.

    References:
        - jcole/golf-shot-simulation (GitHub)
    """

    def __init__(
        self,
        cd: float = 0.28,  # Drag coefficient
        cl_max: float = 0.18,  # Maximum lift coefficient
        enable_wind: bool = True,
        spin_decay_rate: float | None = None,
    ) -> None:
        """Initialize JCole model.

        Args:
            cd: Drag coefficient
            cl_max: Maximum lift coefficient
            enable_wind: If True, include wind in calculations
            spin_decay_rate: If provided, apply exponential spin decay
        """
        self.cd = cd
        self.cl_max = cl_max
        self.enable_wind = enable_wind
        self.spin_decay_rate = spin_decay_rate

    @property
    def name(self) -> str:
        return "JCole"

    @property
    def description(self) -> str:
        return "three.js visualization model (TrackMan-validated)"

    @property
    def reference(self) -> str:
        return "jcole/golf-shot-simulation"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using JCole model."""
        v0 = launch.get_initial_velocity()
        omega_vec = launch.get_spin_vector()
        omega_mag_initial = np.linalg.norm(omega_vec)
        spin_axis = (
            omega_vec / omega_mag_initial
            if omega_mag_initial > 0
            else np.array([0, -1, 0])
        )

        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        area = math.pi * launch.ball_radius**2

        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Simple drag model
            drag_mag = 0.5 * launch.air_density * speed**2 * self.cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # Spin decay
            if self.spin_decay_rate is not None:
                omega_mag = omega_mag_initial * math.exp(-self.spin_decay_rate * t)
            else:
                omega_mag = omega_mag_initial

            # Simple lift model (proportional to spin, capped)
            if omega_mag > 0:
                spin_param = (omega_mag * launch.ball_radius) / speed
                cl = min(self.cl_max, float(0.5 * spin_param))

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    lift_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = lift_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# Rospie Deep Learning Model (asrospie/golf-flight-model)
# =============================================================================


class RospieDLModel(BallFlightModel):
    """Rospie model from asrospie/golf-flight-model.

    Based on the paper "Combining Physics and Deep Learning Models
    to Simulate the Flight of a Golf Ball".

    Uses physics-informed coefficients with enhanced accuracy
    from deep learning training.

    References:
        - asrospie/golf-flight-model (GitHub)
        - "Combining Physics and Deep Learning Models" paper
    """

    def __init__(
        self,
        cd_base: float = 0.22,
        cd_spin: float = 0.08,
        cl_base: float = 0.02,
        cl_spin: float = 0.42,
        enable_wind: bool = True,
        spin_decay_rate: float | None = None,
    ) -> None:
        """Initialize Rospie DL model.

        Args:
            cd_base: Base drag coefficient
            cd_spin: Spin-dependent drag component
            cl_base: Base lift coefficient
            cl_spin: Spin-dependent lift component
            enable_wind: If True, include wind in calculations
            spin_decay_rate: If provided, apply exponential spin decay
        """
        self.cd_base = cd_base
        self.cd_spin = cd_spin
        self.cl_base = cl_base
        self.cl_spin = cl_spin
        self.enable_wind = enable_wind
        self.spin_decay_rate = spin_decay_rate

    @property
    def name(self) -> str:
        return "Rospie-DL"

    @property
    def description(self) -> str:
        return "Physics + Deep Learning hybrid model"

    @property
    def reference(self) -> str:
        return "asrospie/golf-flight-model; Physics+DL Paper"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using Rospie DL model."""
        v0 = launch.get_initial_velocity()
        omega_vec = launch.get_spin_vector()
        omega_mag_initial = np.linalg.norm(omega_vec)
        spin_axis = (
            omega_vec / omega_mag_initial
            if omega_mag_initial > 0
            else np.array([0, -1, 0])
        )

        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        area = math.pi * launch.ball_radius**2

        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Spin decay
            if self.spin_decay_rate is not None:
                omega_mag = omega_mag_initial * math.exp(-self.spin_decay_rate * t)
            else:
                omega_mag = omega_mag_initial

            spin_ratio = float(
                (omega_mag * launch.ball_radius) / speed if omega_mag > 0 else 0.0
            )

            # DL-enhanced drag: Cd = cd_base + cd_spin * S
            cd = self.cd_base + self.cd_spin * spin_ratio
            drag_mag = 0.5 * launch.air_density * speed**2 * cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # DL-enhanced lift: Cl = cl_base + cl_spin * S
            if omega_mag > 0:
                cl = float(self.cl_base + self.cl_spin * spin_ratio)
                cl = min(0.30, cl)  # Cap from DL training

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    lift_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = lift_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# Charry L3 Model (angelocharry/Projet-L3)
# =============================================================================


class CharryL3Model(BallFlightModel):
    """Charry L3 model from angelocharry/Projet-L3.

    French L3 Physics numerical project (Paul Sabatier University).
    Models golf ball dynamics with bounce and slip on 3D surfaces.
    Includes genetic algorithm trajectory optimization.

    This implementation focuses on the flight phase.

    References:
        - angelocharry/Projet-L3 (GitHub)
        - Rapport.pdf in the repository
    """

    def __init__(
        self,
        cd: float = 0.25,  # Drag coefficient
        cl_factor: float = 0.28,  # Lift factor
        spin_decay: float = 0.02,  # Spin decay rate [1/s]
        enable_wind: bool = True,
    ) -> None:
        """Initialize Charry L3 model.

        Args:
            cd: Drag coefficient
            cl_factor: Lift coefficient scaling
            spin_decay: Exponential spin decay rate
            enable_wind: If True, include wind in calculations
        """
        self.cd = cd
        self.cl_factor = cl_factor
        self.spin_decay = spin_decay
        self.enable_wind = enable_wind

    @property
    def name(self) -> str:
        return "Charry-L3"

    @property
    def description(self) -> str:
        return "French L3 physics project (with spin decay)"

    @property
    def reference(self) -> str:
        return "angelocharry/Projet-L3 (Univ. Paul Sabatier)"

    def simulate(
        self,
        launch: UnifiedLaunchConditions,
        max_time: float = 10.0,
        dt: float = 0.01,
    ) -> FlightResult:
        """Simulate trajectory using Charry L3 model."""
        v0 = launch.get_initial_velocity()
        omega_initial = np.linalg.norm(launch.get_spin_vector())
        spin_axis = launch.get_spin_vector()
        if omega_initial > 0:
            spin_axis = spin_axis / omega_initial
        else:
            spin_axis = np.array([0, -1, 0])

        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        area = math.pi * launch.ball_radius**2

        wind_vec = launch.get_wind_vector() if self.enable_wind else np.zeros(3)

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            vel = y[3:6]
            vel_rel = vel - wind_vec
            speed = np.linalg.norm(vel_rel)

            if speed < MIN_SPEED_THRESHOLD:
                return np.array([vel[0], vel[1], vel[2], 0, 0, -launch.gravity])

            vel_unit = vel_rel / speed

            # Drag
            drag_mag = 0.5 * launch.air_density * speed**2 * self.cd * area
            drag_accel = -drag_mag / launch.ball_mass * vel_unit

            # Decaying spin (French model characteristic)
            omega = omega_initial * math.exp(-self.spin_decay * t)

            # Magnus with decay
            if omega > 0:
                spin_ratio = (omega * launch.ball_radius) / speed
                cl = self.cl_factor * spin_ratio

                cross = np.cross(spin_axis, vel_unit)
                cross_mag = np.linalg.norm(cross)
                if cross_mag > NUMERICAL_EPSILON:
                    lift_mag = 0.5 * launch.air_density * speed**2 * cl * area
                    magnus_accel = lift_mag / launch.ball_mass * cross / cross_mag
                else:
                    magnus_accel = np.zeros(3)
            else:
                magnus_accel = np.zeros(3)

            accel = drag_accel + magnus_accel + np.array([0, 0, -launch.gravity])
            return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]])

        def ground_event(t: float, y: np.ndarray) -> float:
            return float(y[2])

        ground_event.terminal = True  # type: ignore[attr-defined]
        ground_event.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_event,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        trajectory: list[TrajectoryPoint] = []

        for t in t_eval:
            y = sol.sol(t)
            trajectory.append(
                TrajectoryPoint(
                    time=t,
                    position=np.array([y[0], y[1], y[2]]),
                    velocity=np.array([y[3], y[4], y[5]]),
                )
            )

        if len(sol.t) > 0:
            y_final = sol.y[:, -1]
            trajectory.append(
                TrajectoryPoint(
                    time=sol.t[-1],
                    position=np.array([y_final[0], y_final[1], y_final[2]]),
                    velocity=np.array([y_final[3], y_final[4], y_final[5]]),
                )
            )

        return self._compute_metrics(trajectory)

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Compute flight metrics from trajectory."""
        if not trajectory:
            return FlightResult([], self.name)

        positions = np.array([p.position for p in trajectory])
        carry = math.sqrt(positions[-1, 0] ** 2 + positions[-1, 1] ** 2)
        max_height = float(np.max(positions[:, 2]))
        flight_time = trajectory[-1].time

        if len(trajectory) >= 2:
            final_vel = trajectory[-1].velocity
            horizontal_speed = math.sqrt(final_vel[0] ** 2 + final_vel[1] ** 2)
            if horizontal_speed > MIN_SPEED_THRESHOLD:
                landing_angle = math.degrees(
                    math.atan2(-final_vel[2], horizontal_speed)
                )
            else:
                landing_angle = 90.0
        else:
            landing_angle = 0.0

        result = FlightResult(trajectory, self.name)
        result.carry_distance = carry
        result.max_height = max_height
        result.flight_time = flight_time
        result.landing_angle = landing_angle
        result.lateral_deviation = float(positions[-1, 1])

        return result


# =============================================================================
# Model Factory and Registry
# =============================================================================


class FlightModelRegistry:
    """Registry for available ball flight models.

    Provides factory methods and model enumeration.
    """

    _models: dict[FlightModelType, type[BallFlightModel]] = {
        FlightModelType.WATERLOO_PENNER: WaterlooPennerModel,
        FlightModelType.MACDONALD_HANZELY: MacDonaldHanzelyModel,
        FlightModelType.NATHAN: NathanModel,
        FlightModelType.BALLANTYNE: BallantyneModel,
        FlightModelType.JCOLE: JColeModel,
        FlightModelType.ROSPIE_DL: RospieDLModel,
        FlightModelType.CHARRY_L3: CharryL3Model,
    }

    @classmethod
    def get_model(cls, model_type: FlightModelType) -> BallFlightModel:
        """Get an instance of the specified model type.

        Args:
            model_type: Type of model to create

        Returns:
            Ball flight model instance
        """
        model_class = cls._models.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class()

    @classmethod
    def get_all_models(cls) -> list[BallFlightModel]:
        """Get instances of all available models.

        Returns:
            List of all ball flight model instances
        """
        return [model_class() for model_class in cls._models.values()]

    @classmethod
    def list_models(cls) -> list[tuple[str, str, str]]:
        """List available models with their names and descriptions.

        Returns:
            List of (enum_value, name, description) tuples
        """
        result = []
        for model_type, model_class in cls._models.items():
            model = model_class()
            result.append((model_type.value, model.name, model.description))
        return result


def compare_models(
    launch: UnifiedLaunchConditions,
    models: list[BallFlightModel] | None = None,
) -> dict[str, FlightResult]:
    """Run all models on the same launch conditions for comparison.

    Args:
        launch: Launch conditions to simulate
        models: List of models to compare (default: all)

    Returns:
        Dictionary mapping model name to FlightResult
    """
    if models is None:
        models = FlightModelRegistry.get_all_models()

    results = {}
    for model in models:
        try:
            result = model.simulate(launch)
            results[model.name] = result
        except Exception as e:
            logger.exception(f"Model {model.name} failed: {e}")

    return results


def print_comparison(results: dict[str, FlightResult]) -> None:
    """Print formatted comparison of model results.

    Args:
        results: Dictionary of FlightResult by model name
    """
    logger.info("\n" + "=" * 70)
    logger.info("Model Comparison Results")
    logger.info("=" * 70)
    logger.info(
        f"{'Model':<20} {'Carry (yd)':<12} {'Height (m)':<12} "
        f"{'Time (s)':<10} {'Landing (°)':<10}"
    )
    logger.info("-" * 70)

    for name, result in results.items():
        carry_yd = result.carry_distance * 1.09361
        logger.info(
            f"{name:<20} {carry_yd:<12.1f} {result.max_height:<12.1f} "
            f"{result.flight_time:<10.2f} {result.landing_angle:<10.1f}"
        )

    logger.info("=" * 70)
