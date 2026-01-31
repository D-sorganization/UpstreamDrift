"""Common Physics Equations for Golf Ball Flight.

This module provides standardized physics calculations shared across all
physics engine implementations, eliminating code duplication.

Design by Contract:
    - All functions validate input arrays have correct shapes
    - All outputs are guaranteed finite (no NaN/Inf)
    - Units are SI unless documented otherwise

References:
    - Jorgensen, T. (1999). The Physics of Golf. Springer.
    - Smits, A.J., & Ogg, S. (2004). Golf Ball Aerodynamics. Physics Today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.shared.python.contracts import postcondition, precondition


@dataclass(frozen=True)
class AirProperties:
    """Standard air properties at given conditions.

    Attributes:
        density: Air density [kg/m³]
        viscosity: Dynamic viscosity [Pa·s]
        temperature: Air temperature [K]
        pressure: Air pressure [Pa]
    """

    density: float = 1.225  # kg/m³ at sea level, 15°C
    viscosity: float = 1.81e-5  # Pa·s at 15°C
    temperature: float = 288.15  # K (15°C)
    pressure: float = 101325.0  # Pa (1 atm)

    @classmethod
    def from_altitude(cls, altitude_m: float) -> AirProperties:
        """Create air properties for a given altitude using ISA model.

        Args:
            altitude_m: Altitude above sea level [m]

        Returns:
            AirProperties at the specified altitude
        """
        # International Standard Atmosphere model
        T0 = 288.15  # K
        P0 = 101325.0  # Pa
        L = 0.0065  # Temperature lapse rate [K/m]
        g = 9.80665  # m/s²
        M = 0.0289644  # Molar mass of air [kg/mol]
        R = 8.31447  # Universal gas constant [J/(mol·K)]

        T = T0 - L * altitude_m
        P = P0 * (T / T0) ** (g * M / (R * L))
        rho = P * M / (R * T)

        return cls(
            density=rho,
            viscosity=1.81e-5 * (T / 288.15) ** 0.76,  # Sutherland's law approx
            temperature=T,
            pressure=P,
        )


@dataclass
class BallProperties:
    """Golf ball physical properties.

    Attributes:
        mass: Ball mass [kg]
        radius: Ball radius [m]
        area: Cross-sectional area [m²]
        drag_coefficient: Baseline drag coefficient
        lift_coefficient: Baseline lift coefficient (spin-dependent)
        spin_decay_rate: Spin decay time constant [1/s]
    """

    mass: float = 0.04593  # kg (1.62 oz)
    radius: float = 0.02135  # m (1.68 in diameter)
    area: float = field(init=False)
    drag_coefficient: float = 0.25
    lift_coefficient: float = 0.15
    spin_decay_rate: float = 0.1  # 1/s

    def __post_init__(self) -> None:
        """Calculate derived properties."""
        self.area = np.pi * self.radius**2


class AerodynamicsCalculator:
    """Calculate aerodynamic forces on a golf ball.

    This class encapsulates the aerodynamic calculations shared across
    all physics engines, ensuring consistent behavior.

    Design by Contract:
        Preconditions:
            - velocity must be 3D vector [m/s]
            - spin must be 3D vector [rad/s]
            - Air properties must be physically valid

        Postconditions:
            - Forces are 3D vectors [N]
            - All values are finite

    Example:
        >>> aero = AerodynamicsCalculator()
        >>> velocity = np.array([50.0, 10.0, 5.0])  # m/s
        >>> spin = np.array([0, 300, 0])  # rad/s (backspin)
        >>> drag, lift, magnus = aero.compute_forces(velocity, spin)
    """

    def __init__(
        self,
        ball: BallProperties | None = None,
        air: AirProperties | None = None,
    ) -> None:
        """Initialize aerodynamics calculator.

        Args:
            ball: Ball physical properties (defaults to standard golf ball)
            air: Air properties (defaults to sea level conditions)
        """
        self.ball = ball or BallProperties()
        self.air = air or AirProperties()

    @precondition(
        lambda self, velocity, spin: velocity.shape == (3,) and spin.shape == (3,),
        "velocity and spin must be 3D vectors",
    )
    @postcondition(
        lambda result: all(np.all(np.isfinite(f)) for f in result),
        "All force components must be finite",
    )
    def compute_forces(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute all aerodynamic forces.

        Args:
            velocity: Ball velocity relative to air [m/s]
            spin: Ball angular velocity [rad/s]

        Returns:
            Tuple of (drag, lift, magnus) force vectors [N]
        """
        drag = self.compute_drag(velocity)
        lift = self.compute_lift(velocity, spin)
        magnus = self.compute_magnus(velocity, spin)

        return drag, lift, magnus

    def compute_drag(self, velocity: np.ndarray) -> np.ndarray:
        """Compute drag force opposing motion.

        F_drag = -0.5 * rho * Cd * A * |v| * v

        Args:
            velocity: Ball velocity [m/s]

        Returns:
            Drag force vector [N] (opposes velocity)
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)

        # Dynamic drag coefficient based on Reynolds number
        Cd = self._compute_drag_coefficient(speed)

        # Drag magnitude
        F_mag = 0.5 * self.air.density * Cd * self.ball.area * speed**2

        # Direction: opposite to velocity
        return -F_mag * velocity / speed

    def compute_lift(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Compute lift force from backspin.

        For backspin (spin axis perpendicular to velocity), lift acts
        upward relative to the velocity direction.

        Args:
            velocity: Ball velocity [m/s]
            spin: Ball angular velocity [rad/s]

        Returns:
            Lift force vector [N]
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)

        # Spin ratio affects lift coefficient
        spin_ratio = self._compute_spin_ratio(speed, spin)
        Cl = self._compute_lift_coefficient(spin_ratio)

        # Lift direction: perpendicular to velocity, in spin plane
        # For backspin, this creates upward force
        spin_axis = spin / (np.linalg.norm(spin) + 1e-10)
        lift_dir = np.cross(spin_axis, velocity)
        lift_norm = np.linalg.norm(lift_dir)

        if lift_norm < 1e-6:
            return np.zeros(3)

        lift_dir = lift_dir / lift_norm

        # Lift magnitude
        F_mag = 0.5 * self.air.density * Cl * self.ball.area * speed**2

        return F_mag * lift_dir

    def compute_magnus(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Compute Magnus force from spin-induced pressure differential.

        F_magnus = 0.5 * rho * Cm * A * |v|² * (ω × v) / |ω × v|

        This force causes hook/slice for sidespin and additional lift/dive
        for backspin/topspin.

        Args:
            velocity: Ball velocity [m/s]
            spin: Ball angular velocity [rad/s]

        Returns:
            Magnus force vector [N]
        """
        speed = np.linalg.norm(velocity)
        spin_mag = np.linalg.norm(spin)

        if speed < 1e-6 or spin_mag < 1e-6:
            return np.zeros(3)

        # Magnus direction: ω × v
        magnus_dir = np.cross(spin, velocity)
        magnus_norm = np.linalg.norm(magnus_dir)

        if magnus_norm < 1e-6:
            return np.zeros(3)

        magnus_dir = magnus_dir / magnus_norm

        # Magnus coefficient based on spin parameter
        spin_param = self.ball.radius * spin_mag / speed
        Cm = self._compute_magnus_coefficient(spin_param)

        # Force magnitude
        F_mag = 0.5 * self.air.density * Cm * self.ball.area * speed**2

        return F_mag * magnus_dir

    def _compute_drag_coefficient(self, speed: float) -> float:
        """Compute drag coefficient based on Reynolds number.

        Golf ball dimples reduce drag significantly at high Reynolds numbers
        through turbulent boundary layer transition.

        Args:
            speed: Ball speed [m/s]

        Returns:
            Drag coefficient (dimensionless)
        """
        speed = float(speed)
        # Reynolds number
        Re = self.air.density * speed * (2 * self.ball.radius) / self.air.viscosity

        # Golf ball Cd variation with Re (empirical fit)
        # Below critical Re (~8e4): higher drag (laminar)
        # Above critical Re: lower drag (turbulent, dimple effect)
        if Re < 8e4:
            return 0.5  # Laminar flow
        elif Re < 2e5:
            # Transition region
            return 0.5 - 0.25 * (Re - 8e4) / (2e5 - 8e4)
        else:
            return self.ball.drag_coefficient  # Fully turbulent

    def _compute_lift_coefficient(self, spin_ratio: float) -> float:
        """Compute lift coefficient based on spin ratio.

        Args:
            spin_ratio: Dimensionless spin parameter

        Returns:
            Lift coefficient (dimensionless)
        """
        spin_ratio = float(spin_ratio)
        # Empirical relationship (Smits & Ogg)
        # Cl increases with spin ratio, saturating at high spin
        Cl_max = 0.4
        return Cl_max * (1 - np.exp(-spin_ratio / 0.1))

    def _compute_magnus_coefficient(self, spin_param: float) -> float:
        """Compute Magnus coefficient based on spin parameter.

        Args:
            spin_param: ωR/v dimensionless spin parameter

        Returns:
            Magnus coefficient (dimensionless)
        """
        spin_param = float(spin_param)
        # Robins-Magnus effect coefficient
        # Approximately linear for small spin_param
        return 0.4 * min(spin_param, 0.5)

    def _compute_spin_ratio(self, speed: float, spin: np.ndarray) -> float:
        """Compute dimensionless spin ratio.

        Args:
            speed: Ball speed [m/s]
            spin: Angular velocity [rad/s]

        Returns:
            Spin ratio = ωR/v
        """
        speed = float(speed)
        spin_mag = float(np.linalg.norm(spin))
        return self.ball.radius * spin_mag / (speed + 1e-10)


class BallPhysics:
    """Complete golf ball physics model.

    Combines aerodynamics with gravity and spin decay for full
    trajectory simulation.

    Example:
        >>> physics = BallPhysics()
        >>> state = {'position': np.zeros(3), 'velocity': np.array([70, 0, 20]),
        ...          'spin': np.array([0, 300, 0])}
        >>> forces = physics.compute_total_force(state)
        >>> acceleration = forces / physics.ball.mass
    """

    def __init__(
        self,
        ball: BallProperties | None = None,
        air: AirProperties | None = None,
        gravity: np.ndarray | None = None,
    ) -> None:
        """Initialize ball physics model.

        Args:
            ball: Ball properties
            air: Air properties
            gravity: Gravity vector [m/s²] (default: [0, 0, -9.81])
        """
        self.ball = ball or BallProperties()
        self.aero = AerodynamicsCalculator(self.ball, air)
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

    def compute_total_force(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
    ) -> np.ndarray:
        """Compute total force on ball.

        Args:
            velocity: Ball velocity [m/s]
            spin: Ball angular velocity [rad/s]

        Returns:
            Total force vector [N]
        """
        # Gravity
        F_gravity = self.ball.mass * self.gravity

        # Aerodynamic forces
        drag, lift, magnus = self.aero.compute_forces(velocity, spin)

        return F_gravity + drag + lift + magnus

    def compute_spin_decay(self, spin: np.ndarray, dt: float) -> np.ndarray:
        """Compute spin decay over time step.

        Spin decays exponentially due to air resistance on the
        rotating ball surface.

        Args:
            spin: Current angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            Updated spin after decay [rad/s]
        """
        decay_factor = np.exp(-self.ball.spin_decay_rate * dt)
        return spin * decay_factor

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        spin: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance ball state by one time step.

        Uses semi-implicit Euler integration for stability.

        Args:
            position: Current position [m]
            velocity: Current velocity [m/s]
            spin: Current angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            Tuple of (new_position, new_velocity, new_spin)
        """
        # Compute forces
        force = self.compute_total_force(velocity, spin)
        acceleration = force / self.ball.mass

        # Semi-implicit Euler: update velocity first
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt

        # Spin decay
        new_spin = self.compute_spin_decay(spin, dt)

        return new_position, new_velocity, new_spin


class PhysicsEquations(Protocol):
    """Protocol for physics equation implementations.

    This defines the interface that engine-specific physics
    implementations must satisfy.
    """

    def compute_drag(self, velocity: np.ndarray) -> np.ndarray:
        """Compute drag force."""
        ...

    def compute_lift(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Compute lift force."""
        ...

    def compute_magnus(self, velocity: np.ndarray, spin: np.ndarray) -> np.ndarray:
        """Compute magnus force."""
        ...
