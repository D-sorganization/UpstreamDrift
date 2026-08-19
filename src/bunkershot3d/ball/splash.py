"""Sand-mediated momentum transfer for splash shots (issue #8613).

In a splash shot, the club enters the sand behind the ball and never makes
direct contact. Momentum is transferred through the displaced sand grains
that impact the ball.

Physics model:
1. Club displaces a volume of sand as it passes
2. Displaced sand has velocity proportional to club velocity
3. Sand grains collide with the ball, transferring momentum
4. Energy is dissipated through sand-grain friction and inelastic collisions

Key parameters from research-digest-addendum.md:
- Sand bulk density: 1550 kg/m^3 (USGA spec)
- Friction angle: ~33 deg (borrowed from similar sand)
- Ejecta velocity coefficient: ~0.3-0.5 of club velocity
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from src.shared.python.contracts import require

from .lie import BallLie, BallProperties

__all__ = [
    "ContactType",
    "SplashTransferResult",
    "compute_ball_launch_from_splash",
    "compute_sand_ejecta_velocity",
    "compute_splash_impulse",
]

# Physics constants (borrowed from research-digest-addendum.md)
_SAND_BULK_DENSITY_KG_M3: float = 1550.0  # USGA bunker sand
_EJECTA_VELOCITY_COEFF: float = 0.6  # fraction of club velocity to ejecta
_MOMENTUM_TRANSFER_EFFICIENCY: float = 0.5  # partially inelastic sand-ball collisions
_SAND_BALL_FRICTION: float = 0.5  # friction for spin generation


class ContactType(enum.Enum):
    """Type of club-ball-sand interaction."""

    SPLASH = "splash"  # Club never touches ball
    THIN = "thin"  # Club strikes ball directly (blade/thin shot)
    MIXED = "mixed"  # Both sand and direct contact


@dataclass(frozen=True, slots=True)
class SplashTransferResult:
    """Result of splash impulse calculation.

    Attributes:
        impulse_x_ns: Impulse in x direction [N.s].
        impulse_y_ns: Impulse in y direction [N.s].
        impulse_z_ns: Impulse in z direction [N.s].
        impulse_magnitude_ns: Total impulse magnitude [N.s].
        angular_impulse_x_ns: Angular impulse about x axis [N.m.s].
        angular_impulse_y_ns: Angular impulse about y axis [N.m.s].
        angular_impulse_z_ns: Angular impulse about z axis [N.m.s].
        sand_mass_kg: Effective sand mass involved [kg].
        contact_duration_s: Estimated contact duration [s].
    """

    impulse_x_ns: float
    impulse_y_ns: float
    impulse_z_ns: float
    impulse_magnitude_ns: float
    angular_impulse_x_ns: float
    angular_impulse_y_ns: float
    angular_impulse_z_ns: float
    sand_mass_kg: float
    contact_duration_s: float


@dataclass(frozen=True, slots=True)
class BallLaunchResult:
    """Ball launch conditions from splash.

    Attributes:
        ball_speed_m_s: Ball launch speed [m/s].
        launch_angle_rad: Launch angle from horizontal [rad].
        azimuth_rad: Azimuth angle [rad], 0 = forward.
        spin_rate_rpm: Spin rate [RPM].
        spin_axis: Spin axis unit vector [3].
        ball_velocity: Ball velocity vector [m/s] (3).
        ball_angular_velocity: Ball angular velocity [rad/s] (3).
        contact_type: Type of contact.
        energy_transfer_fraction: Fraction of club KE transferred to ball.
    """

    ball_speed_m_s: float
    launch_angle_rad: float
    azimuth_rad: float
    spin_rate_rpm: float
    spin_axis: tuple[float, float, float]
    ball_velocity: tuple[float, float, float]
    ball_angular_velocity: tuple[float, float, float]
    contact_type: ContactType
    energy_transfer_fraction: float


def compute_sand_ejecta_velocity(
    club_velocity_m_s: float,
    club_loft_rad: float,
) -> float:
    """Compute velocity of sand ejecta displaced by club.

    The ejecta velocity is a fraction of club velocity, modified by
    the club loft (higher loft directs more energy upward).

    Args:
        club_velocity_m_s: Club head speed [m/s].
        club_loft_rad: Club loft angle [rad].

    Returns:
        Sand ejecta velocity [m/s].
    """
    require(club_velocity_m_s >= 0, "club velocity must be non-negative")
    require(0 < club_loft_rad < math.pi / 2, "loft must be in (0, pi/2)")

    if club_velocity_m_s == 0:
        return 0.0

    # Ejecta velocity proportional to club velocity
    # Higher loft reduces horizontal momentum transfer
    loft_factor = 0.8 + 0.2 * (1 - math.sin(club_loft_rad))
    return _EJECTA_VELOCITY_COEFF * club_velocity_m_s * loft_factor


def compute_splash_impulse(
    lie: BallLie,
    ball: BallProperties,
    club_velocity_m_s: float,
    club_loft_rad: float,
    entry_depth_m: float,
    sole_width_m: float = 0.015,
    sole_length_m: float = 0.08,
) -> SplashTransferResult:
    """Compute impulse delivered to ball through sand.

    The model estimates the volume of sand displaced by the club,
    its velocity, and the momentum transfer to the ball.

    Args:
        lie: Ball lie specification.
        ball: Ball properties.
        club_velocity_m_s: Club head speed [m/s].
        club_loft_rad: Club loft angle [rad].
        entry_depth_m: Depth club enters sand [m].
        sole_width_m: Club sole width [m].
        sole_length_m: Club sole length [m].

    Returns:
        SplashTransferResult with impulse components.
    """
    require(club_velocity_m_s >= 0, "club velocity must be non-negative")
    require(entry_depth_m >= 0, "entry depth must be non-negative")

    # Estimate displaced sand volume
    # The club sweeps through sand, displacing a wedge-shaped volume
    # Volume = sole_length * entry_depth * sweep_length (approximately)
    # For a typical bunker shot: 8cm sole, 1.5cm depth, 12cm sweep = 144 cm^3
    sweep_length = 0.12  # ~12cm of sand contact (typical splash shot)
    displaced_volume = sole_length_m * entry_depth_m * sweep_length

    # Sand mass displaced
    sand_mass = _SAND_BULK_DENSITY_KG_M3 * displaced_volume

    # Ejecta velocity
    v_ejecta = compute_sand_ejecta_velocity(club_velocity_m_s, club_loft_rad)

    # Direction of ejecta: mostly upward and forward
    # Higher loft = more upward
    vertical_fraction = math.sin(club_loft_rad)
    forward_fraction = math.cos(club_loft_rad)

    # Impulse from sand hitting ball
    # Only a fraction of the sand actually hits the ball
    # This fraction increases with burial depth (more sand in path)
    hit_fraction = 0.3 + 0.7 * (lie.depth_m / ball.diameter_m)
    hit_fraction = min(1.0, max(0.1, hit_fraction))

    effective_sand_mass = sand_mass * hit_fraction
    sand_momentum = effective_sand_mass * v_ejecta

    # Transfer efficiency (inelastic collisions)
    transferred_impulse = sand_momentum * _MOMENTUM_TRANSFER_EFFICIENCY

    # Decompose into components
    impulse_z = transferred_impulse * vertical_fraction
    impulse_x = transferred_impulse * forward_fraction
    impulse_y = 0.0  # No lateral component for centered hit

    impulse_mag = math.sqrt(impulse_x**2 + impulse_y**2 + impulse_z**2)

    # Angular impulse (creates backspin)
    # Sand hitting below ball center creates backspin
    # The sand stream hits the lower hemisphere and transfers tangential momentum
    # Higher friction + larger contact area = more spin
    contact_offset = ball.radius_m * 0.7  # Sand hits below center
    # Total angular impulse is from both friction and direct tangential momentum
    angular_impulse_y = -contact_offset * transferred_impulse * _SAND_BALL_FRICTION

    # Contact duration estimate
    contact_duration = 0.005  # ~5ms for sand interaction

    return SplashTransferResult(
        impulse_x_ns=impulse_x,
        impulse_y_ns=impulse_y,
        impulse_z_ns=impulse_z,
        impulse_magnitude_ns=impulse_mag,
        angular_impulse_x_ns=0.0,
        angular_impulse_y_ns=angular_impulse_y,
        angular_impulse_z_ns=0.0,
        sand_mass_kg=effective_sand_mass,
        contact_duration_s=contact_duration,
    )


def compute_ball_launch_from_splash(
    lie: BallLie,
    ball: BallProperties,
    club_velocity_m_s: float,
    club_loft_rad: float,
    entry_depth_m: float,
    sole_width_m: float = 0.015,
    sole_length_m: float = 0.08,
    club_mass_kg: float = 0.30,
) -> BallLaunchResult:
    """Compute ball launch conditions from splash shot.

    Converts splash impulse to ball velocity and spin.

    Args:
        lie: Ball lie specification.
        ball: Ball properties.
        club_velocity_m_s: Club head speed [m/s].
        club_loft_rad: Club loft angle [rad].
        entry_depth_m: Depth club enters sand [m].
        sole_width_m: Club sole width [m].
        sole_length_m: Club sole length [m].
        club_mass_kg: Club head mass [kg].

    Returns:
        BallLaunchResult with velocity, angle, and spin.
    """
    # Compute impulse
    splash = compute_splash_impulse(
        lie=lie,
        ball=ball,
        club_velocity_m_s=club_velocity_m_s,
        club_loft_rad=club_loft_rad,
        entry_depth_m=entry_depth_m,
        sole_width_m=sole_width_m,
        sole_length_m=sole_length_m,
    )

    # Ball velocity from impulse
    v_x = splash.impulse_x_ns / ball.mass_kg
    v_y = splash.impulse_y_ns / ball.mass_kg
    v_z = splash.impulse_z_ns / ball.mass_kg

    ball_speed = math.sqrt(v_x**2 + v_y**2 + v_z**2)

    # Launch angle
    if ball_speed > 0:
        horizontal_speed = math.sqrt(v_x**2 + v_y**2)
        if horizontal_speed > 1e-10:
            launch_angle = math.atan2(v_z, horizontal_speed)
        else:
            launch_angle = math.pi / 2
        azimuth = math.atan2(v_y, v_x) if horizontal_speed > 1e-10 else 0.0
    else:
        launch_angle = 0.0
        azimuth = 0.0

    # Angular velocity from angular impulse
    # I_ball * omega = angular_impulse
    omega_y = splash.angular_impulse_y_ns / ball.moi_kg_m2
    omega_x = splash.angular_impulse_x_ns / ball.moi_kg_m2
    omega_z = splash.angular_impulse_z_ns / ball.moi_kg_m2

    spin_rate_rad_s = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    spin_rate_rpm = spin_rate_rad_s * 60.0 / (2.0 * math.pi)

    # Spin axis
    if spin_rate_rad_s > 1e-10:
        spin_axis = (
            omega_x / spin_rate_rad_s,
            omega_y / spin_rate_rad_s,
            omega_z / spin_rate_rad_s,
        )
    else:
        spin_axis = (0.0, -1.0, 0.0)  # Default to pure backspin

    # Energy transfer fraction
    club_ke = 0.5 * club_mass_kg * club_velocity_m_s**2
    ball_ke = 0.5 * ball.mass_kg * ball_speed**2
    ball_rot_ke = 0.5 * ball.moi_kg_m2 * spin_rate_rad_s**2
    total_ball_energy = ball_ke + ball_rot_ke

    energy_fraction = total_ball_energy / club_ke if club_ke > 0 else 0.0

    return BallLaunchResult(
        ball_speed_m_s=ball_speed,
        launch_angle_rad=launch_angle,
        azimuth_rad=azimuth,
        spin_rate_rpm=spin_rate_rpm,
        spin_axis=spin_axis,
        ball_velocity=(v_x, v_y, v_z),
        ball_angular_velocity=(omega_x, omega_y, omega_z),
        contact_type=ContactType.SPLASH,
        energy_transfer_fraction=energy_fraction,
    )
