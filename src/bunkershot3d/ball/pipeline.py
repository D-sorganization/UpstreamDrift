"""Pipeline integration for bunkershot3d ball model (issue #8613).

Provides the handoff interface between bunkershot3d and the existing
SwingBallFlightPipeline infrastructure. Converts bunker shot results
to PostImpactState for flight simulation.

Usage:
    state = BunkerShotState(...)
    result = compute_bunker_launch(state)
    post = to_post_impact_state(result, state)
    # Then use SwingBallFlightPipeline with post_state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.physics.impact_model import PostImpactState

from .lie import BallLie, BallProperties
from .splash import BallLaunchResult, compute_ball_launch_from_splash

__all__ = [
    "BunkerShotState",
    "compute_bunker_launch",
    "to_post_impact_state",
]


@dataclass(slots=True)
class BunkerShotState:
    """Complete specification of a bunker shot.

    Attributes:
        club_velocity_m_s: Club head speed at impact [m/s].
        club_loft_deg: Club loft angle [degrees].
        ball_lie: Ball position and burial in sand.
        entry_depth_m: How deep club enters sand [m].
        sole_width_m: Club sole width [m].
        sole_length_m: Club sole length [m].
        club_mass_kg: Club head mass [kg].
        ball: Ball properties.
    """

    club_velocity_m_s: float
    club_loft_deg: float
    ball_lie: BallLie
    entry_depth_m: float = 0.015  # 15mm typical
    sole_width_m: float = 0.015  # 15mm
    sole_length_m: float = 0.08  # 80mm
    club_mass_kg: float = 0.30  # 300g wedge head
    ball: BallProperties = field(default_factory=BallProperties)

    def __post_init__(self) -> None:
        require(self.club_velocity_m_s >= 0, "club velocity must be non-negative")
        require(0 < self.club_loft_deg < 90, "loft must be in (0, 90) degrees")
        require(self.entry_depth_m >= 0, "entry depth must be non-negative")
        require(self.sole_width_m > 0, "sole width must be positive")
        require(self.sole_length_m > 0, "sole length must be positive")
        require(self.club_mass_kg > 0, "club mass must be positive")


def compute_bunker_launch(state: BunkerShotState) -> BallLaunchResult:
    """Compute ball launch conditions from bunker shot.

    This is the main entry point for bunker shot physics. It determines
    whether to use splash transfer (typical) or direct contact (thin shot).

    Args:
        state: Complete bunker shot specification.

    Returns:
        BallLaunchResult with velocity, spin, and energy accounting.
    """
    club_loft_rad = math.radians(state.club_loft_deg)

    # Always use splash transfer (thin/blade direct contact is out of scope for #8613)
    return compute_ball_launch_from_splash(
        lie=state.ball_lie,
        ball=state.ball,
        club_velocity_m_s=state.club_velocity_m_s,
        club_loft_rad=club_loft_rad,
        entry_depth_m=state.entry_depth_m,
        sole_width_m=state.sole_width_m,
        sole_length_m=state.sole_length_m,
        club_mass_kg=state.club_mass_kg,
    )


def to_post_impact_state(
    result: BallLaunchResult,
    state: BunkerShotState,
) -> PostImpactState:
    """Convert BallLaunchResult to PostImpactState for pipeline handoff.

    The PostImpactState is the interface expected by SwingBallFlightPipeline
    and other flight simulation components.

    Args:
        result: Ball launch result from bunker shot.
        state: Original bunker shot state.

    Returns:
        PostImpactState ready for flight simulation.
    """
    # Ball velocity as numpy array
    ball_velocity = np.array(result.ball_velocity, dtype=float)

    # Ball angular velocity as numpy array
    ball_angular_velocity = np.array(result.ball_angular_velocity, dtype=float)

    # Clubhead velocity after passing through sand
    # The club slows down due to sand resistance
    # Estimate: club loses energy proportional to what went to ball + sand
    club_loft_rad = math.radians(state.club_loft_deg)
    club_ke_in = 0.5 * state.club_mass_kg * state.club_velocity_m_s**2

    # Energy dissipated = input - ball KE (most goes to sand)
    ball_ke = 0.5 * state.ball.mass_kg * result.ball_speed_m_s**2
    ball_rot_ke = (
        0.5 * state.ball.moi_kg_m2 * np.linalg.norm(ball_angular_velocity) ** 2
    )

    # Club retains some energy after passing through
    # In a splash shot, club loses significant energy to sand
    # Club keeps maybe 30-40% of its KE
    club_retention_factor = 0.35
    club_ke_out = club_ke_in * club_retention_factor
    club_speed_out = math.sqrt(2 * club_ke_out / state.club_mass_kg)

    # Club direction is forward and slightly down after divot
    club_velocity = np.array(
        [
            club_speed_out * math.cos(club_loft_rad * 0.3),  # Forward
            0.0,  # No lateral
            -club_speed_out * math.sin(club_loft_rad * 0.3),  # Down through sand
        ],
        dtype=float,
    )

    # Club angular velocity (face rotation after impact)
    clubhead_angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)

    # Energy transferred to ball
    energy_transfer = ball_ke + ball_rot_ke

    # Contact duration estimate (sand interaction is longer than ball-club)
    contact_duration = 0.005  # 5ms

    # Impact location (center for splash shot)
    impact_location = np.array([0.0, 0.0], dtype=float)

    return PostImpactState(
        ball_velocity=ball_velocity,
        ball_angular_velocity=ball_angular_velocity,
        clubhead_velocity=club_velocity,
        clubhead_angular_velocity=clubhead_angular_velocity,
        contact_duration=contact_duration,
        energy_transfer=float(energy_transfer),
        impact_location=impact_location,
    )
