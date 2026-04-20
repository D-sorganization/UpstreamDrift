import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
)

from .types import ImpactParameters, PostImpactState, PreImpactState


@precondition(  # fmt: skip
    lambda impact_offset,
    clubhead_velocity,
    clubface_normal,
    gear_factor=0.5,
    h_scale=100.0,
    v_scale=50.0: (0 <= gear_factor <= 1),
    "Gear effect factor must be between 0 and 1",
)
def compute_gear_effect_spin(
    impact_offset: np.ndarray,
    clubhead_velocity: np.ndarray,
    clubface_normal: np.ndarray,
    gear_factor: float = 0.5,
    h_scale: float = 100.0,
    v_scale: float = 50.0,
) -> np.ndarray:
    """Compute spin from gear effect for off-center impact.

    Gear effect occurs when the ball contacts the clubface
    away from the center of gravity, causing the clubhead
    to rotate and impart spin to the ball.

    Args:
        impact_offset: Offset from clubface center [m] (2,) [horizontal, vertical]
        clubhead_velocity: Clubhead velocity at impact [m/s] (3,)
        clubface_normal: Clubface normal vector [unitless] (3,)
        gear_factor: Gear effect amplification (0-1)
        h_scale: Scaling factor for horizontal offset
        v_scale: Scaling factor for vertical offset

    Returns:
        Additional spin from gear effect [rad/s] (3,)
    """
    # Horizontal offset creates hook/slice spin (vertical axis)
    # Vertical offset creates topspin/backspin
    if impact_offset is None:
        raise ValueError("impact_offset must be provided")
    h_offset = impact_offset[0]  # + = toe side
    v_offset = impact_offset[1]  # + = high on face

    # Speed affects spin magnitude
    speed = np.linalg.norm(clubhead_velocity)

    # Gear effect spin rate (empirical relationship)
    # Higher offset = more spin, proportional to speed
    horizontal_spin = -gear_factor * h_offset * speed * h_scale  # [rad/s]
    vertical_spin = gear_factor * v_offset * speed * v_scale  # [rad/s]

    # Convert to 3D spin vector
    # Assuming clubface normal is approximately in X direction
    # Vertical axis is Z, horizontal axis perpendicular to both
    up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(clubface_normal, up)
    if np.linalg.norm(horizontal_axis) > 1e-6:
        horizontal_axis /= np.linalg.norm(horizontal_axis)
    else:
        horizontal_axis = np.array([0.0, 1.0, 0.0])

    spin = horizontal_spin * up + vertical_spin * horizontal_axis

    return np.asarray(spin)


def validate_energy_balance(
    pre_state: PreImpactState,
    post_state: PostImpactState,
    params: ImpactParameters,
) -> dict[str, float]:
    """Validate energy balance before and after impact.

    Total mechanical energy should be conserved up to COR losses.

    Args:
        pre_state: Pre-impact state
        post_state: Post-impact state
        params: Impact parameters

    Returns:
        Dictionary with energy analysis results
    """
    if pre_state is None:
        raise ValueError("pre_state must be provided")
    m_ball = GOLF_BALL_MASS_KG
    m_club = pre_state.clubhead_mass
    I_ball = GOLF_BALL_MOMENT_OF_INERTIA_KG_M2

    # Pre-impact kinetic energy
    ke_ball_pre = (
        0.5 * m_ball * np.dot(pre_state.ball_velocity, pre_state.ball_velocity)
    )
    ke_ball_rot_pre = (
        0.5
        * I_ball
        * np.dot(pre_state.ball_angular_velocity, pre_state.ball_angular_velocity)
    )
    ke_club_pre = (
        0.5 * m_club * np.dot(pre_state.clubhead_velocity, pre_state.clubhead_velocity)
    )
    total_ke_pre = ke_ball_pre + ke_ball_rot_pre + ke_club_pre

    # Post-impact kinetic energy
    ke_ball_post = (
        0.5 * m_ball * np.dot(post_state.ball_velocity, post_state.ball_velocity)
    )
    ke_ball_rot_post = (
        0.5
        * I_ball
        * np.dot(post_state.ball_angular_velocity, post_state.ball_angular_velocity)
    )
    ke_club_post = (
        0.5
        * m_club
        * np.dot(post_state.clubhead_velocity, post_state.clubhead_velocity)
    )
    total_ke_post = ke_ball_post + ke_ball_rot_post + ke_club_post

    # Energy loss
    energy_lost = total_ke_pre - total_ke_post
    expected_loss_factor = 1 - params.cor**2  # COR relates velocities, not energy

    return {
        "total_ke_pre": float(total_ke_pre),
        "total_ke_post": float(total_ke_post),
        "energy_lost": float(energy_lost),
        "energy_loss_ratio": (
            float(energy_lost / total_ke_pre) if total_ke_pre > 0 else 0
        ),
        "expected_loss_factor": expected_loss_factor,
        "ball_ke_post": float(ke_ball_post),
        "ball_launch_speed": float(np.linalg.norm(post_state.ball_velocity)),
    }
