import math
import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
)

from .contact_kinematics import (
    compute_effective_club_mass,
    pre_impact_contact_relative_velocity,
    reduced_normal_effective_mass,
    unit_face_normal,
)
from .types import ImpactParameters, PostImpactState, PreImpactState

_TANGENTIAL_FRICTION_EXCLUSION_REASON = (
    "expected_loss_j is the central normal COR dissipation "
    "0.5*mu_eff*v_rel,n^2*(1-e^2) only; tangential friction impulse work is "
    "included in energy_lost but not modeled in the COR budget."
)

__all__ = [
    "compute_effective_club_mass",
    "compute_gear_effect_spin",
    "pre_impact_contact_relative_velocity",
    "validate_energy_balance",
]


# fmt: off
@precondition(  # fmt: skip
    lambda impact_offset, clubhead_velocity, clubface_normal, gear_factor=0.5, h_scale=350.0, v_scale=150.0: (
        0 <= gear_factor <= 1
    ),
    "Gear effect factor must be between 0 and 1",
)
def compute_gear_effect_spin(
    impact_offset: np.ndarray,
    clubhead_velocity: np.ndarray,
    clubface_normal: np.ndarray,
    gear_factor: float = 0.5,
    h_scale: float = 350.0,
    v_scale: float = 150.0,
) -> np.ndarray:
# fmt: on
    """Estimate an additive gear-spin contribution for right-handed delivery.

    Gear effect occurs when the ball contacts the clubface
    away from the center of gravity, causing the clubhead
    to rotate and impart spin to the ball.

    This legacy heuristic does not solve coupled club/ball angular momentum
    or represent measured equipment calibration. For forward +X and up +Z,
    +Y is left. Positive toe offset adds +Z spin, whose Magnus direction
    (spin cross velocity) is left; heel offset reverses it. Positive high-face
    offset adds topspin about +Y, reducing a negative-Y backspin component.

    Args:
        impact_offset: Offset from clubface center [m] (2,), toe-positive and
            high-positive for the right-handed convention.
        clubhead_velocity: Clubhead velocity at impact [m/s] (3,)
        clubface_normal: Clubface normal vector [unitless] (3,)
        gear_factor: Gear effect amplification (0-1)
        h_scale: Heuristic horizontal scale [rad/m²]; not equipment calibration.
        v_scale: Heuristic vertical scale [rad/m²]; not equipment calibration.

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
    speed = math.sqrt(np.dot(clubhead_velocity, clubhead_velocity))  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm

    # Gear effect spin rate (uncalibrated heuristic relationship)
    # Higher offset = more spin, proportional to speed
    horizontal_spin = gear_factor * h_offset * speed * h_scale  # [rad/s]
    vertical_spin = gear_factor * v_offset * speed * v_scale  # [rad/s]

    # Convert to 3D spin vector
    # Assuming clubface normal is approximately in X direction
    # Vertical axis is Z, horizontal axis perpendicular to both
    up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(up, clubface_normal)
    if math.sqrt(np.dot(horizontal_axis, horizontal_axis)) > 1e-6:  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
        horizontal_axis /= math.sqrt(np.dot(horizontal_axis, horizontal_axis))  # ⚡ Bolt: math.sqrt(np.dot) is ~3x faster than np.linalg.norm
    else:
        horizontal_axis = np.array([0.0, 1.0, 0.0])

    spin = horizontal_spin * up + vertical_spin * horizontal_axis

    return np.asarray(spin)


def validate_energy_balance(
    pre_state: PreImpactState,
    post_state: PostImpactState,
    params: ImpactParameters,
) -> dict[str, float | bool | str]:
    """Summarize translational and rotational kinetic energy around impact.

    Compares measured kinetic-energy loss against the normal COR dissipation
    budget. Both total KE (translational plus rotational) and the expected loss
    use contact-point relative velocity v_rel,n and reduced normal effective mass
    1/(1/m + n·((I^{-1}(r×n))×r)) for club and ball. Tangential friction work appears in
    ``energy_lost`` but is excluded from ``expected_loss_j``; use
    ``normal_energy_budget_met`` rather than equating the two when friction is
    nonzero.
    """
    if pre_state is None:
        raise ValueError("pre_state must be provided")
    m_ball = float(GOLF_BALL_MASS_KG)
    m_club_effective = compute_effective_club_mass(pre_state)
    I_ball = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)
    I_club = float(pre_state.clubhead_moi)

    n_unit = unit_face_normal(pre_state.clubhead_orientation)
    v_rel = pre_impact_contact_relative_velocity(pre_state)
    v_rel_n = float(np.dot(v_rel, n_unit))
    mu_eff = reduced_normal_effective_mass(pre_state, n_unit)
    expected_loss_j = 0.5 * mu_eff * v_rel_n**2 * (1.0 - params.cor**2)

    ke_ball_pre = 0.5 * m_ball * float(
        np.dot(pre_state.ball_velocity, pre_state.ball_velocity)
    )
    ke_ball_rot_pre = 0.5 * I_ball * float(
        np.dot(pre_state.ball_angular_velocity, pre_state.ball_angular_velocity)
    )
    ke_club_pre = 0.5 * float(pre_state.clubhead_mass) * float(
        np.dot(pre_state.clubhead_velocity, pre_state.clubhead_velocity)
    )
    omega_club_pre = np.asarray(pre_state.clubhead_angular_velocity, dtype=float)
    club_rot_ke_pre = 0.5 * I_club * float(np.dot(omega_club_pre, omega_club_pre))

    ke_ball_post = 0.5 * m_ball * float(
        np.dot(post_state.ball_velocity, post_state.ball_velocity)
    )
    ke_ball_rot_post = 0.5 * I_ball * float(
        np.dot(post_state.ball_angular_velocity, post_state.ball_angular_velocity)
    )
    ke_club_post = 0.5 * float(pre_state.clubhead_mass) * float(
        np.dot(post_state.clubhead_velocity, post_state.clubhead_velocity)
    )
    omega_club_post = np.asarray(post_state.clubhead_angular_velocity, dtype=float)
    club_rot_ke_post = 0.5 * I_club * float(np.dot(omega_club_post, omega_club_post))

    total_ke_pre = ke_ball_pre + ke_ball_rot_pre + ke_club_pre + club_rot_ke_pre
    total_ke_post = ke_ball_post + ke_ball_rot_post + ke_club_post + club_rot_ke_post
    energy_lost = total_ke_pre - total_ke_post
    expected_loss_factor = 1.0 - params.cor**2

    friction = float(params.friction_coefficient)
    normal_budget_met = energy_lost + 1e-9 >= expected_loss_j

    return {
        "total_ke_pre": float(total_ke_pre),
        "total_ke_post": float(total_ke_post),
        "energy_lost": float(energy_lost),
        "energy_loss_ratio": (
            float(energy_lost / total_ke_pre) if total_ke_pre > 0 else 0.0
        ),
        "expected_loss_factor": expected_loss_factor,
        "expected_loss_j": float(expected_loss_j),
        "effective_club_mass_kg": float(m_club_effective),
        "club_rot_ke_pre": float(club_rot_ke_pre),
        "club_rot_ke_post": float(club_rot_ke_post),
        "tangential_friction_in_expected_loss": False,
        "tangential_friction_exclusion_reason": _TANGENTIAL_FRICTION_EXCLUSION_REASON,
        "friction_coefficient": friction,
        "normal_energy_budget_met": normal_budget_met,
        "ball_ke_post": float(ke_ball_post),
        "ball_launch_speed": float(
            math.sqrt(np.dot(post_state.ball_velocity, post_state.ball_velocity))
        ),
    }
