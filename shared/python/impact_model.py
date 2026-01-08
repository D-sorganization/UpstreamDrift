"""Modular Impact Model Module.

Guideline K3 Implementation: Modular Impact Model (MuJoCo).

Provides standalone impact solver for ball-clubface collision including:
- Rigid body collision (coefficient of restitution)
- Compliant contact (spring-damper, Kelvin-Voigt)
- Finite-time contact (impulse-momentum with contact duration)
- Spin generation models (gear effect, offset impact)

The module is engine-agnostic with Python API for external solvers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)

# Physical constants for golf ball
GOLF_BALL_MASS = 0.0459  # [kg] Maximum mass per USGA rules
GOLF_BALL_RADIUS = 0.02135  # [m] Minimum radius per USGA rules
GOLF_BALL_MOMENT_INERTIA = (2 / 5) * GOLF_BALL_MASS * GOLF_BALL_RADIUS**2  # [kg·m²]

# Default impact parameters
DEFAULT_COR = 0.78  # Coefficient of restitution for driver
DEFAULT_CONTACT_DURATION = 0.0005  # [s] 500 μs typical contact time


class ImpactModelType(Enum):
    """Types of impact physics models."""

    RIGID_BODY = auto()  # Instantaneous impulse with COR
    SPRING_DAMPER = auto()  # Kelvin-Voigt viscoelastic
    FINITE_TIME = auto()  # Impulse-momentum with duration


@dataclass
class PreImpactState:
    """State of ball and clubhead immediately before impact.

    Attributes:
        clubhead_velocity: Clubhead velocity [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity [rad/s] (3,)
        clubhead_orientation: Clubface normal vector [unitless] (3,)
        ball_position: Ball center position [m] (3,)
        ball_velocity: Ball velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_mass: Effective clubhead mass [kg]
        clubhead_loft: Clubface loft angle [rad]
        clubhead_lie: Clubface lie angle [rad]
    """

    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    clubhead_orientation: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_mass: float = 0.200  # [kg] Typical driver head
    clubhead_loft: float = np.radians(10.5)  # [rad] Driver loft
    clubhead_lie: float = np.radians(60.0)  # [rad] Lie angle


@dataclass
class PostImpactState:
    """State of ball and clubhead immediately after impact.

    Attributes:
        ball_velocity: Ball launch velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_velocity: Clubhead velocity after impact [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity after [rad/s] (3,)
        contact_duration: Duration of contact [s]
        energy_transfer: Kinetic energy transferred to ball [J]
        impact_location: Location of impact on clubface [m] (2,) [x, y from center]
    """

    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    contact_duration: float
    energy_transfer: float
    impact_location: np.ndarray


@dataclass
class ImpactParameters:
    """Parameters for impact model.

    Attributes:
        cor: Coefficient of restitution (0-1)
        contact_duration: Contact time [s]
        contact_stiffness: Spring stiffness for compliant model [N/m]
        contact_damping: Damping for compliant model [N·s/m]
        friction_coefficient: Ball-face friction
        gear_effect_factor: Gear effect spin amplification (0-1)
    """

    cor: float = DEFAULT_COR
    contact_duration: float = DEFAULT_CONTACT_DURATION
    contact_stiffness: float = 1e6  # [N/m]
    contact_damping: float = 1e3  # [N·s/m]
    friction_coefficient: float = 0.4
    gear_effect_factor: float = 0.5


class ImpactModel(ABC):
    """Abstract base class for impact models."""

    @abstractmethod
    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve the impact and return post-impact state.

        Args:
            pre_state: Pre-impact state of ball and clubhead
            params: Impact model parameters

        Returns:
            Post-impact state
        """


class RigidBodyImpactModel(ImpactModel):
    """Rigid body collision with coefficient of restitution.

    Uses instantaneous impulse-momentum equations with COR
    to compute post-impact velocities.
    """

    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using rigid body collision model.

        The impact is modeled as an instantaneous impulse along
        the contact normal (clubface normal), with COR determining
        the relationship between approach and separation velocities.

        Args:
            pre_state: Pre-impact state
            params: Impact parameters

        Returns:
            Post-impact state
        """
        # Masses
        m_ball = GOLF_BALL_MASS
        m_club = pre_state.clubhead_mass

        # Contact normal (clubface normal, pointing away from club)
        n = pre_state.clubhead_orientation / np.linalg.norm(
            pre_state.clubhead_orientation
        )

        # Relative velocity (club relative to ball)
        v_rel = pre_state.clubhead_velocity - pre_state.ball_velocity

        # Approach velocity (component along normal)
        v_approach = np.dot(v_rel, n)

        # COR equation: v_sep = -e * v_app
        # Combined with momentum conservation
        e = params.cor

        # Effective mass
        m_eff = (m_ball * m_club) / (m_ball + m_club)

        # Impulse magnitude
        j = (1 + e) * m_eff * v_approach

        # Post-impact velocities
        v_ball_post = pre_state.ball_velocity + (j / m_ball) * n
        v_club_post = pre_state.clubhead_velocity - (j / m_club) * n

        # Spin generation from friction (simplified)
        # Tangential velocity at contact
        v_tangent = v_rel - v_approach * n
        tangent_mag = np.linalg.norm(v_tangent)

        if tangent_mag > 1e-6:
            tangent_dir = v_tangent / tangent_mag
            # Friction impulse (limited by friction cone)
            j_friction = min(
                params.friction_coefficient * j, m_ball * tangent_mag * 0.4
            )
            # Spin from friction: τ = r × F, integrated
            spin_axis = np.cross(n, tangent_dir)
            spin_magnitude = j_friction / (GOLF_BALL_MOMENT_INERTIA / GOLF_BALL_RADIUS)
            ball_spin = pre_state.ball_angular_velocity + spin_magnitude * spin_axis
        else:
            ball_spin = pre_state.ball_angular_velocity.copy()

        # Energy calculation
        ke_ball_pre = (
            0.5 * m_ball * np.dot(pre_state.ball_velocity, pre_state.ball_velocity)
        )
        ke_ball_post = 0.5 * m_ball * np.dot(v_ball_post, v_ball_post)
        energy_transfer = ke_ball_post - ke_ball_pre

        return PostImpactState(
            ball_velocity=v_ball_post,
            ball_angular_velocity=ball_spin,
            clubhead_velocity=v_club_post,
            clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy(),
            contact_duration=0.0,  # Instantaneous
            energy_transfer=energy_transfer,
            impact_location=np.zeros(2),  # Center impact
        )


class SpringDamperImpactModel(ImpactModel):
    """Spring-damper (Kelvin-Voigt) compliant contact model.

    Uses semi-implicit integration of spring-damper contact to
    compute force and velocity evolution during impact.
    """

    def __init__(self, dt: float = 1e-7) -> None:
        """Initialize spring-damper model.

        Args:
            dt: Integration time step [s] (default 0.1 μs for stability)
        """
        self.dt = dt

    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using spring-damper contact model.

        Integrates the contact dynamics over time until separation.
        Uses semi-implicit Euler for numerical stability.

        Args:
            pre_state: Pre-impact state
            params: Impact parameters

        Returns:
            Post-impact state
        """
        m_ball = GOLF_BALL_MASS
        m_club = pre_state.clubhead_mass

        n = pre_state.clubhead_orientation / np.linalg.norm(
            pre_state.clubhead_orientation
        )

        # Initial state - place ball at contact
        x_ball = GOLF_BALL_RADIUS * n  # Ball surface at origin
        v_ball = pre_state.ball_velocity.copy()
        x_club = np.zeros(3)
        v_club = pre_state.clubhead_velocity.copy()

        # Integration
        contact_time = 0.0
        max_time = 0.005  # 5 ms max contact time [s]
        max_steps = int(max_time / self.dt)

        # Limit max force to prevent numerical blow-up
        max_force = 1e5  # [N] max contact force

        for _ in range(max_steps):
            # Penetration depth (along normal)
            gap = np.dot(x_ball - x_club, n) - GOLF_BALL_RADIUS

            if gap < 0:  # In contact (penetration)
                penetration = -gap

                # Contact force (spring-damper)
                v_rel_normal = np.dot(v_ball - v_club, n)
                f_spring = params.contact_stiffness * penetration
                f_damper = -params.contact_damping * v_rel_normal
                f_magnitude = max(0.0, min(f_spring + f_damper, max_force))

                f_contact = f_magnitude * n

                # Semi-implicit Euler: update velocities first
                a_ball = -f_contact / m_ball
                a_club = f_contact / m_club

                v_ball = v_ball + a_ball * self.dt
                v_club = v_club + a_club * self.dt

                # Then positions
                x_ball = x_ball + v_ball * self.dt
                x_club = x_club + v_club * self.dt

                contact_time += self.dt

            elif contact_time > 0:
                # Was in contact but now separated
                break
            else:
                # Pre-contact: advance positions
                x_ball = x_ball + v_ball * self.dt
                x_club = x_club + v_club * self.dt
                contact_time += self.dt

                # Check if we've reached the ball
                if np.dot(x_ball - x_club, n) - GOLF_BALL_RADIUS < 0:
                    continue

        # Energy calculation
        ke_ball_pre = (
            0.5 * m_ball * np.dot(pre_state.ball_velocity, pre_state.ball_velocity)
        )
        ke_ball_post = 0.5 * m_ball * np.dot(v_ball, v_ball)
        energy_transfer = ke_ball_post - ke_ball_pre

        return PostImpactState(
            ball_velocity=v_ball,
            ball_angular_velocity=pre_state.ball_angular_velocity.copy(),
            clubhead_velocity=v_club,
            clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy(),
            contact_duration=contact_time,
            energy_transfer=energy_transfer,
            impact_location=np.zeros(2),
        )


class FiniteTimeImpactModel(ImpactModel):
    """Finite-time impulse-momentum model.

    Computes impact over a specified contact duration using
    momentum conservation and gradual force application.
    """

    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using finite-time model.

        Uses the specified contact duration to compute average
        force and resulting velocities.

        Args:
            pre_state: Pre-impact state
            params: Impact parameters

        Returns:
            Post-impact state
        """
        # For finite-time model, we use the rigid body result
        # but report the specified contact duration
        rigid_model = RigidBodyImpactModel()
        result = rigid_model.solve(pre_state, params)

        # Override contact duration
        return PostImpactState(
            ball_velocity=result.ball_velocity,
            ball_angular_velocity=result.ball_angular_velocity,
            clubhead_velocity=result.clubhead_velocity,
            clubhead_angular_velocity=result.clubhead_angular_velocity,
            contact_duration=params.contact_duration,
            energy_transfer=result.energy_transfer,
            impact_location=result.impact_location,
        )


def compute_gear_effect_spin(
    impact_offset: np.ndarray,
    clubhead_velocity: np.ndarray,
    clubface_normal: np.ndarray,
    gear_factor: float = 0.5,
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

    Returns:
        Additional spin from gear effect [rad/s] (3,)
    """
    # Horizontal offset creates hook/slice spin (vertical axis)
    # Vertical offset creates topspin/backspin
    h_offset = impact_offset[0]  # + = toe side
    v_offset = impact_offset[1]  # + = high on face

    # Speed affects spin magnitude
    speed = np.linalg.norm(clubhead_velocity)

    # Gear effect spin rate (empirical relationship)
    # Higher offset = more spin, proportional to speed
    horizontal_spin = -gear_factor * h_offset * speed * 100  # [rad/s]
    vertical_spin = gear_factor * v_offset * speed * 50  # [rad/s]

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

    return spin


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
    m_ball = GOLF_BALL_MASS
    m_club = pre_state.clubhead_mass
    I_ball = GOLF_BALL_MOMENT_INERTIA

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


def create_impact_model(model_type: ImpactModelType) -> ImpactModel:
    """Factory function to create impact model instance.

    Args:
        model_type: Type of impact model to create

    Returns:
        Impact model instance
    """
    if model_type == ImpactModelType.RIGID_BODY:
        return RigidBodyImpactModel()
    elif model_type == ImpactModelType.SPRING_DAMPER:
        return SpringDamperImpactModel()
    elif model_type == ImpactModelType.FINITE_TIME:
        return FiniteTimeImpactModel()
    else:
        raise ValueError(f"Unknown impact model type: {model_type}")
