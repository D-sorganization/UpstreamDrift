"""Impact physics: data models, collision solvers, and helper functions.

Extracted from impact_model.py. Import via impact_model module.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from src.shared.python.core.contracts import precondition

from ..core.physics_constants import (
    DRIVER_COR,
    DRIVER_MOI_KG_M2,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    TYPICAL_CONTACT_DURATION_S,
)


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
        clubhead_moi: Clubhead moment of inertia about CG [kg·m²]
        impact_offset: Impact location offset from CG on clubface [m] (2,) [horizontal, vertical]
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
    clubhead_moi: float = DRIVER_MOI_KG_M2
    impact_offset: np.ndarray | None = None


@dataclass
class PostImpactState:
    """State of ball and clubhead immediately after impact.

    Attributes:
        ball_velocity: Post-impact ball velocity [m/s] (3,)
        ball_angular_velocity: Post-impact ball spin [rad/s] (3,)
        clubhead_velocity: Post-impact clubhead velocity [m/s] (3,)
        clubhead_angular_velocity: Post-impact clubhead spin [rad/s] (3,)
        contact_duration: Duration of contact [s]
        energy_transfer: Energy transferred to ball [J]
        impact_location: Location on clubface where contact occurred [m] (2,)
    """

    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    contact_duration: float = 0.0
    energy_transfer: float = 0.0
    impact_location: np.ndarray | None = None


@dataclass
class ImpactParameters:
    """Parameters for impact physics models.

    Attributes:
        cor: Coefficient of restitution (0=perfectly inelastic, 1=perfectly elastic)
        friction_coefficient: Tangential friction during contact
        contact_stiffness: Spring stiffness for compliant contact [N/m]
        contact_damping: Damping coefficient for compliant contact [N·s/m]
        contact_duration: Contact duration for finite-time model [s]
        gear_effect_factor: Gear effect spin amplification (0-1)
        gear_effect_h_scale: Horizontal gear effect scaling [1/m]
        gear_effect_v_scale: Vertical gear effect scaling [1/m]
    """

    cor: float = DRIVER_COR
    friction_coefficient: float = 0.15
    contact_stiffness: float = 1e7  # [N/m]
    contact_damping: float = 1e3  # [N·s/m]
    contact_duration: float = TYPICAL_CONTACT_DURATION_S
    gear_effect_factor: float = 0.5
    gear_effect_h_scale: float = 100.0
    gear_effect_v_scale: float = 50.0


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
        ...


class RigidBodyImpactModel(ImpactModel):
    """Rigid body collision with coefficient of restitution.

    Uses instantaneous impulse-momentum equations with COR
    to compute post-impact velocities.
    """

    def _compute_effective_club_mass(self, pre_state: PreImpactState) -> float:
        # Known limitation: this uses a simplified scalar effective mass model.
        # It ignores the full 3D inertia tensor and the direction of the impact force.
        # Should be replaced with J = (1/m + r x (I^-1 * (r x n)))^-1 * (1 + e) * v_rel
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        m_club = pre_state.clubhead_mass
        club_moi = pre_state.clubhead_moi

        if pre_state.impact_offset is not None and club_moi > 0:
            r_offset = float(
                0.0
                if np.asarray(pre_state.impact_offset, dtype=float).reshape(-1).size
                == 0
                else math.hypot(
                    *np.asarray(pre_state.impact_offset, dtype=float).reshape(-1)
                )
            )
            if r_offset > 1e-6:
                return 1.0 / (1.0 / m_club + r_offset**2 / club_moi)
        return m_club

    def _compute_impulse(
        self,
        v_rel: np.ndarray,
        n: np.ndarray,
        m_club_effective: float,
        cor: float,
    ) -> tuple[float, float]:
        if v_rel is None:
            raise ValueError("v_rel must be provided")
        v_approach = np.dot(v_rel, n)
        m_eff = (GOLF_BALL_MASS_KG * m_club_effective) / (
            GOLF_BALL_MASS_KG + m_club_effective
        )
        j = (1 + cor) * m_eff * v_approach
        return j, v_approach

    def _compute_friction_spin(
        self,
        pre_state: PreImpactState,
        v_rel: np.ndarray,
        v_approach: float,
        n: np.ndarray,
        j: float,
        friction_coefficient: float,
    ) -> np.ndarray:
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        v_tangent = v_rel - v_approach * n
        tangent_mag = (
            0.0
            if np.asarray(v_tangent, dtype=float).reshape(-1).size == 0
            else math.hypot(*np.asarray(v_tangent, dtype=float).reshape(-1))
        )

        if tangent_mag <= 1e-6:
            return pre_state.ball_angular_velocity.copy()

        tangent_dir = v_tangent / tangent_mag
        j_friction = min(
            float(friction_coefficient * j),
            float(GOLF_BALL_MASS_KG * tangent_mag * 0.4),
        )
        spin_axis = np.cross(n, tangent_dir)
        spin_magnitude = j_friction / (
            GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 / GOLF_BALL_RADIUS_M
        )
        return pre_state.ball_angular_velocity + spin_magnitude * spin_axis

    def _compute_energy_transfer(
        self,
        pre_ball_velocity: np.ndarray,
        post_ball_velocity: np.ndarray,
    ) -> float:
        if pre_ball_velocity is None:
            raise ValueError("pre_ball_velocity must be provided")
        ke_pre = 0.5 * GOLF_BALL_MASS_KG * np.dot(pre_ball_velocity, pre_ball_velocity)
        ke_post = (
            0.5 * GOLF_BALL_MASS_KG * np.dot(post_ball_velocity, post_ball_velocity)
        )
        return ke_post - ke_pre

    @precondition(
        lambda self, pre_state, params: pre_state.clubhead_mass > 0,
        "Clubhead mass must be positive",
    )
    @precondition(
        lambda self, pre_state, params: 0 <= params.cor <= 1,
        "Coefficient of restitution must be between 0 and 1",
    )
    @precondition(
        lambda self, pre_state, params: params.friction_coefficient >= 0,
        "Friction coefficient must be non-negative",
    )
    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using rigid body collision model with MOI.

        The clubhead is modeled as a rigid body with moment of inertia.
        For off-center impacts, the effective mass at the impact point
        is reduced: m_eff_at_point = 1 / (1/m + r²/I), which reduces
        energy transfer to the ball (ball speed drop-off).

        For center impacts (offset=0), this reduces to the standard
        point-mass collision.

        Args:
            pre_state: Pre-impact state
            params: Impact parameters

        Returns:
            Post-impact state
        """
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        m_club_effective = self._compute_effective_club_mass(pre_state)

        n = pre_state.clubhead_orientation / (
            0.0
            if np.asarray(pre_state.clubhead_orientation, dtype=float).reshape(-1).size
            == 0
            else math.hypot(
                *np.asarray(pre_state.clubhead_orientation, dtype=float).reshape(-1)
            )
        )
        v_rel = pre_state.clubhead_velocity - pre_state.ball_velocity

        j, v_approach = self._compute_impulse(v_rel, n, m_club_effective, params.cor)

        v_ball_post = pre_state.ball_velocity + (j / GOLF_BALL_MASS_KG) * n
        v_club_post = pre_state.clubhead_velocity - (j / pre_state.clubhead_mass) * n

        ball_spin = self._compute_friction_spin(
            pre_state,
            v_rel,
            v_approach,
            n,
            j,
            params.friction_coefficient,
        )
        energy_transfer = self._compute_energy_transfer(
            pre_state.ball_velocity,
            v_ball_post,
        )

        impact_loc = (
            pre_state.impact_offset.copy()
            if pre_state.impact_offset is not None
            else np.zeros(2)
        )

        return PostImpactState(
            ball_velocity=v_ball_post,
            ball_angular_velocity=ball_spin,
            clubhead_velocity=v_club_post,
            clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy(),
            contact_duration=0.0,
            energy_transfer=energy_transfer,
            impact_location=impact_loc,
        )


class SpringDamperImpactModel(ImpactModel):
    """Spring-damper (Kelvin-Voigt) compliant contact model.

    Uses semi-implicit integration of spring-damper contact to
    compute force and velocity evolution during impact.

    Note:
        This model uses very small timesteps (default 0.1 μs) to handle
        stiff contact forces (~10 MN/m). The impact duration is typically
        ~0.5 ms, resulting in ~5000 integration steps per impact.
        For performance-critical applications, consider the
        RigidBodyImpactModel or FiniteTimeImpactModel.

    Warning:
        The spring-damper approach may exhibit numerical instability
        for very stiff contacts. If you observe blow-up (extreme
        velocities), try reducing dt or increasing damping_ratio.
        Implicit integration would provide better stability but is
        not yet implemented.
    """

    @precondition(
        lambda self, dt=1e-7: dt > 0,
        "Integration time step must be positive",
    )
    def __init__(self, dt: float = 1e-7) -> None:
        """Initialize spring-damper model.

        Args:
            dt: Integration time step [s]. Default: 0.1 μs (1e-7 s).
                Smaller values increase stability but decrease performance.
                Typical range: 1e-8 to 1e-6 s.
        """
        if dt is None:
            raise ValueError("dt must be provided")
        self.dt = dt

    @precondition(
        lambda self, pre_state, params: pre_state.clubhead_mass > 0,
        "Clubhead mass must be positive",
    )
    @precondition(
        lambda self, pre_state, params: params.contact_stiffness > 0,
        "Contact stiffness must be positive",
    )
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
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        m_ball = GOLF_BALL_MASS_KG
        m_club = pre_state.clubhead_mass

        n = pre_state.clubhead_orientation / (
            0.0
            if np.asarray(pre_state.clubhead_orientation, dtype=float).reshape(-1).size
            == 0
            else math.hypot(
                *np.asarray(pre_state.clubhead_orientation, dtype=float).reshape(-1)
            )
        )

        # Initial state - place ball at contact
        x_ball = GOLF_BALL_RADIUS_M * n  # Ball surface at origin
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
            gap = np.dot(x_ball - x_club, n) - GOLF_BALL_RADIUS_M

            if gap < 0:  # In contact (penetration)
                penetration = -gap

                # Contact force (spring-damper)
                v_rel_normal = np.dot(v_ball - v_club, n)
                f_spring = params.contact_stiffness * penetration
                f_damper = -params.contact_damping * v_rel_normal
                f_magnitude = max(0.0, min(f_spring + f_damper, max_force))

                f_contact = f_magnitude * n

                # Semi-implicit Euler: update velocities first
                # Force on ball is in direction of normal (away from club)
                a_ball = f_contact / m_ball
                # Force on club is opposite to normal (reaction force)
                a_club = -f_contact / m_club

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
                x_ball = x_ball + v_ball * self.dt  # type: ignore[assignment]
                x_club = x_club + v_club * self.dt  # type: ignore[assignment]
                # Don't increment contact_time here, it's only for contact duration

                # Check if we've reached the ball
                if np.dot(x_ball - x_club, n) - GOLF_BALL_RADIUS_M < 0:
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
        if pre_state is None:
            raise ValueError("pre_state must be provided")
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


@precondition(
    lambda impact_offset, clubhead_velocity, clubface_normal, gear_factor=0.5, h_scale=100.0, v_scale=50.0: (
        0 <= gear_factor <= 1
    ),
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
    speed = (
        0.0
        if np.asarray(clubhead_velocity, dtype=float).reshape(-1).size == 0
        else math.hypot(*np.asarray(clubhead_velocity, dtype=float).reshape(-1))
    )

    # Gear effect spin rate (empirical relationship)
    # Higher offset = more spin, proportional to speed
    horizontal_spin = -gear_factor * h_offset * speed * h_scale  # [rad/s]
    vertical_spin = gear_factor * v_offset * speed * v_scale  # [rad/s]

    # Convert to 3D spin vector
    # Assuming clubface normal is approximately in X direction
    # Vertical axis is Z, horizontal axis perpendicular to both
    up = np.array([0.0, 0.0, 1.0])
    horizontal_axis = np.cross(clubface_normal, up)
    if (
        0.0
        if np.asarray(horizontal_axis, dtype=float).reshape(-1).size == 0
        else math.hypot(*np.asarray(horizontal_axis, dtype=float).reshape(-1))
    ) > 1e-6:
        horizontal_axis /= (
            0.0
            if np.asarray(horizontal_axis, dtype=float).reshape(-1).size == 0
            else math.hypot(*np.asarray(horizontal_axis, dtype=float).reshape(-1))
        )
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
        "ball_launch_speed": float(
            0.0
            if np.asarray(post_state.ball_velocity, dtype=float).reshape(-1).size == 0
            else math.hypot(
                *np.asarray(post_state.ball_velocity, dtype=float).reshape(-1)
            )
        ),
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
    if model_type == ImpactModelType.SPRING_DAMPER:
        return SpringDamperImpactModel()
    if model_type == ImpactModelType.FINITE_TIME:
        return FiniteTimeImpactModel()
    raise ValueError(f"Unknown impact model type: {model_type}")


__all__ = [
    "FiniteTimeImpactModel",
    "ImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "PostImpactState",
    "PreImpactState",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "compute_gear_effect_spin",
    "create_impact_model",
    "validate_energy_balance",
]
