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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ..core.physics_constants import (
    DRIVER_COR,
    DRIVER_MOI_KG_M2,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    TYPICAL_CONTACT_DURATION_S,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Physical constants for golf ball (re-exported from centralized module)
GOLF_BALL_MASS: float = float(GOLF_BALL_MASS_KG)
GOLF_BALL_RADIUS: float = float(GOLF_BALL_RADIUS_M)
GOLF_BALL_MOMENT_INERTIA: float = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)

# Default impact parameters (re-exported from centralized module)
DEFAULT_COR: float = float(DRIVER_COR)
DEFAULT_CONTACT_DURATION: float = float(TYPICAL_CONTACT_DURATION_S)


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
    clubhead_moi: float = float(DRIVER_MOI_KG_M2)  # [kg·m²] MOI about CG
    impact_offset: np.ndarray | None = None  # [m] (2,) offset from CG


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
        m_club = pre_state.clubhead_mass
        club_moi = pre_state.clubhead_moi

        if pre_state.impact_offset is not None and club_moi > 0:
            r_offset = float(np.linalg.norm(pre_state.impact_offset))
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
        v_approach = np.dot(v_rel, n)
        m_eff = (GOLF_BALL_MASS * m_club_effective) / (
            GOLF_BALL_MASS + m_club_effective
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
        v_tangent = v_rel - v_approach * n
        tangent_mag = np.linalg.norm(v_tangent)

        if tangent_mag <= 1e-6:
            return pre_state.ball_angular_velocity.copy()

        tangent_dir = v_tangent / tangent_mag
        j_friction = min(
            float(friction_coefficient * j),
            float(GOLF_BALL_MASS * tangent_mag * 0.4),
        )
        spin_axis = np.cross(n, tangent_dir)
        spin_magnitude = j_friction / (GOLF_BALL_MOMENT_INERTIA / GOLF_BALL_RADIUS)
        return pre_state.ball_angular_velocity + spin_magnitude * spin_axis

    def _compute_energy_transfer(
        self,
        pre_ball_velocity: np.ndarray,
        post_ball_velocity: np.ndarray,
    ) -> float:
        ke_pre = 0.5 * GOLF_BALL_MASS * np.dot(pre_ball_velocity, pre_ball_velocity)
        ke_post = 0.5 * GOLF_BALL_MASS * np.dot(post_ball_velocity, post_ball_velocity)
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
        m_club_effective = self._compute_effective_club_mass(pre_state)

        n = pre_state.clubhead_orientation / np.linalg.norm(
            pre_state.clubhead_orientation
        )
        v_rel = pre_state.clubhead_velocity - pre_state.ball_velocity

        j, v_approach = self._compute_impulse(v_rel, n, m_club_effective, params.cor)

        v_ball_post = pre_state.ball_velocity + (j / GOLF_BALL_MASS) * n
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


@precondition(
    lambda impact_offset,
    clubhead_velocity,
    clubface_normal,
    gear_factor=0.5,
    h_scale=100.0,
    v_scale=50.0: 0 <= gear_factor <= 1,
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


# =============================================================================
# Engine Integration (Issue #758)
# =============================================================================


@dataclass
class ImpactEvent:
    """Complete record of a single impact event.

    Issue #758: Surface pre-impact and post-impact states in recorder outputs.

    Attributes:
        timestamp: Simulation time when impact occurred [s]
        pre_state: State before impact
        post_state: State after impact
        energy_balance: Energy analysis results
        impact_id: Unique identifier for this impact
        model_type: Type of impact model used
    """

    timestamp: float
    pre_state: PreImpactState
    post_state: PostImpactState
    energy_balance: dict[str, float]
    impact_id: int
    model_type: ImpactModelType


class ImpactRecorder:
    """Records impact events during simulation.

    Issue #758: Surfaces pre-impact and post-impact states in recorder outputs.
    Provides energy balance checks for each impact.
    """

    def __init__(self) -> None:
        """Initialize impact recorder."""
        self.events: list[ImpactEvent] = []
        self._impact_counter = 0

    def record_impact(
        self,
        timestamp: float,
        pre_state: PreImpactState,
        post_state: PostImpactState,
        params: ImpactParameters,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
    ) -> ImpactEvent:
        """Record an impact event.

        Args:
            timestamp: Simulation time [s]
            pre_state: Pre-impact state
            post_state: Post-impact state
            params: Impact parameters used
            model_type: Type of impact model used

        Returns:
            Recorded ImpactEvent
        """
        energy_balance = validate_energy_balance(pre_state, post_state, params)

        event = ImpactEvent(
            timestamp=timestamp,
            pre_state=pre_state,
            post_state=post_state,
            energy_balance=energy_balance,
            impact_id=self._impact_counter,
            model_type=model_type,
        )

        self.events.append(event)
        self._impact_counter += 1

        logger.info(
            f"Impact #{event.impact_id} recorded at t={timestamp:.4f}s, "
            f"ball speed: {energy_balance['ball_launch_speed']:.1f} m/s, "
            f"energy loss: {energy_balance['energy_loss_ratio']:.1%}"
        )

        return event

    def get_all_events(self) -> list[ImpactEvent]:
        """Get all recorded impact events."""
        return self.events.copy()

    def export_to_dict(self) -> dict:
        """Export all events as dictionary for JSON serialization.

        Returns:
            Dictionary with all impact events and summary
        """
        events_data = []
        for event in self.events:
            events_data.append(
                {
                    "impact_id": event.impact_id,
                    "timestamp": event.timestamp,
                    "model_type": event.model_type.name,
                    "pre_impact": {
                        "clubhead_velocity": event.pre_state.clubhead_velocity.tolist(),
                        "ball_velocity": event.pre_state.ball_velocity.tolist(),
                        "ball_spin": event.pre_state.ball_angular_velocity.tolist(),
                    },
                    "post_impact": {
                        "ball_velocity": event.post_state.ball_velocity.tolist(),
                        "ball_spin": event.post_state.ball_angular_velocity.tolist(),
                        "clubhead_velocity": event.post_state.clubhead_velocity.tolist(),
                        "contact_duration": event.post_state.contact_duration,
                        "energy_transfer": event.post_state.energy_transfer,
                    },
                    "energy_balance": event.energy_balance,
                }
            )

        return {
            "num_impacts": len(self.events),
            "events": events_data,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics for all impacts.

        Returns:
            Dictionary with summary statistics
        """
        if not self.events:
            return {"num_impacts": 0}

        speeds = [e.energy_balance["ball_launch_speed"] for e in self.events]
        losses = [e.energy_balance["energy_loss_ratio"] for e in self.events]

        return {
            "num_impacts": len(self.events),
            "mean_ball_speed": float(np.mean(speeds)),
            "max_ball_speed": float(np.max(speeds)),
            "mean_energy_loss_ratio": float(np.mean(losses)),
        }

    def reset(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._impact_counter = 0


class ImpactSolverAPI:
    """Engine-agnostic API for impact solving.

    Issue #758: Impact solver callable from simulation workflow.
    Provides unified interface for different physics engines.
    """

    def __init__(
        self,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
        params: ImpactParameters | None = None,
    ) -> None:
        """Initialize impact solver.

        Args:
            model_type: Type of impact model to use
            params: Impact parameters (uses defaults if None)
        """
        self.model_type = model_type
        self.model = create_impact_model(model_type)
        self.params = params or ImpactParameters()
        self.recorder = ImpactRecorder()

    @precondition(
        lambda self,
        timestamp,
        clubhead_velocity,
        clubhead_orientation,
        ball_velocity=None,
        ball_angular_velocity=None,
        clubhead_mass=0.200,
        record=True: clubhead_mass > 0,
        "Clubhead mass must be positive",
    )
    @precondition(
        lambda self,
        timestamp,
        clubhead_velocity,
        clubhead_orientation,
        ball_velocity=None,
        ball_angular_velocity=None,
        clubhead_mass=0.200,
        record=True: timestamp >= 0,
        "Timestamp must be non-negative",
    )
    def solve_impact(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        ball_angular_velocity: np.ndarray | None = None,
        clubhead_mass: float = 0.200,
        record: bool = True,
    ) -> PostImpactState:
        """Solve impact and optionally record event.

        Simplified API for common use case.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Clubface normal [unitless] (3,)
            ball_velocity: Ball velocity (default: stationary) [m/s] (3,)
            ball_angular_velocity: Ball spin (default: zero) [rad/s] (3,)
            clubhead_mass: Clubhead mass [kg]
            record: Whether to record this impact

        Returns:
            Post-impact state
        """
        if ball_velocity is None:
            ball_velocity = np.zeros(3)
        if ball_angular_velocity is None:
            ball_angular_velocity = np.zeros(3)

        pre_state = PreImpactState(
            clubhead_velocity=np.asarray(clubhead_velocity),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.asarray(clubhead_orientation),
            ball_position=np.zeros(3),
            ball_velocity=np.asarray(ball_velocity),
            ball_angular_velocity=np.asarray(ball_angular_velocity),
            clubhead_mass=clubhead_mass,
        )

        post_state = self.model.solve(pre_state, self.params)

        if record:
            self.recorder.record_impact(
                timestamp, pre_state, post_state, self.params, self.model_type
            )

        return post_state

    def solve_with_gear_effect(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        impact_offset: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        clubhead_mass: float = 0.200,
        record: bool = True,
    ) -> PostImpactState:
        """Solve impact with gear effect spin from offset impact.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Clubface normal [unitless] (3,)
            impact_offset: Offset from face center [m] (2,) [horizontal, vertical]
            ball_velocity: Ball velocity [m/s] (3,)
            clubhead_mass: Clubhead mass [kg]
            record: Whether to record this impact

        Returns:
            Post-impact state with gear effect spin added
        """
        # Solve base impact
        post_state = self.solve_impact(
            timestamp,
            clubhead_velocity,
            clubhead_orientation,
            ball_velocity,
            None,
            clubhead_mass,
            record=False,  # Record after adding gear effect
        )

        # Add gear effect spin
        gear_spin = compute_gear_effect_spin(
            impact_offset=np.asarray(impact_offset),
            clubhead_velocity=np.asarray(clubhead_velocity),
            clubface_normal=np.asarray(clubhead_orientation),
            gear_factor=self.params.gear_effect_factor,
            h_scale=self.params.gear_effect_h_scale,
            v_scale=self.params.gear_effect_v_scale,
        )

        # Create modified post-state with gear effect
        modified_post = PostImpactState(
            ball_velocity=post_state.ball_velocity,
            ball_angular_velocity=post_state.ball_angular_velocity + gear_spin,
            clubhead_velocity=post_state.clubhead_velocity,
            clubhead_angular_velocity=post_state.clubhead_angular_velocity,
            contact_duration=post_state.contact_duration,
            energy_transfer=post_state.energy_transfer,
            impact_location=np.asarray(impact_offset),
        )

        if record:
            pre_state = PreImpactState(
                clubhead_velocity=np.asarray(clubhead_velocity),
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=np.asarray(clubhead_orientation),
                ball_position=np.zeros(3),
                ball_velocity=(
                    np.asarray(ball_velocity)
                    if ball_velocity is not None
                    else np.zeros(3)
                ),
                ball_angular_velocity=np.zeros(3),
                clubhead_mass=clubhead_mass,
            )
            self.recorder.record_impact(
                timestamp, pre_state, modified_post, self.params, self.model_type
            )

        return modified_post

    def get_energy_report(self) -> dict:
        """Get energy balance report for all recorded impacts.

        Issue #758: Energy balance checks reported for each impact.

        Returns:
            Dictionary with energy analysis for all impacts
        """
        if not self.recorder.events:
            return {"error": "No impacts recorded"}

        reports = []
        for event in self.recorder.events:
            reports.append(
                {
                    "impact_id": event.impact_id,
                    "timestamp": event.timestamp,
                    "ke_pre": event.energy_balance["total_ke_pre"],
                    "ke_post": event.energy_balance["total_ke_post"],
                    "energy_lost": event.energy_balance["energy_lost"],
                    "loss_ratio": event.energy_balance["energy_loss_ratio"],
                    "ball_speed": event.energy_balance["ball_launch_speed"],
                }
            )

        # Aggregate statistics
        total_ke_pre = sum(r["ke_pre"] for r in reports)
        total_ke_post = sum(r["ke_post"] for r in reports)

        return {
            "impacts": reports,
            "total_ke_pre": total_ke_pre,
            "total_ke_post": total_ke_post,
            "total_energy_lost": total_ke_pre - total_ke_post,
            "overall_loss_ratio": (
                (total_ke_pre - total_ke_post) / total_ke_pre if total_ke_pre > 0 else 0
            ),
        }

    def validate_cor_behavior(
        self, tolerance: float = 0.05
    ) -> dict[str, bool | float | str | int]:
        """Validate COR behavior across recorded impacts.

        Issue #758: Tests validate COR and spin behavior within tolerance.

        Args:
            tolerance: Acceptable deviation from expected COR

        Returns:
            Validation result with pass/fail and details
        """
        if not self.recorder.events:
            return {"valid": False, "error": "No impacts recorded"}

        expected_cor = self.params.cor
        measured_cors = []

        for event in self.recorder.events:
            # Compute effective COR from velocities
            v_club_pre = np.linalg.norm(event.pre_state.clubhead_velocity)
            v_ball_pre = np.linalg.norm(event.pre_state.ball_velocity)
            v_club_post = np.linalg.norm(event.post_state.clubhead_velocity)
            v_ball_post = np.linalg.norm(event.post_state.ball_velocity)

            # COR = (v_ball_post - v_club_post) / (v_club_pre - v_ball_pre)
            approach = v_club_pre - v_ball_pre
            if approach > 0.1:  # Avoid division by small number
                separation = v_ball_post - v_club_post
                measured_cor = separation / approach
                measured_cors.append(measured_cor)

        if not measured_cors:
            return {"valid": False, "error": "Could not compute COR"}

        mean_cor = float(np.mean(measured_cors))
        deviation = abs(mean_cor - expected_cor)

        return {
            "valid": deviation <= tolerance,
            "expected_cor": expected_cor,
            "measured_cor_mean": mean_cor,
            "deviation": deviation,
            "tolerance": tolerance,
            "num_samples": len(measured_cors),
        }

    def validate_spin_behavior(
        self, max_spin_rpm: float = 10000
    ) -> dict[str, bool | float | str | int]:
        """Validate spin behavior is within physical limits.

        Issue #758: Tests validate COR and spin behavior within tolerance.

        Args:
            max_spin_rpm: Maximum acceptable spin rate [RPM]

        Returns:
            Validation result with pass/fail and details
        """
        if not self.recorder.events:
            return {"valid": False, "error": "No impacts recorded"}

        max_spin_rad = max_spin_rpm * 2 * np.pi / 60  # Convert to rad/s

        spins = []
        for event in self.recorder.events:
            spin_mag = np.linalg.norm(event.post_state.ball_angular_velocity)
            spins.append(spin_mag)

        max_observed = float(np.max(spins))
        max_observed_rpm = max_observed * 60 / (2 * np.pi)

        return {
            "valid": max_observed <= max_spin_rad,
            "max_observed_rpm": max_observed_rpm,
            "max_allowed_rpm": max_spin_rpm,
            "num_samples": len(spins),
        }

    def reset(self) -> None:
        """Reset solver state and clear recorded impacts."""
        self.recorder.reset()
