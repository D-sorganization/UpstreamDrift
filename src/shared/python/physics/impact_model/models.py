import math
from abc import ABC, abstractmethod

import numpy as np

from src.shared.python.contracts import require, require_finite
from src.shared.python.core.contracts import precondition
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)

from .types import ImpactModelType, ImpactParameters, PostImpactState, PreImpactState


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
        # Rolling-without-slip cap must use the *slip* velocity at the contact
        # point, not the raw tangential velocity (issue #6986). The contact
        # point sits at -R*n from the ball centre; its surface velocity from
        # spin is omega x (-R*n). The slip along the tangent is the tangential
        # closing speed minus that surface speed: v_slip = v_t - (omega x r)·t.
        contact_arm = -GOLF_BALL_RADIUS_M * n
        surface_vel_tangent = float(
            np.dot(np.cross(pre_state.ball_angular_velocity, contact_arm), tangent_dir)
        )
        v_slip = tangent_mag - surface_vel_tangent
        slip_cap = max(0.0, float(GOLF_BALL_MASS_KG * v_slip * 0.4))
        j_friction = min(float(friction_coefficient * j), slip_cap)
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

        Args:
            pre_state: Pre-impact state
            params: Impact parameters

        Returns:
            Post-impact state
        """
        require_finite(pre_state.clubhead_velocity, "clubhead_velocity")
        require_finite(pre_state.ball_velocity, "ball_velocity")
        require_finite(pre_state.clubhead_orientation, "clubhead_orientation")
        require(
            bool(
                (
                    0.0
                    if np.asarray(pre_state.clubhead_orientation, dtype=float)
                    .reshape(-1)
                    .size
                    == 0
                    else math.hypot(
                        *np.asarray(
                            pre_state.clubhead_orientation, dtype=float
                        ).reshape(-1)
                    )
                )
                > 1e-10
            ),
            "clubhead_orientation must be non-zero",
        )
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
    """

    @precondition(
        lambda self, dt=1e-7: dt > 0,
        "Integration time step must be positive",
    )
    def __init__(self, dt: float = 1e-7) -> None:
        """Initialize spring-damper model.

        Args:
            dt: Integration time step [s]. Default: 0.1 μs (1e-7 s).
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

        # Initial state. Start the ball at a small *positive* gap and let the
        # solver event-detect the contact onset (issue #6982). Starting exactly
        # touching (gap == 0) made the onset depend on dt overshoot and on
        # rounding of the initial separation. The starting separation is
        # immaterial to the result because the pre-contact phase is free flight.
        initial_gap = 1e-4  # [m] small positive separation
        x_ball: np.ndarray = (GOLF_BALL_RADIUS_M + initial_gap) * n
        v_ball: np.ndarray = pre_state.ball_velocity.copy()
        x_club: np.ndarray = np.zeros(3)
        v_club: np.ndarray = pre_state.clubhead_velocity.copy()

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

                # Contact force (spring-damper). The same f_contact is applied
                # equal-and-opposite to ball and club, so linear momentum is
                # conserved exactly regardless of dt or the force clamp.
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
                # Pre-contact free flight: advance positions only. Contact onset
                # is detected on the next iteration's gap test, independent of
                # the (arbitrary) initial separation.
                x_ball = x_ball + v_ball * self.dt  # type: ignore[assignment]
                x_club = x_club + v_club * self.dt  # type: ignore[assignment]

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
