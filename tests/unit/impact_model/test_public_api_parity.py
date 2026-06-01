"""Public-API parity regression tests for the impact_model package (#7015/#7017).

The documented public import path is ``src.shared.python.physics.impact_model``.
A prior fix landed #6982 (reduced-mass expected loss), #6984 (spring-damper
contact onset), and #6986 (friction rolling cap with pre-existing spin) only in
the private ``_impact_physics`` copy. These tests lock in that the public
package exposes the same fixed behavior so pipeline/test callers using the
documented path receive the corrections.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    validate_energy_balance,
)

M_BALL = float(GOLF_BALL_MASS_KG)
R_BALL = float(GOLF_BALL_RADIUS_M)


def _head_on_state(
    v_club: float = 50.0,
    v_ball: float = 0.0,
    m_club: float = 0.2,
    ball_spin: np.ndarray | None = None,
) -> PreImpactState:
    """1-D head-on pre-impact state along +X."""
    return PreImpactState(
        clubhead_velocity=np.array([v_club, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        ball_position=np.array([R_BALL, 0.0, 0.0]),
        ball_velocity=np.array([v_ball, 0.0, 0.0]),
        ball_angular_velocity=np.zeros(3) if ball_spin is None else ball_spin,
        clubhead_mass=m_club,
    )


class TestPublicValidateEnergyBalance:
    """#6984 — expected_loss_j must be exposed via the public package."""

    @pytest.mark.unit
    def test_expected_loss_j_key_present(self) -> None:
        params = ImpactParameters(cor=0.78, friction_coefficient=0.0)
        state = _head_on_state(v_club=50.0, m_club=0.2)
        post = RigidBodyImpactModel().solve(state, params)
        result = validate_energy_balance(state, post, params)
        assert "expected_loss_j" in result

    @pytest.mark.unit
    def test_expected_loss_j_matches_reduced_mass_formula(self) -> None:
        """expected_loss_j == 0.5 * mu * v_rel_n^2 * (1 - e^2)."""
        e = 0.78
        v = 50.0
        m_club = 0.2
        params = ImpactParameters(cor=e, friction_coefficient=0.0)
        state = _head_on_state(v_club=v, m_club=m_club)
        post = RigidBodyImpactModel().solve(state, params)
        result = validate_energy_balance(state, post, params)

        mu = (M_BALL * m_club) / (M_BALL + m_club)
        expected = 0.5 * mu * v**2 * (1 - e**2)
        assert math.isclose(result["expected_loss_j"], expected, rel_tol=1e-6)


class TestPublicSpringDamperOnset:
    """#6982 — dt-independent contact onset via the public package."""

    @pytest.mark.unit
    def test_result_independent_of_dt_within_tolerance(self) -> None:
        params = ImpactParameters(contact_stiffness=1e5, contact_damping=10.0)
        state = _head_on_state(v_club=45.0, m_club=0.2)

        post_coarse = SpringDamperImpactModel(dt=1e-6).solve(state, params)
        post_fine = SpringDamperImpactModel(dt=5e-7).solve(state, params)

        coarse_speed = float(np.linalg.norm(post_coarse.ball_velocity))
        fine_speed = float(np.linalg.norm(post_fine.ball_velocity))
        assert abs(coarse_speed - fine_speed) / fine_speed < 0.05


class TestPublicFrictionRollingCap:
    """#6986 — pre-existing spin reduces the friction impulse via public package."""

    @pytest.mark.unit
    def test_backspin_reduces_rolling_cap(self) -> None:
        params = ImpactParameters(cor=0.78, friction_coefficient=0.5)
        m_club = 0.2

        state_no_spin = PreImpactState(
            clubhead_velocity=np.array([45.0, 5.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([R_BALL, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=m_club,
        )
        state_with_spin = PreImpactState(
            clubhead_velocity=np.array([45.0, 5.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([R_BALL, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.array([0.0, 0.0, 100.0]),
            clubhead_mass=m_club,
        )

        model = RigidBodyImpactModel()
        post_no_spin = model.solve(state_no_spin, params)
        post_with_spin = model.solve(state_with_spin, params)

        added_z_no_spin = float(post_no_spin.ball_angular_velocity[2])
        added_z_with_spin = float(post_with_spin.ball_angular_velocity[2]) - 100.0
        assert added_z_with_spin < added_z_no_spin
