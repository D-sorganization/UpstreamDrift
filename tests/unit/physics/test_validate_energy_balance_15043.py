"""Physics-derived tests for validate_energy_balance (#15043).

The validator must use effective normal mass (club translational inertia at an
off-center contact) and account for 6-DOF kinetic energy when comparing against
the central frictionless COR loss budget. Tangential friction dissipation is
documented separately from the normal expected-loss term.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG
from src.shared.python.physics.impact_model import (
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    validate_energy_balance,
)
from src.shared.python.physics.impact_model.utils import (
    compute_effective_club_mass,
    impact_offset_vector_3d,
    pre_impact_contact_relative_velocity,
    reduced_normal_effective_mass,
)


def _lofted_frictional_pre_state() -> PreImpactState:
    """6-DOF lofted strike with friction and a toe/high face offset."""
    loft_rad = math.radians(10.5)
    face_normal = np.array([math.cos(loft_rad), 0.0, math.sin(loft_rad)])
    return PreImpactState(
        clubhead_velocity=np.array([50.0, 1.5, -3.0]),
        clubhead_angular_velocity=np.array([0.0, 8.0, 2.0]),
        clubhead_orientation=face_normal,
        ball_position=np.array([0.021, 0.0, 0.0]),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.array([0.0, 40.0, 0.0]),
        clubhead_mass=0.2,
        clubhead_moi=4.5e-4,
        clubhead_loft=loft_rad,
        impact_offset=np.array([0.015, 0.005]),
    )


def _effective_club_mass(pre_state: PreImpactState) -> float:
    return compute_effective_club_mass(pre_state)


def _unit_normal(orientation: np.ndarray) -> np.ndarray:
    raw = np.asarray(orientation, dtype=float).reshape(-1)
    norm = float(math.hypot(*raw)) if raw.size else 0.0
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return raw / norm


@pytest.mark.unit
class TestValidateEnergyBalance15043:
    def test_expected_loss_uses_effective_club_mass(self) -> None:
        """Off-center contact must use 1/(1/m + r²/I), not bare clubhead mass."""
        pre = _lofted_frictional_pre_state()
        params = ImpactParameters(cor=0.78, friction_coefficient=0.0)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        m_eff = _effective_club_mass(pre)
        assert m_eff < pre.clubhead_mass
        n = _unit_normal(pre.clubhead_orientation)
        v_rel = pre_impact_contact_relative_velocity(pre)
        v_rel_n = float(np.dot(v_rel, n))
        mu = reduced_normal_effective_mass(pre, n)
        expected = 0.5 * mu * v_rel_n**2 * (1.0 - params.cor**2)
        assert math.isclose(result["expected_loss_j"], expected, rel_tol=1e-6)

        mu_point = (float(GOLF_BALL_MASS_KG) * pre.clubhead_mass) / (
            float(GOLF_BALL_MASS_KG) + pre.clubhead_mass
        )
        point_mass_budget = 0.5 * mu_point * v_rel_n**2 * (1.0 - params.cor**2)
        assert result["expected_loss_j"] < point_mass_budget

    def test_total_ke_includes_club_rotational_energy(self) -> None:
        """Pre/post KE sums must include club rotational kinetic energy."""
        pre = _lofted_frictional_pre_state()
        params = ImpactParameters(cor=0.78, friction_coefficient=0.2)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        omega_pre = np.asarray(pre.clubhead_angular_velocity, dtype=float)
        omega_post = np.asarray(post.clubhead_angular_velocity, dtype=float)
        rot_pre = 0.5 * pre.clubhead_moi * float(np.dot(omega_pre, omega_pre))
        rot_post = 0.5 * pre.clubhead_moi * float(np.dot(omega_post, omega_post))
        assert result["club_rot_ke_pre"] == pytest.approx(rot_pre, rel=1e-9)
        assert result["club_rot_ke_post"] == pytest.approx(rot_post, rel=1e-9)
        assert rot_pre > 0.0

    def test_lofted_frictional_fixture_normal_budget_coherence(self) -> None:
        """Measured loss exceeds normal COR budget; surplus is friction, not a false fail."""
        pre = _lofted_frictional_pre_state()
        params = ImpactParameters(cor=0.78, friction_coefficient=0.35)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        assert result["energy_lost"] >= result["expected_loss_j"]
        surplus = result["energy_lost"] - result["expected_loss_j"]
        assert surplus > 1e-6
        assert result["normal_energy_budget_met"] is True

    def test_tangential_friction_documented_out_of_expected_loss(self) -> None:
        """Tangential friction work is excluded from expected_loss_j with an explicit reason."""
        pre = _lofted_frictional_pre_state()
        params = ImpactParameters(cor=0.78, friction_coefficient=0.35)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        assert result["tangential_friction_in_expected_loss"] is False
        assert "friction_coefficient" in result
        assert isinstance(result["tangential_friction_exclusion_reason"], str)
        assert len(result["tangential_friction_exclusion_reason"]) > 20

    def test_solver_frictionless_off_center_meets_cor_energy_budget(self) -> None:
        """Rigid-body solver output must satisfy the same COR energy budget as the validator."""
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.015, 0.005]),
        )
        params = ImpactParameters(cor=0.78, friction_coefficient=0.0)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        assert result["normal_energy_budget_met"] is True
        assert math.isclose(
            result["energy_lost"], result["expected_loss_j"], rel_tol=1e-9
        )
        assert result["club_rot_ke_post"] > result["club_rot_ke_pre"]

    def test_spinning_off_center_frictionless_momentum_and_cor_energy(self) -> None:
        """Off-center spin changes contact v_rel; impulse conserves P/L and COR energy."""
        pre = PreImpactState(
            clubhead_velocity=np.array([48.0, 0.2, -0.5]),
            clubhead_angular_velocity=np.array([0.0, 12.0, -4.0]),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.018, 0.006]),
        )
        params = ImpactParameters(cor=0.82, friction_coefficient=0.0)
        post = RigidBodyImpactModel().solve(pre, params)
        result = validate_energy_balance(pre, post, params)

        m_club = float(pre.clubhead_mass)
        p_pre = m_club * np.asarray(pre.clubhead_velocity) + float(
            GOLF_BALL_MASS_KG
        ) * np.asarray(pre.ball_velocity)
        p_post = m_club * np.asarray(post.clubhead_velocity) + float(
            GOLF_BALL_MASS_KG
        ) * np.asarray(post.ball_velocity)
        assert np.linalg.norm(p_post - p_pre) < 1e-6

        n = _unit_normal(pre.clubhead_orientation)
        v_rel = pre_impact_contact_relative_velocity(pre)
        mu = reduced_normal_effective_mass(pre, n)
        j_n = (1.0 + params.cor) * mu * float(np.dot(v_rel, n))
        impulse = j_n * n
        i_club = float(pre.clubhead_moi)
        r_club = impact_offset_vector_3d(pre)
        delta_omega_club = np.cross(r_club, -impulse) / i_club
        assert np.allclose(
            post.clubhead_angular_velocity - pre.clubhead_angular_velocity,
            delta_omega_club,
            atol=1e-9,
        )
        assert np.allclose(
            post.ball_angular_velocity, pre.ball_angular_velocity, atol=1e-12
        )

        assert result["normal_energy_budget_met"] is True
        assert math.isclose(
            result["energy_lost"], result["expected_loss_j"], rel_tol=1e-8
        )
