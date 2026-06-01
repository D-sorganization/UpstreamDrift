"""Value-asserting tests for impact physics (#6993, #6982, #6984, #6986).

The existing ``test_impact_model_split_2456.py`` suite is smoke-only (module
exists / LOC budget). These tests pin the *numerical* behaviour of the public
impact-model API against hand-computed closed-form results:

- ``RigidBodyImpactModel.solve`` post-impact speeds, launch direction, spin.
- ``_compute_effective_club_mass`` / ``_compute_impulse`` /
  ``_compute_friction_spin`` / ``_compute_energy_transfer`` closed-form values.
- ``compute_gear_effect_spin`` centre / toe / heel sign behaviour.
- ``validate_energy_balance`` relative-frame expected loss (#6984).
- ``SpringDamperImpactModel`` momentum conservation (#6982).
- friction rolling cap that accounts for incoming ball spin (#6986).
- ``create_impact_model`` enum dispatch and negative/invalid inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.physics.impact_model import (
    FiniteTimeImpactModel,
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)

M_BALL = float(GOLF_BALL_MASS_KG)
R_BALL = float(GOLF_BALL_RADIUS_M)
I_BALL = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)


def _make_state(
    *,
    clubhead_velocity: np.ndarray,
    ball_angular_velocity: np.ndarray | None = None,
    clubhead_mass: float = 0.200,
    clubhead_moi: float = 4.5e-4,
    impact_offset: np.ndarray | None = None,
) -> PreImpactState:
    return PreImpactState(
        clubhead_velocity=clubhead_velocity,
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=(
            np.zeros(3) if ball_angular_velocity is None else ball_angular_velocity
        ),
        clubhead_mass=clubhead_mass,
        clubhead_moi=clubhead_moi,
        impact_offset=impact_offset,
    )


class TestRigidBodySolveClosedForm:
    """Post-impact velocities vs hand-computed 1D collision."""

    @pytest.mark.unit
    def test_center_hit_ball_and_club_speed(self) -> None:
        speed = 50.0
        cor = 0.8
        m_club = 0.200
        state = _make_state(
            clubhead_velocity=np.array([speed, 0.0, 0.0]),
            clubhead_mass=m_club,
        )
        params = ImpactParameters(cor=cor, friction_coefficient=0.0)
        result = RigidBodyImpactModel().solve(state, params)

        m_eff = (M_BALL * m_club) / (M_BALL + m_club)
        j = (1.0 + cor) * m_eff * speed
        expected_ball = j / M_BALL
        expected_club = speed - j / m_club

        np.testing.assert_allclose(
            result.ball_velocity, [expected_ball, 0.0, 0.0], rtol=1e-12
        )
        np.testing.assert_allclose(
            result.clubhead_velocity, [expected_club, 0.0, 0.0], rtol=1e-12
        )

    @pytest.mark.unit
    def test_launch_along_normal_for_square_hit(self) -> None:
        """A square (no tangential) hit launches the ball along the normal."""
        state = _make_state(clubhead_velocity=np.array([45.0, 0.0, 0.0]))
        params = ImpactParameters(cor=0.78, friction_coefficient=0.0)
        result = RigidBodyImpactModel().solve(state, params)
        assert result.ball_velocity[0] > 0.0
        assert result.ball_velocity[1] == pytest.approx(0.0, abs=1e-12)
        assert result.ball_velocity[2] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.unit
    def test_zero_cor_is_perfectly_inelastic(self) -> None:
        """COR=0 ⇒ ball and club reach the common (reduced) speed factor (1+0)."""
        state = _make_state(clubhead_velocity=np.array([50.0, 0.0, 0.0]))
        params = ImpactParameters(cor=0.0, friction_coefficient=0.0)
        result = RigidBodyImpactModel().solve(state, params)
        m_eff = (M_BALL * 0.200) / (M_BALL + 0.200)
        j = 1.0 * m_eff * 50.0
        np.testing.assert_allclose(result.ball_velocity[0], j / M_BALL, rtol=1e-12)


class TestRigidBodyHelpers:
    """Closed-form checks of the private helper methods."""

    @pytest.mark.unit
    def test_effective_club_mass_center(self) -> None:
        model = RigidBodyImpactModel()
        state = _make_state(clubhead_velocity=np.array([45.0, 0.0, 0.0]))
        # No offset ⇒ full mass.
        assert model._compute_effective_club_mass(state) == pytest.approx(0.200)

    @pytest.mark.unit
    def test_effective_club_mass_offset(self) -> None:
        model = RigidBodyImpactModel()
        r = 0.025
        state = _make_state(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubhead_moi=4.5e-4,
            impact_offset=np.array([r, 0.0]),
        )
        expected = 1.0 / (1.0 / 0.200 + r**2 / 4.5e-4)
        assert model._compute_effective_club_mass(state) == pytest.approx(expected)

    @pytest.mark.unit
    def test_compute_impulse_closed_form(self) -> None:
        model = RigidBodyImpactModel()
        v_rel = np.array([50.0, 0.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        m_club = 0.200
        cor = 0.8
        j, v_approach = model._compute_impulse(v_rel, n, m_club, cor)
        m_eff = (M_BALL * m_club) / (M_BALL + m_club)
        assert v_approach == pytest.approx(50.0)
        assert j == pytest.approx((1.0 + cor) * m_eff * 50.0)

    @pytest.mark.unit
    def test_compute_friction_spin_closed_form(self) -> None:
        model = RigidBodyImpactModel()
        state = _make_state(clubhead_velocity=np.array([50.0, 0.0, 10.0]))
        n = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([50.0, 0.0, 10.0])
        v_approach = 50.0
        m_eff = (M_BALL * 0.200) / (M_BALL + 0.200)
        j = (1.0 + 0.8) * m_eff * v_approach
        spin = model._compute_friction_spin(state, v_rel, v_approach, n, j, 0.2)

        v_tangent = v_rel - v_approach * n
        tangent_mag = float(np.linalg.norm(v_tangent))
        tangent_dir = v_tangent / tangent_mag
        # No incoming spin ⇒ slip == tangent_mag.
        j_friction = min(0.2 * j, M_BALL * tangent_mag * 0.4)
        spin_mag = j_friction / (I_BALL / R_BALL)
        expected = spin_mag * np.cross(n, tangent_dir)
        np.testing.assert_allclose(spin, expected, rtol=1e-12)

    @pytest.mark.unit
    def test_compute_energy_transfer_closed_form(self) -> None:
        model = RigidBodyImpactModel()
        pre = np.zeros(3)
        post = np.array([70.0, 0.0, 0.0])
        expected = 0.5 * M_BALL * 70.0**2
        assert model._compute_energy_transfer(pre, post) == pytest.approx(expected)


class TestFrictionCapHonoursBallSpin:
    """#6986: rolling cap is relative to slip (v_t - ω·R), not bulk v_t."""

    @pytest.mark.unit
    def test_prespin_changes_friction_limited_spin(self) -> None:
        model = RigidBodyImpactModel()
        # High friction so the *rolling* cap binds rather than μ·J.
        params = ImpactParameters(cor=0.8, friction_coefficient=0.5)

        def solve_with_spin(omega_y: float) -> float:
            state = _make_state(
                clubhead_velocity=np.array([50.0, 0.0, 10.0]),
                ball_angular_velocity=np.array([0.0, omega_y, 0.0]),
            )
            result = model.solve(state, params)
            return float(result.ball_angular_velocity[1] - omega_y)

        # omega_y>0 reduces slip ⇒ smaller friction spin delta;
        # omega_y<0 increases slip ⇒ larger delta. The old code ignored spin
        # entirely and produced an identical delta for all three.
        delta_zero = solve_with_spin(0.0)
        delta_pos = solve_with_spin(200.0)
        delta_neg = solve_with_spin(-200.0)
        assert abs(delta_pos) < abs(delta_zero) < abs(delta_neg)


class TestGearEffectSpin:
    """#6993: gear-effect sign behaviour for centre / toe / heel."""

    @pytest.mark.unit
    def test_center_hit_no_gear_spin(self) -> None:
        spin = compute_gear_effect_spin(
            np.array([0.0, 0.0]),
            np.array([50.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(spin, np.zeros(3), atol=1e-12)

    @pytest.mark.unit
    def test_toe_and_heel_opposite_sidespin(self) -> None:
        velocity = np.array([50.0, 0.0, 0.0])
        normal = np.array([1.0, 0.0, 0.0])
        toe = compute_gear_effect_spin(np.array([0.01, 0.0]), velocity, normal)
        heel = compute_gear_effect_spin(np.array([-0.01, 0.0]), velocity, normal)
        # Sidespin lives on the vertical (z) axis and flips sign toe↔heel.
        assert toe[2] == pytest.approx(-heel[2])
        assert toe[2] != pytest.approx(0.0)


class TestEnergyBalanceRelativeFrame:
    """#6984: expected loss is ½·μ·v_rel²·(1-e²), not a lab-frame fraction."""

    @pytest.mark.unit
    def test_expected_energy_loss_matches_actual_for_rigid_impact(self) -> None:
        state = _make_state(clubhead_velocity=np.array([50.0, 0.0, 0.0]))
        params = ImpactParameters(cor=0.8, friction_coefficient=0.0)
        post = RigidBodyImpactModel().solve(state, params)
        report = validate_energy_balance(state, post, params)

        m_club = 0.200
        mu = (M_BALL * m_club) / (M_BALL + m_club)
        v_rel = 50.0
        expected = 0.5 * mu * v_rel**2 * (1.0 - 0.8**2)
        assert report["expected_energy_loss"] == pytest.approx(expected)
        # For a normal-direction rigid impact the actual lab-frame loss equals
        # the relative-frame loss, so the new key is directly comparable.
        assert report["energy_lost"] == pytest.approx(
            report["expected_energy_loss"], rel=1e-9
        )

    @pytest.mark.unit
    def test_lab_frame_fraction_is_not_one_minus_cor_sq(self) -> None:
        """Regression: the lab-frame ratio is far below (1-e²) (#6984)."""
        state = _make_state(clubhead_velocity=np.array([50.0, 0.0, 0.0]))
        params = ImpactParameters(cor=0.8, friction_coefficient=0.0)
        post = RigidBodyImpactModel().solve(state, params)
        report = validate_energy_balance(state, post, params)
        # The lab-frame fraction (~0.067) is far below (1-e²)=0.36 because the
        # club retains most of its KE: the two quantities are incomparable.
        assert report["energy_loss_ratio"] < 0.5 * (1.0 - 0.8**2)


class TestSpringDamperMomentumConservation:
    """#6982: spring-damper impact conserves linear momentum."""

    @pytest.mark.unit
    def test_momentum_conserved(self) -> None:
        state = _make_state(clubhead_velocity=np.array([40.0, 0.0, 0.0]))
        params = ImpactParameters()
        post = SpringDamperImpactModel(dt=1e-7).solve(state, params)

        dv_ball = post.ball_velocity - state.ball_velocity
        dv_club = post.clubhead_velocity - state.clubhead_velocity
        residual = M_BALL * dv_ball + state.clubhead_mass * dv_club
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-9)

    @pytest.mark.unit
    def test_contact_detected_and_ball_accelerated(self) -> None:
        state = _make_state(clubhead_velocity=np.array([40.0, 0.0, 0.0]))
        params = ImpactParameters()
        post = SpringDamperImpactModel(dt=1e-7).solve(state, params)
        assert post.contact_duration > 0.0
        assert post.ball_velocity[0] > 0.0

    @pytest.mark.unit
    def test_dt_insensitive_ball_speed(self) -> None:
        """Post-impact ball speed is stable across dt (#6982)."""
        state = _make_state(clubhead_velocity=np.array([40.0, 0.0, 0.0]))
        params = ImpactParameters()
        fine = SpringDamperImpactModel(dt=5e-8).solve(state, params)
        coarse = SpringDamperImpactModel(dt=1e-7).solve(state, params)
        assert fine.ball_velocity[0] == pytest.approx(coarse.ball_velocity[0], rel=0.05)


class TestFactoryAndValidation:
    """create_impact_model dispatch and precondition failures (#6993)."""

    @pytest.mark.unit
    def test_factory_dispatch(self) -> None:
        assert isinstance(
            create_impact_model(ImpactModelType.RIGID_BODY), RigidBodyImpactModel
        )
        assert isinstance(
            create_impact_model(ImpactModelType.SPRING_DAMPER),
            SpringDamperImpactModel,
        )
        assert isinstance(
            create_impact_model(ImpactModelType.FINITE_TIME), FiniteTimeImpactModel
        )

    @pytest.mark.unit
    def test_negative_clubhead_mass_raises(self) -> None:
        state = _make_state(
            clubhead_velocity=np.array([45.0, 0.0, 0.0]), clubhead_mass=-0.1
        )
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(state, ImpactParameters())

    @pytest.mark.unit
    def test_cor_above_one_raises(self) -> None:
        state = _make_state(clubhead_velocity=np.array([45.0, 0.0, 0.0]))
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(state, ImpactParameters(cor=1.5))

    @pytest.mark.unit
    def test_negative_friction_raises(self) -> None:
        state = _make_state(clubhead_velocity=np.array([45.0, 0.0, 0.0]))
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(
                state, ImpactParameters(friction_coefficient=-0.1)
            )
