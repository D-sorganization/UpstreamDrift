"""Value-asserting tests for _impact_physics.py.

Covers issues #6982 (SpringDamperImpactModel dt-sensitivity / momentum conservation),
#6984 (validate_energy_balance reduced-mass expected loss),
#6986 (friction rolling cap with pre-existing spin), and
#6993 (numerical behavior, not just smoke tests).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.shared.python.physics._impact_physics import (
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    compute_gear_effect_spin,
    create_impact_model,
    validate_energy_balance,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

M_BALL = float(GOLF_BALL_MASS_KG)
R_BALL = float(GOLF_BALL_RADIUS_M)
I_BALL = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)


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


# ---------------------------------------------------------------------------
# RigidBodyImpactModel — numerical value checks  (#6993)
# ---------------------------------------------------------------------------


class TestRigidBodyImpactModelValues:
    """Verify post-impact velocities match impulse-momentum closed form."""

    @pytest.mark.unit
    def test_ball_speed_matches_analytical(self) -> None:
        """Ball post-impact speed equals (1+e)*M/(m+M)*V_club for head-on."""
        e = 0.78
        V = 50.0
        M = 0.2
        m = M_BALL

        params = ImpactParameters(cor=e, friction_coefficient=0.0)
        state = _head_on_state(v_club=V, m_club=M)
        model = RigidBodyImpactModel()
        post = model.solve(state, params)

        expected_v_ball = (1 + e) * M / (m + M) * V
        assert math.isclose(float(post.ball_velocity[0]), expected_v_ball, rel_tol=1e-6)

    @pytest.mark.unit
    def test_club_speed_matches_analytical(self) -> None:
        """Club post-impact speed equals (M-e*m)/(m+M)*V_club for head-on."""
        e = 0.78
        V = 50.0
        M = 0.2
        m = M_BALL

        params = ImpactParameters(cor=e, friction_coefficient=0.0)
        state = _head_on_state(v_club=V, m_club=M)
        model = RigidBodyImpactModel()
        post = model.solve(state, params)

        expected_v_club = (M - e * m) / (m + M) * V
        assert math.isclose(
            float(post.clubhead_velocity[0]), expected_v_club, rel_tol=1e-6
        )

    @pytest.mark.unit
    def test_momentum_conserved(self) -> None:
        """Total linear momentum must be conserved (Newton's 3rd law)."""
        M = 0.2
        m = M_BALL
        params = ImpactParameters(cor=0.83, friction_coefficient=0.15)
        state = _head_on_state(v_club=45.0, m_club=M)
        model = RigidBodyImpactModel()
        post = model.solve(state, params)

        p_pre = M * state.clubhead_velocity + m * state.ball_velocity
        p_post = M * post.clubhead_velocity + m * post.ball_velocity
        np.testing.assert_allclose(p_pre, p_post, atol=1e-10)

    @pytest.mark.unit
    def test_cor_condition(self) -> None:
        """Separation speed / approach speed == COR along contact normal."""
        e = 0.75
        V = 40.0
        M = 0.2
        params = ImpactParameters(cor=e, friction_coefficient=0.0)
        state = _head_on_state(v_club=V, m_club=M)
        n = np.array([1.0, 0.0, 0.0])
        model = RigidBodyImpactModel()
        post = model.solve(state, params)

        v_app = float(np.dot(state.clubhead_velocity - state.ball_velocity, n))
        v_sep = float(np.dot(post.ball_velocity - post.clubhead_velocity, n))
        assert math.isclose(v_sep, e * v_app, rel_tol=1e-6)

    @pytest.mark.unit
    def test_energy_transfer_positive_for_stationary_ball(self) -> None:
        """Energy transferred to a stationary ball is always positive."""
        params = ImpactParameters(cor=0.78, friction_coefficient=0.15)
        state = _head_on_state(v_club=45.0, v_ball=0.0)
        model = RigidBodyImpactModel()
        post = model.solve(state, params)
        assert post.energy_transfer > 0


class TestComputeEffectiveClubMass:
    """Unit tests for _compute_effective_club_mass."""

    @pytest.mark.unit
    def test_center_impact_equals_full_mass(self) -> None:
        """With no offset, effective mass equals full clubhead mass."""
        state = _head_on_state(m_club=0.2)
        state.impact_offset = None
        model = RigidBodyImpactModel()
        assert model._compute_effective_club_mass(state) == pytest.approx(0.2)

    @pytest.mark.unit
    def test_off_center_reduces_effective_mass(self) -> None:
        """Off-center offset reduces effective mass below full mass."""
        state = _head_on_state(m_club=0.2)
        state.impact_offset = np.array([0.02, 0.0])  # 2 cm toe
        model = RigidBodyImpactModel()
        m_eff = model._compute_effective_club_mass(state)
        assert m_eff < 0.2

    @pytest.mark.unit
    def test_effective_mass_formula(self) -> None:
        """Effective mass = 1/(1/m + r²/moi) for point-offset."""
        m = 0.2
        moi = float("4.5e-4")
        r = 0.02  # offset [m]
        expected = 1.0 / (1.0 / m + r**2 / moi)

        state = _head_on_state(m_club=m)
        state.impact_offset = np.array([r, 0.0])
        model = RigidBodyImpactModel()
        assert model._compute_effective_club_mass(state) == pytest.approx(
            expected, rel=1e-6
        )


class TestComputeImpulse:
    """Unit tests for _compute_impulse."""

    @pytest.mark.unit
    def test_impulse_formula(self) -> None:
        """j = (1+e)*mu*v_approach, mu = reduced mass."""
        m_eff = 0.18
        e = 0.78
        V = 50.0
        v_rel = np.array([V, 0.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        mu = (M_BALL * m_eff) / (M_BALL + m_eff)
        expected_j = (1 + e) * mu * V

        model = RigidBodyImpactModel()
        j, v_app = model._compute_impulse(v_rel, n, m_eff, e)
        assert math.isclose(j, expected_j, rel_tol=1e-9)
        assert math.isclose(v_app, V, rel_tol=1e-9)


class TestComputeEnergyTransfer:
    """Unit tests for _compute_energy_transfer."""

    @pytest.mark.unit
    def test_stationary_to_moving(self) -> None:
        """From rest to speed v: energy transfer = 0.5*m*v^2."""
        v = 60.0
        model = RigidBodyImpactModel()
        et = model._compute_energy_transfer(
            np.array([0.0, 0.0, 0.0]),
            np.array([v, 0.0, 0.0]),
        )
        expected = 0.5 * M_BALL * v**2
        assert math.isclose(et, expected, rel_tol=1e-9)

    @pytest.mark.unit
    def test_deceleration_is_negative(self) -> None:
        """Slowing down the ball results in negative energy transfer."""
        model = RigidBodyImpactModel()
        et = model._compute_energy_transfer(
            np.array([60.0, 0.0, 0.0]),
            np.array([30.0, 0.0, 0.0]),
        )
        assert et < 0


# ---------------------------------------------------------------------------
# compute_gear_effect_spin  (#6993)
# ---------------------------------------------------------------------------


class TestComputeGearEffectSpin:
    @pytest.mark.unit
    def test_center_impact_zero_spin(self) -> None:
        """Zero offset → zero gear-effect spin."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.0, 0.0]),
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(spin, np.zeros(3), atol=1e-10)

    @pytest.mark.unit
    def test_toe_impact_produces_draw_spin(self) -> None:
        """Toe impact (positive h_offset) → negative Z-axis spin (draw)."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([0.02, 0.0]),  # toe
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert spin[2] < 0

    @pytest.mark.unit
    def test_heel_impact_produces_fade_spin(self) -> None:
        """Heel impact (negative h_offset) → positive Z-axis spin (fade)."""
        spin = compute_gear_effect_spin(
            impact_offset=np.array([-0.02, 0.0]),  # heel
            clubhead_velocity=np.array([45.0, 0.0, 0.0]),
            clubface_normal=np.array([1.0, 0.0, 0.0]),
        )
        assert spin[2] > 0

    @pytest.mark.unit
    def test_toe_heel_are_opposite(self) -> None:
        """Toe and heel impacts produce equal-and-opposite spin vectors."""
        v = np.array([45.0, 0.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        spin_toe = compute_gear_effect_spin(np.array([0.02, 0.0]), v, n)
        spin_heel = compute_gear_effect_spin(np.array([-0.02, 0.0]), v, n)
        np.testing.assert_allclose(spin_toe, -spin_heel, rtol=1e-9)


# ---------------------------------------------------------------------------
# validate_energy_balance  (#6984, #6993)
# ---------------------------------------------------------------------------


class TestValidateEnergyBalance:
    @pytest.mark.unit
    def test_contains_required_keys(self) -> None:
        params = ImpactParameters(cor=0.78)
        state = _head_on_state()
        model = RigidBodyImpactModel()
        post = model.solve(state, params)
        result = validate_energy_balance(state, post, params)
        for key in (
            "total_ke_pre",
            "total_ke_post",
            "energy_lost",
            "energy_loss_ratio",
            "expected_loss_factor",
            "expected_loss_j",
        ):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.unit
    def test_energy_lost_positive_for_inelastic(self) -> None:
        """Inelastic COR < 1 must lose energy."""
        params = ImpactParameters(cor=0.78)
        state = _head_on_state()
        model = RigidBodyImpactModel()
        post = model.solve(state, params)
        result = validate_energy_balance(state, post, params)
        assert result["energy_lost"] > 0

    @pytest.mark.unit
    def test_expected_loss_j_matches_reduced_mass_formula(self) -> None:
        """expected_loss_j == 0.5 * mu * v_rel_n^2 * (1 - e^2)."""
        e = 0.78
        V = 50.0
        M = 0.2
        m = M_BALL

        params = ImpactParameters(cor=e, friction_coefficient=0.0)
        state = _head_on_state(v_club=V, m_club=M)
        model = RigidBodyImpactModel()
        post = model.solve(state, params)
        result = validate_energy_balance(state, post, params)

        mu = (m * M) / (m + M)
        v_rel_n = V  # ball stationary
        expected = 0.5 * mu * v_rel_n**2 * (1 - e**2)
        assert math.isclose(result["expected_loss_j"], expected, rel_tol=1e-6)

    @pytest.mark.unit
    def test_elastic_collision_loses_no_energy(self) -> None:
        """COR=1 (elastic) → energy_lost ≈ 0 for head-on impact."""
        params = ImpactParameters(cor=1.0, friction_coefficient=0.0)
        state = _head_on_state()
        model = RigidBodyImpactModel()
        post = model.solve(state, params)
        result = validate_energy_balance(state, post, params)
        assert math.isclose(result["energy_lost"], 0.0, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# SpringDamperImpactModel — momentum conservation  (#6982)
# ---------------------------------------------------------------------------


class TestSpringDamperMomentumConservation:
    @pytest.mark.unit
    def test_momentum_conserved(self) -> None:
        """m_ball*(v_ball_post-v_ball_pre) == -m_club*(v_club_post-v_club_pre)."""
        M = 0.2
        m = M_BALL
        params = ImpactParameters(
            contact_stiffness=1e5,
            contact_damping=10.0,
        )
        state = _head_on_state(v_club=45.0, m_club=M)
        model = SpringDamperImpactModel(dt=1e-6)
        post = model.solve(state, params)

        delta_p_ball = m * (post.ball_velocity - state.ball_velocity)
        delta_p_club = M * (post.clubhead_velocity - state.clubhead_velocity)
        np.testing.assert_allclose(delta_p_ball, -delta_p_club, atol=1e-4)

    @pytest.mark.unit
    def test_ball_gains_positive_x_velocity(self) -> None:
        """Ball must move forward after a head-on impact."""
        params = ImpactParameters(contact_stiffness=1e5, contact_damping=10.0)
        state = _head_on_state(v_club=45.0)
        model = SpringDamperImpactModel(dt=1e-6)
        post = model.solve(state, params)
        assert post.ball_velocity[0] > 0

    @pytest.mark.unit
    def test_contact_duration_positive(self) -> None:
        """Contact duration must be positive (contact actually happened)."""
        params = ImpactParameters(contact_stiffness=1e5, contact_damping=10.0)
        state = _head_on_state(v_club=45.0)
        model = SpringDamperImpactModel(dt=1e-6)
        post = model.solve(state, params)
        assert post.contact_duration > 0

    @pytest.mark.unit
    def test_result_independent_of_dt_within_tolerance(self) -> None:
        """Ball exit speed at dt=1e-6 and dt=5e-7 must agree within 5%."""
        M = 0.2
        params = ImpactParameters(contact_stiffness=1e5, contact_damping=10.0)
        state = _head_on_state(v_club=45.0, m_club=M)

        post_coarse = SpringDamperImpactModel(dt=1e-6).solve(state, params)
        post_fine = SpringDamperImpactModel(dt=5e-7).solve(state, params)

        coarse_speed = float(np.linalg.norm(post_coarse.ball_velocity))
        fine_speed = float(np.linalg.norm(post_fine.ball_velocity))
        assert abs(coarse_speed - fine_speed) / fine_speed < 0.05


# ---------------------------------------------------------------------------
# Friction rolling cap with pre-existing spin  (#6986)
# ---------------------------------------------------------------------------


class TestFrictionRollingCapWithSpin:
    @pytest.mark.unit
    def test_backspin_reduces_rolling_cap(self) -> None:
        """Pre-existing backspin aligned with the rolling direction reduces
        the tangential friction impulse compared to zero spin."""
        e = 0.78
        M = 0.2

        # Glancing impact: club moving mostly +X with +Y component
        params = ImpactParameters(cor=e, friction_coefficient=0.5)
        state_no_spin = PreImpactState(
            clubhead_velocity=np.array([45.0, 5.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([R_BALL, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=M,
        )

        # Same but ball has backspin aligned with the rolling-without-slip axis
        # v_tangent along +Y, n along +X → spin_axis = n × tangent = +Z
        # Positive Z spin contributes positively to rolling → reduces needed impulse
        state_with_spin = PreImpactState(
            clubhead_velocity=np.array([45.0, 5.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.array([R_BALL, 0.0, 0.0]),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.array([0.0, 0.0, 100.0]),  # pre-existing +Z spin
            clubhead_mass=M,
        )

        model = RigidBodyImpactModel()
        post_no_spin = model.solve(state_no_spin, params)
        post_with_spin = model.solve(state_with_spin, params)

        # With pre-existing spin aligned in rolling direction, less friction needed
        # → less ADDITIONAL spin is generated
        added_z_no_spin = float(post_no_spin.ball_angular_velocity[2])
        added_z_with_spin = float(post_with_spin.ball_angular_velocity[2]) - 100.0
        assert added_z_with_spin < added_z_no_spin


# ---------------------------------------------------------------------------
# create_impact_model factory  (#6993)
# ---------------------------------------------------------------------------


class TestCreateImpactModel:
    @pytest.mark.unit
    def test_rigid_body(self) -> None:
        from src.shared.python.physics._impact_physics import RigidBodyImpactModel

        assert isinstance(
            create_impact_model(ImpactModelType.RIGID_BODY), RigidBodyImpactModel
        )

    @pytest.mark.unit
    def test_spring_damper(self) -> None:
        from src.shared.python.physics._impact_physics import SpringDamperImpactModel

        assert isinstance(
            create_impact_model(ImpactModelType.SPRING_DAMPER), SpringDamperImpactModel
        )

    @pytest.mark.unit
    def test_finite_time(self) -> None:
        from src.shared.python.physics._impact_physics import FiniteTimeImpactModel

        assert isinstance(
            create_impact_model(ImpactModelType.FINITE_TIME), FiniteTimeImpactModel
        )

    @pytest.mark.unit
    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            create_impact_model("not_a_type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Precondition violations  (#6993)
# ---------------------------------------------------------------------------


class TestPreconditionViolations:
    @pytest.mark.unit
    def test_negative_mass_raises(self) -> None:
        """Negative clubhead mass violates precondition."""
        state = _head_on_state(m_club=-0.1)
        params = ImpactParameters()
        model = RigidBodyImpactModel()
        with pytest.raises(PreconditionError):
            model.solve(state, params)

    @pytest.mark.unit
    def test_cor_above_one_raises(self) -> None:
        """COR > 1 violates precondition."""
        state = _head_on_state()
        params = ImpactParameters(cor=1.1)
        model = RigidBodyImpactModel()
        with pytest.raises(PreconditionError):
            model.solve(state, params)

    @pytest.mark.unit
    def test_negative_cor_raises(self) -> None:
        """Negative COR violates precondition."""
        state = _head_on_state()
        params = ImpactParameters(cor=-0.1)
        model = RigidBodyImpactModel()
        with pytest.raises(PreconditionError):
            model.solve(state, params)

    @pytest.mark.unit
    def test_negative_friction_raises(self) -> None:
        """Negative friction coefficient violates precondition."""
        state = _head_on_state()
        params = ImpactParameters(friction_coefficient=-0.1)
        model = RigidBodyImpactModel()
        with pytest.raises(PreconditionError):
            model.solve(state, params)
