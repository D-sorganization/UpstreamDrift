"""Value-asserting tests for impact physics (issues #6993/#6982/#6984/#6986).

These replace the smoke-only coverage of ``_impact_physics`` /
``impact_model`` with hand-computed numerical assertions, momentum
conservation, and relative-frame energy-balance checks.

All tests target the public ``impact_model`` package API, which is what
consumers import.
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
M_CLUB = 0.200
I_BALL = float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)
R_BALL = float(GOLF_BALL_RADIUS_M)


def _center_pre_state(
    club_speed: float = 45.0,
    ball_velocity: np.ndarray | None = None,
    ball_spin: np.ndarray | None = None,
) -> PreImpactState:
    """Head-on center impact along +x with the clubface normal +x."""
    n = np.array([1.0, 0.0, 0.0])
    return PreImpactState(
        clubhead_velocity=np.array([club_speed, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=n,
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3) if ball_velocity is None else ball_velocity,
        ball_angular_velocity=np.zeros(3) if ball_spin is None else ball_spin,
        clubhead_mass=M_CLUB,
    )


def _reduced_mass(m_a: float, m_b: float) -> float:
    return (m_a * m_b) / (m_a + m_b)


# --------------------------------------------------------------------------- #
# #6993 — RigidBodyImpactModel.solve value assertions
# --------------------------------------------------------------------------- #
class TestRigidBodyClosedForm:
    @pytest.mark.unit
    def test_center_impact_post_speeds_match_hand_computed(self) -> None:
        cor = 0.78
        pre = _center_pre_state(club_speed=45.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=cor))

        m_eff = _reduced_mass(M_BALL, M_CLUB)
        j = (1.0 + cor) * m_eff * 45.0
        expected_ball = j / M_BALL
        expected_club = 45.0 - j / M_CLUB

        assert post.ball_velocity[0] == pytest.approx(expected_ball, rel=1e-9)
        assert post.clubhead_velocity[0] == pytest.approx(expected_club, rel=1e-9)
        # 45 m/s driver @ COR 0.78 -> ~65 m/s ball, ~146 mph
        assert post.ball_velocity[0] == pytest.approx(65.1405, abs=1e-3)

    @pytest.mark.unit
    def test_perfectly_elastic_cor_one(self) -> None:
        pre = _center_pre_state(club_speed=40.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=1.0))
        m_eff = _reduced_mass(M_BALL, M_CLUB)
        j = 2.0 * m_eff * 40.0
        assert post.ball_velocity[0] == pytest.approx(j / M_BALL, rel=1e-9)

    @pytest.mark.unit
    def test_perfectly_inelastic_cor_zero_common_velocity(self) -> None:
        pre = _center_pre_state(club_speed=50.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.0))
        # COR 0 -> ball and club share the common (momentum) velocity along n
        assert post.ball_velocity[0] == pytest.approx(
            post.clubhead_velocity[0], rel=1e-9
        )
        v_common = M_CLUB * 50.0 / (M_BALL + M_CLUB)
        assert post.ball_velocity[0] == pytest.approx(v_common, rel=1e-9)

    @pytest.mark.unit
    def test_rigid_body_momentum_conserved_along_normal(self) -> None:
        pre = _center_pre_state(club_speed=45.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.78))
        p_pre = M_CLUB * 45.0
        p_post = M_BALL * post.ball_velocity[0] + M_CLUB * post.clubhead_velocity[0]
        assert p_post == pytest.approx(p_pre, rel=1e-9)

    @pytest.mark.unit
    def test_solve_launch_angle_for_lofted_normal(self) -> None:
        loft = math.radians(15.0)
        n = np.array([math.cos(loft), 0.0, math.sin(loft)])
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=n,
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=M_CLUB,
        )
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.78))
        # ball impulse is entirely along the face normal -> launch == loft
        launch = math.atan2(post.ball_velocity[2], post.ball_velocity[0])
        assert launch == pytest.approx(loft, abs=1e-9)


class TestRigidBodyHelpers:
    @pytest.mark.unit
    def test_effective_club_mass_center_is_full_mass(self) -> None:
        model = RigidBodyImpactModel()
        pre = _center_pre_state()
        assert model._compute_effective_club_mass(pre) == pytest.approx(M_CLUB)

    @pytest.mark.unit
    def test_effective_club_mass_offset_reduces_mass(self) -> None:
        model = RigidBodyImpactModel()
        pre = _center_pre_state()
        pre.impact_offset = np.array([0.02, 0.0])
        moi = pre.clubhead_moi
        expected = 1.0 / (1.0 / M_CLUB + 0.02**2 / moi)
        assert model._compute_effective_club_mass(pre) == pytest.approx(expected)
        assert model._compute_effective_club_mass(pre) < M_CLUB

    @pytest.mark.unit
    def test_compute_impulse_closed_form(self) -> None:
        model = RigidBodyImpactModel()
        n = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([45.0, 0.0, 0.0])
        j, v_approach = model._compute_impulse(v_rel, n, M_CLUB, 0.78)
        m_eff = _reduced_mass(M_BALL, M_CLUB)
        assert v_approach == pytest.approx(45.0)
        assert j == pytest.approx((1.0 + 0.78) * m_eff * 45.0, rel=1e-9)

    @pytest.mark.unit
    def test_compute_energy_transfer_matches_ke_delta(self) -> None:
        model = RigidBodyImpactModel()
        pre_v = np.zeros(3)
        post_v = np.array([65.0, 0.0, 0.0])
        et = model._compute_energy_transfer(pre_v, post_v)
        assert et == pytest.approx(0.5 * M_BALL * 65.0**2, rel=1e-9)

    @pytest.mark.unit
    def test_friction_spin_zero_for_pure_normal_impact(self) -> None:
        model = RigidBodyImpactModel()
        pre = _center_pre_state()
        n = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([45.0, 0.0, 0.0])
        spin = model._compute_friction_spin(pre, v_rel, 45.0, n, 3.0, 0.4)
        assert np.allclose(spin, np.zeros(3))

    @pytest.mark.unit
    def test_friction_spin_oblique_impact_nonzero(self) -> None:
        model = RigidBodyImpactModel()
        pre = _center_pre_state()
        n = np.array([1.0, 0.0, 0.0])
        # tangential component along +z
        v_rel = np.array([45.0, 0.0, 5.0])
        v_approach = float(np.dot(v_rel, n))
        spin = model._compute_friction_spin(pre, v_rel, v_approach, n, 3.0, 0.4)
        assert np.linalg.norm(spin) > 0.0


# --------------------------------------------------------------------------- #
# #6993 — gear effect, dispatch, validation
# --------------------------------------------------------------------------- #
class TestGearEffectSpin:
    @pytest.mark.unit
    def test_center_offset_gives_no_gear_spin(self) -> None:
        spin = compute_gear_effect_spin(
            np.array([0.0, 0.0]),
            np.array([45.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        assert np.allclose(spin, np.zeros(3))

    @pytest.mark.unit
    def test_toe_and_heel_give_opposite_sidespin(self) -> None:
        v = np.array([45.0, 0.0, 0.0])
        n = np.array([1.0, 0.0, 0.0])
        toe = compute_gear_effect_spin(np.array([0.02, 0.0]), v, n)
        heel = compute_gear_effect_spin(np.array([-0.02, 0.0]), v, n)
        # vertical-axis (z) component flips sign between toe and heel
        assert np.sign(toe[2]) == -np.sign(heel[2])
        assert toe[2] != pytest.approx(0.0)

    @pytest.mark.unit
    def test_gear_factor_out_of_range_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            compute_gear_effect_spin(
                np.array([0.01, 0.0]),
                np.array([45.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                gear_factor=1.5,
            )


class TestFactoryDispatch:
    @pytest.mark.unit
    def test_dispatch_rigid_body(self) -> None:
        assert isinstance(
            create_impact_model(ImpactModelType.RIGID_BODY), RigidBodyImpactModel
        )

    @pytest.mark.unit
    def test_dispatch_spring_damper(self) -> None:
        assert isinstance(
            create_impact_model(ImpactModelType.SPRING_DAMPER),
            SpringDamperImpactModel,
        )

    @pytest.mark.unit
    def test_dispatch_finite_time(self) -> None:
        assert isinstance(
            create_impact_model(ImpactModelType.FINITE_TIME), FiniteTimeImpactModel
        )


class TestPreconditionViolations:
    @pytest.mark.unit
    def test_cor_above_one_raises(self) -> None:
        pre = _center_pre_state()
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(pre, ImpactParameters(cor=1.5))

    @pytest.mark.unit
    def test_negative_clubhead_mass_raises(self) -> None:
        pre = _center_pre_state()
        pre.clubhead_mass = -0.2
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.78))

    @pytest.mark.unit
    def test_negative_friction_raises(self) -> None:
        pre = _center_pre_state()
        with pytest.raises((ValueError, Exception)):
            RigidBodyImpactModel().solve(
                pre, ImpactParameters(cor=0.78, friction_coefficient=-0.1)
            )

    @pytest.mark.unit
    def test_unknown_model_type_raises(self) -> None:
        with pytest.raises(ValueError):
            create_impact_model("not-a-type")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# #6982 — SpringDamperImpactModel momentum conservation + dt-insensitivity
# --------------------------------------------------------------------------- #
class TestSpringDamperMomentum:
    @pytest.mark.unit
    def test_momentum_conserved_along_normal(self) -> None:
        """m_b*dv_b must equal -m_c*dv_c (Newton's third law)."""
        pre = _center_pre_state(club_speed=45.0)
        params = ImpactParameters(contact_stiffness=1e7, contact_damping=1e3)
        post = SpringDamperImpactModel(dt=1e-7).solve(pre, params)

        dp_ball = M_BALL * (post.ball_velocity[0] - 0.0)
        dp_club = M_CLUB * (post.clubhead_velocity[0] - 45.0)
        assert dp_ball == pytest.approx(-dp_club, abs=1e-6)

    @pytest.mark.unit
    def test_total_linear_momentum_conserved(self) -> None:
        pre = _center_pre_state(club_speed=45.0)
        params = ImpactParameters(contact_stiffness=1e7, contact_damping=1e3)
        post = SpringDamperImpactModel(dt=1e-7).solve(pre, params)
        p_pre = M_CLUB * 45.0
        p_post = M_BALL * post.ball_velocity[0] + M_CLUB * post.clubhead_velocity[0]
        assert p_post == pytest.approx(p_pre, rel=1e-6)

    @pytest.mark.unit
    def test_ball_speed_dt_insensitive(self) -> None:
        """Halving dt must not change ball speed beyond integration error."""
        pre_a = _center_pre_state(club_speed=45.0)
        pre_b = _center_pre_state(club_speed=45.0)
        params = ImpactParameters(contact_stiffness=1e7, contact_damping=1e3)
        post_a = SpringDamperImpactModel(dt=2e-7).solve(pre_a, params)
        post_b = SpringDamperImpactModel(dt=1e-7).solve(pre_b, params)
        assert post_a.ball_velocity[0] == pytest.approx(
            post_b.ball_velocity[0], rel=2e-3
        )

    @pytest.mark.unit
    def test_initial_gap_insensitive(self) -> None:
        """A small change in starting separation must not change the result.

        Regression for #6982: contact onset must be event-detected rather
        than relying on a gap==0 dt-overshoot.
        """
        params = ImpactParameters(contact_stiffness=1e7, contact_damping=1e3)
        pre_touch = _center_pre_state(club_speed=45.0)
        pre_gap = _center_pre_state(club_speed=45.0)
        pre_gap.ball_position = np.array([5e-4, 0.0, 0.0])
        post_touch = SpringDamperImpactModel(dt=1e-7).solve(pre_touch, params)
        post_gap = SpringDamperImpactModel(dt=1e-7).solve(pre_gap, params)
        assert post_touch.ball_velocity[0] == pytest.approx(
            post_gap.ball_velocity[0], rel=2e-3
        )

    @pytest.mark.unit
    def test_ball_leaves_faster_than_it_arrived(self) -> None:
        pre = _center_pre_state(club_speed=45.0)
        params = ImpactParameters(contact_stiffness=1e7, contact_damping=1e3)
        post = SpringDamperImpactModel(dt=1e-7).solve(pre, params)
        assert post.ball_velocity[0] > 0.0
        assert post.contact_duration > 0.0


# --------------------------------------------------------------------------- #
# #6984 — energy balance in the relative (COM) frame
# --------------------------------------------------------------------------- #
class TestEnergyBalance:
    @pytest.mark.unit
    def test_expected_loss_is_relative_frame_energy(self) -> None:
        """expected_relative_ke_loss == 1/2 mu v_rel^2 (1-e^2)."""
        cor = 0.78
        pre = _center_pre_state(club_speed=45.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=cor))
        report = validate_energy_balance(pre, post, ImpactParameters(cor=cor))

        mu = _reduced_mass(M_BALL, M_CLUB)
        v_rel = 45.0
        expected = 0.5 * mu * v_rel**2 * (1.0 - cor**2)
        assert "expected_relative_ke_loss" in report
        assert report["expected_relative_ke_loss"] == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_actual_relative_loss_matches_expected_for_rigid_body(self) -> None:
        """The rigid-body solver loses exactly 1/2 mu v_rel^2 (1-e^2)."""
        cor = 0.6
        pre = _center_pre_state(club_speed=40.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=cor))
        report = validate_energy_balance(pre, post, ImpactParameters(cor=cor))
        assert report["relative_ke_loss"] == pytest.approx(
            report["expected_relative_ke_loss"], rel=1e-6
        )

    @pytest.mark.unit
    def test_lab_frame_keys_still_present(self) -> None:
        cor = 0.78
        pre = _center_pre_state(club_speed=45.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=cor))
        report = validate_energy_balance(pre, post, ImpactParameters(cor=cor))
        for key in ("total_ke_pre", "total_ke_post", "energy_lost", "ball_ke_post"):
            assert key in report

    @pytest.mark.unit
    def test_elastic_impact_has_zero_relative_loss(self) -> None:
        pre = _center_pre_state(club_speed=45.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=1.0))
        report = validate_energy_balance(pre, post, ImpactParameters(cor=1.0))
        assert report["expected_relative_ke_loss"] == pytest.approx(0.0, abs=1e-9)
        assert report["relative_ke_loss"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# #6986 — friction rolling cap relative to (v_t - omega*R)
# --------------------------------------------------------------------------- #
class TestFrictionRollingCap:
    @pytest.mark.unit
    def test_prespinning_ball_does_not_exceed_rolling_limit(self) -> None:
        """A ball already spinning near the rolling condition should get
        little/no extra friction spin; the cap is on the *slip* velocity
        (v_t - omega*R), not the raw tangential velocity.
        """
        model = RigidBodyImpactModel()
        n = np.array([1.0, 0.0, 0.0])
        # tangential relative velocity along +z
        v_t = 5.0
        v_rel = np.array([45.0, 0.0, v_t])
        v_approach = float(np.dot(v_rel, n))
        j = 3.0

        # spin axis for tangent +z about normal +x is n x t_hat = +y
        # choose omega so that omega*R ~ v_t (already rolling) -> omega_y
        omega_y = v_t / R_BALL
        pre_rolling = _center_pre_state(ball_spin=np.array([0.0, omega_y, 0.0]))
        pre_static = _center_pre_state(ball_spin=np.zeros(3))

        spin_rolling = model._compute_friction_spin(
            pre_rolling, v_rel, v_approach, n, j, 0.4
        )
        spin_static = model._compute_friction_spin(
            pre_static, v_rel, v_approach, n, j, 0.4
        )
        # the added spin (delta from pre-existing) must be smaller when the
        # ball is already rolling than when it is static
        delta_rolling = np.linalg.norm(spin_rolling - pre_rolling.ball_angular_velocity)
        delta_static = np.linalg.norm(spin_static - pre_static.ball_angular_velocity)
        # already-rolling ball has ~zero slip -> strictly less added spin
        assert delta_rolling < delta_static
        assert delta_rolling == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_static_ball_friction_spin_unchanged(self) -> None:
        """A non-spinning ball's friction spin must match the rolling cap on
        the full tangential velocity (v_slip == v_t when omega == 0)."""
        model = RigidBodyImpactModel()
        n = np.array([1.0, 0.0, 0.0])
        v_rel = np.array([45.0, 0.0, 5.0])
        v_approach = float(np.dot(v_rel, n))
        spin = model._compute_friction_spin(
            _center_pre_state(ball_spin=np.zeros(3)), v_rel, v_approach, n, 3.0, 0.4
        )
        assert np.linalg.norm(spin) > 0.0
