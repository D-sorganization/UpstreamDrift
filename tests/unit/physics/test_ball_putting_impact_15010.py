"""Regression tests for issue #15010 — putting slope, friction, flight, impact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)
from src.shared.python.physics import ball_properties as bp
from src.shared.python.physics import flight_models as fm
from src.shared.python.physics.impact_model.models import RigidBodyImpactModel
from src.shared.python.physics.impact_model.types import (
    ImpactParameters,
    PreImpactState,
)
from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions
from src.shared.python.putting_dynamics import (
    BallState,
    FrictionField,
    FrictionParams,
    HeightField,
    SurfaceSpec,
    simulate_ball,
    stimp_to_rolling_mu,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

_G = float(GRAVITY_M_S2)
_R = float(GOLF_BALL_RADIUS_M)
_ROLLING_GRAVITY_FACTOR = 5.0 / 7.0


def _downhill_surface(grade_pct: float, stimp: float = 10.0) -> SurfaceSpec:
    return SurfaceSpec(
        height=HeightField.planar(grade_pct, 0.0, extent_m=30.0),
        friction_field=FrictionField.uniform(extent_m=30.0),
        friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp)),
    )


class TestRollingSlopeGravityFactor:
    """Rolling spheres accelerate downhill at (5/7) g sin(alpha), not g sin(alpha)."""

    def test_three_percent_downhill_rollout_matches_5_over_7_slope_term(self) -> None:
        grade_pct = 3.0
        stimp = 10.0
        v0 = 1.5
        mu_r = stimp_to_rolling_mu(stimp)
        sin_alpha = grade_pct / 100.0
        a_along = _ROLLING_GRAVITY_FACTOR * _G * sin_alpha - mu_r * _G
        expected = v0**2 / (2.0 * abs(a_along))

        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        result = simulate_ball(ball, _downhill_surface(grade_pct, stimp))

        assert result.total_distance_m == pytest.approx(expected, rel=0.02)
        assert result.total_distance_m == pytest.approx(3.3165, rel=0.02)
        assert result.total_distance_m != pytest.approx(4.409, rel=0.02)


class TestPuttingFrictionLawUnity:
    def test_legacy_turf_uses_shared_stimp_law_at_stimp_ten(self) -> None:
        from src.engines.physics_engines.putting_green.python.turf_properties import (
            TurfProperties,
        )

        turf = TurfProperties(stimp_rating=10.0)
        assert turf.rolling_friction_coefficient == pytest.approx(
            stimp_to_rolling_mu(10.0), rel=1e-12
        )


class TestFlightEntryPointAttribution:
    """Cross-entry paths must declare distinct coefficient sets (#8978, #15010)."""

    # Tour-average driver launch cited in the review (m/s, deg, rpm).
    _BALL_SPEED_MPS = 74.66
    _LAUNCH_ANGLE_DEG = 10.9
    _SPIN_RPM = 2686.0

    def test_core_and_waterloo_penner_remain_attributably_different(self) -> None:
        launch = fm.UnifiedLaunchConditions(
            ball_speed=self._BALL_SPEED_MPS,
            launch_angle=math.radians(self._LAUNCH_ANGLE_DEG),
            spin_rate=self._SPIN_RPM,
        )
        wp = fm.FlightModelRegistry.get_model(fm.FlightModelType.WATERLOO_PENNER)
        wp_result = wp.simulate(launch, max_time=12.0, dt=0.01)

        from src.shared.python.physics.ball_flight_physics import (
            EnhancedBallFlightSimulator,
            LaunchConditions,
        )

        core = EnhancedBallFlightSimulator()
        core_traj = core.simulate_trajectory(
            LaunchConditions(
                velocity=self._BALL_SPEED_MPS,
                launch_angle=math.radians(self._LAUNCH_ANGLE_DEG),
                spin_rate=self._SPIN_RPM,
            ),
            max_time=12.0,
            dt=0.01,
        )
        core_carry = core.analyze_trajectory(core_traj)["carry_distance"]

        assert wp_result.coefficients == fm.WATERLOO_PENNER_COEFFICIENTS.as_dict()
        assert bp.BallProperties().cd1 != fm.WATERLOO_PENNER_COEFFICIENTS.cd1
        assert core_carry != pytest.approx(wp_result.carry_distance, rel=1e-3)


class TestSpinAxisAngleSignConvention:
    def test_positive_tilt_documented_as_left_curve_in_ud_frame(self) -> None:
        launch = fm.UnifiedLaunchConditions(
            ball_speed=70.0,
            launch_angle=math.radians(12.0),
            spin_rate=3000.0,
            spin_axis_angle=math.radians(10.0),
        )
        model = fm.WaterlooPennerModel()
        result = model.simulate(launch, max_time=8.0, dt=0.01)
        lateral = float(result.trajectory[-1].position[1])
        assert lateral > 1.0, "positive spin_axis_angle must curve left (+Y) in UD-07"


class TestOffCenterImpactAngularMomentum:
    def test_toe_offset_imparts_clubhead_spin(self) -> None:
        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.83, friction_coefficient=0.2)
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.020, 0.0]),
        )
        post = model.solve(pre, params)
        assert float(np.linalg.norm(post.clubhead_angular_velocity)) > 1e-6

    def test_linear_momentum_conserved_for_toe_hit(self) -> None:
        m_ball = 0.04593
        m_club = 0.200
        model = RigidBodyImpactModel()
        params = ImpactParameters(cor=0.83, friction_coefficient=0.0)
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=m_club,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.020, 0.0]),
        )
        post = model.solve(pre, params)
        p_before = m_club * pre.clubhead_velocity + m_ball * pre.ball_velocity
        p_after = m_club * post.clubhead_velocity + m_ball * post.ball_velocity
        np.testing.assert_allclose(p_after, p_before, rtol=1e-9, atol=1e-9)


class TestRustAirTemperatureUnits:
    def test_environment_temperature_is_celsius_sea_level_reference(self) -> None:
        env = EnvironmentalConditions.from_altitude(altitude_m=0.0, temperature_c=15.0)
        assert env.temperature == pytest.approx(15.0)

    def test_ball_simulator_passes_kelvin_to_rust_air(self) -> None:
        from src.shared.python.physics import ball_simulator as bs

        env = EnvironmentalConditions(temperature=15.0)
        assert bs._air_temperature_kelvin(env) == pytest.approx(288.15, rel=1e-9)


class TestRemainingHonestyCriteria15010:
    """C4, C9, C10, C12, C13 regression tests for #15010."""

    def test_simulators_expose_model_metadata_and_force_provider_parity(self) -> None:
        from src.shared.python.physics.ball_flight_physics import (
            BallFlightSimulator,
            EnhancedBallFlightSimulator,
            LaunchConditions,
        )

        expected_cl_max = {
            BallFlightSimulator: bp.MAX_LIFT_COEFFICIENT,
            EnhancedBallFlightSimulator: 0.25,
        }
        for cls in (BallFlightSimulator, EnhancedBallFlightSimulator):
            meta = cls().model_metadata
            assert meta["drag_model"] == "reynolds_dimpled_sphere"
            assert meta["base_cd"] == pytest.approx(0.25)
            assert meta["cl_max"] == pytest.approx(expected_cl_max[cls])
            assert meta["spin_decay_rate"] == pytest.approx(0.05)

        wp_set = fm.WATERLOO_PENNER_COEFFICIENTS
        r_b = float(GOLF_BALL_RADIUS_M)
        area = math.pi * r_b**2
        omega0 = 2686.0 * 2.0 * math.pi / 60.0

        def wp_force(
            vel: np.ndarray, _spin: np.ndarray, _t: float, _pos: np.ndarray
        ) -> np.ndarray:
            v = float(np.linalg.norm(vel))
            if v < 0.1:
                return np.zeros(3)
            s = r_b * omega0 / v
            cd = wp_set.cd0 + wp_set.cd1 * s + wp_set.cd2 * s**2
            cl = min(wp_set.cl_max, wp_set.lift_scale * s**wp_set.lift_exponent)
            q = 0.5 * 1.225 * area * v**2
            lift_dir = np.cross(np.array([0.0, -1.0, 0.0]), vel / v)
            lift_dir /= float(np.linalg.norm(lift_dir))
            return -(q * cd) * (vel / v) + (q * cl) * lift_dir

        sim = EnhancedBallFlightSimulator(force_provider=wp_force)
        traj = sim.simulate_trajectory(
            LaunchConditions(74.66, math.radians(10.9), spin_rate=2686.0),
            max_time=12.0,
            dt=0.01,
        )
        enhanced_wp_carry = sim.analyze_trajectory(traj)["carry_distance"]
        wp = fm.WaterlooPennerModel().simulate(
            fm.UnifiedLaunchConditions(
                ball_speed=74.66, launch_angle=math.radians(10.9), spin_rate=2686.0
            ),
            max_time=12.0,
            dt=0.01,
        )
        assert enhanced_wp_carry == pytest.approx(wp.carry_distance, abs=1.0)

    def test_constant_coeff_models_use_illustrative_citations(self) -> None:
        fm.FlightModelRegistry.reset()
        stale = {
            "Nathan et al. (2018)",
            "Ballantyne et al. (2012)",
            "Cole (2016)",
            "Rospie & Layton (2014)",
            "Charry et al. (2017)",
        }
        for model in fm.FlightModelRegistry.get_all_models():
            if isinstance(model, fm.ConstantCoefficientModel):
                assert model.reference not in stale
                assert "Illustrative" in model.reference

    def test_air_density_local_ambient_temperature_mode(self) -> None:
        from src.shared.python.physics.atmosphere import (
            DRY_AIR_R_SPECIFIC_J_KG_K,
            ISA_GRAVITY_M_S2,
            ISA_LAPSE_RATE_K_PER_M,
            ISA_P0_PA,
            ISA_T0_K,
            air_density,
        )

        rho_local = air_density(1600.0, 30.0, temperature_is_local=True)
        t_isa = ISA_T0_K - ISA_LAPSE_RATE_K_PER_M * 1600.0
        p_isa = ISA_P0_PA * (t_isa / ISA_T0_K) ** (
            ISA_GRAVITY_M_S2 / (DRY_AIR_R_SPECIFIC_J_KG_K * ISA_LAPSE_RATE_K_PER_M)
        )
        assert rho_local == pytest.approx(
            p_isa / (DRY_AIR_R_SPECIFIC_J_KG_K * (273.15 + 30.0)), rel=1e-9
        )
        assert rho_local < air_density(1600.0, 30.0, temperature_is_local=False)

    def test_launch_conditions_from_imperial_converts_mph(self) -> None:
        from src.shared.python.physics.ball_flight_physics import LaunchConditions

        lc = LaunchConditions.from_imperial(167.0, 10.9, spin_rate_rpm=2686.0)
        assert lc.velocity == pytest.approx(167.0 * 0.44704, rel=1e-9)

    def test_off_center_high_friction_impact_does_not_reverse_tangential_slip(
        self,
    ) -> None:
        from src.shared.python.physics.impact_model.contact_kinematics import (
            impact_offset_vector_3d,
        )

        pre = PreImpactState(
            clubhead_velocity=np.array([45.0, 0.0, 5.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.200,
            clubhead_moi=4.5e-4,
            impact_offset=np.array([0.020, 0.0]),
        )
        post = RigidBodyImpactModel().solve(
            pre, ImpactParameters(cor=0.83, friction_coefficient=5.0)
        )
        n = pre.clubhead_orientation
        r_c = impact_offset_vector_3d(pre)
        v_club_c = post.clubhead_velocity + np.cross(
            post.clubhead_angular_velocity, r_c
        )
        v_ball_c = post.ball_velocity - _R * np.cross(post.ball_angular_velocity, n)
        slip_post = v_club_c - v_ball_c
        slip_post -= np.dot(slip_post, n) * n
        np.testing.assert_allclose(slip_post, np.zeros(3), atol=1e-9)
