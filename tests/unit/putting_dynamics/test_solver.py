"""Solver analytic-limit and mode-machine tests (#8345 P2+P3).

The flat uniform surface with the basic laws must reproduce the
closed-form skid/roll results restated from Tools
``swing_sim.putting.roll`` (5/7 transition, stimp deceleration)
within RK4 tolerance.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)
from src.shared.python.putting_dynamics import (
    BallState,
    FrictionField,
    FrictionParams,
    HeightField,
    Mode,
    PutterState,
    SurfaceSpec,
    bumpy_friction_field,
    bumpy_height_field,
    capture_speed_mps,
    simulate_ball,
    simulate_strike,
    stimp_to_rolling_mu,
)

pytestmark = pytest.mark.unit

_G = float(GRAVITY_M_S2)
_R = float(GOLF_BALL_RADIUS_M)


def _sloped_surface(grade: float, aspect: float, stimp: float = 10.0) -> SurfaceSpec:
    return SurfaceSpec(
        height=HeightField.planar(grade, aspect, extent_m=30.0),
        friction_field=FrictionField.uniform(extent_m=30.0),
        friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(stimp)),
    )


class TestFlatUniformAnalyticLimits:
    def test_skid_roll_matches_closed_form(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        v0 = 2.0
        result = simulate_ball(BallState(x_m=0.0, y_m=0.0, vx_mps=v0), surface)
        mu_k = surface.friction.mu_slide
        mu_r = surface.friction.mu_roll0
        # Closed forms restated from Tools swing_sim.putting.roll.
        t_skid = v0 / (3.5 * mu_k * _G)
        skid_dist = v0 * t_skid - 0.5 * mu_k * _G * t_skid**2
        v_roll = 5.0 * v0 / 7.0
        roll_out = v_roll**2 / (2.0 * mu_r * _G)
        assert result.skid_distance_m == pytest.approx(skid_dist, rel=0.02)
        assert result.total_distance_m == pytest.approx(skid_dist + roll_out, rel=0.01)
        assert not result.holed
        assert result.final_mode is Mode.REST

    def test_five_sevenths_transition_speed(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        v0 = 2.0
        result = simulate_ball(BallState(x_m=0.0, y_m=0.0, vx_mps=v0), surface)
        first_roll = next(s for s in result.samples if s.mode is Mode.ROLL)
        assert first_roll.speed_mps == pytest.approx(5.0 * v0 / 7.0, rel=0.01)

    def test_pure_roll_start_skips_the_skid_phase(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        v0 = 1.5
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        result = simulate_ball(ball, surface)
        assert result.skid_distance_m == 0.0
        mu_r = surface.friction.mu_roll0
        assert result.total_distance_m == pytest.approx(
            v0**2 / (2.0 * mu_r * _G), rel=0.01
        )

    def test_overspin_settles_to_the_general_rolling_limit(self) -> None:
        """Stimpmeter-style overspin must not be clamped away at t=0."""
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        v0 = 1.5
        surface_speed0 = 1.8
        ball = BallState(
            x_m=0.0,
            y_m=0.0,
            vx_mps=v0,
            spin_rad_s=surface_speed0 / _R,
        )
        result = simulate_ball(ball, surface)
        first_roll = next(s for s in result.samples[1:] if s.mode is Mode.ROLL)
        expected = (5.0 * v0 + 2.0 * surface_speed0) / 7.0
        assert result.samples[0].mode is Mode.SLIDE
        assert first_roll.speed_mps == pytest.approx(expected, rel=0.01)

    def test_velocity_dependent_rolling_shortens_the_putt(self) -> None:
        base = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        fast_decel = SurfaceSpec.flat_uniform(
            friction=FrictionParams(mu_roll0=base.friction.mu_roll0, k_v_per_mps=0.5)
        )
        v0 = 1.5
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        assert (
            simulate_ball(ball, fast_decel).total_distance_m
            < simulate_ball(ball, base).total_distance_m
        )


class TestSlopesAndSymmetry:
    def test_cross_slope_break_is_mirror_symmetric(self) -> None:
        v0 = 1.5
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        left = simulate_ball(ball, _sloped_surface(2.0, 90.0))
        right = simulate_ball(ball, _sloped_surface(2.0, -90.0))
        assert left.rest_y_m > 0.01  # breaks toward the downhill side
        assert left.rest_y_m == pytest.approx(-right.rest_y_m, rel=1e-9)
        assert left.rest_x_m == pytest.approx(right.rest_x_m, rel=1e-9)

    def test_downhill_putts_run_farther_than_uphill(self) -> None:
        v0 = 1.5
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        down = simulate_ball(ball, _sloped_surface(2.0, 0.0))
        up = simulate_ball(ball, _sloped_surface(2.0, 180.0))
        assert down.total_distance_m > up.total_distance_m

    def test_static_hold_on_a_shallow_slope(self) -> None:
        result = simulate_ball(BallState(x_m=0.0, y_m=0.0), _sloped_surface(2.0, 0.0))
        assert result.final_mode is Mode.REST
        assert math.hypot(result.rest_x_m, result.rest_y_m) < 1e-9

    def test_static_restart_on_a_steep_slope(self) -> None:
        # 10 % grade exceeds the 0.08 static rolling hold: the resting
        # ball must re-start rolling downhill (+x is downhill here).
        result = simulate_ball(BallState(x_m=0.0, y_m=0.0), _sloped_surface(10.0, 0.0))
        assert result.rest_x_m > 0.5
        assert any(s.mode is Mode.ROLL for s in result.samples)


class TestBumpySurfaces:
    def test_zero_amplitude_bumpy_equals_flat_trajectory(self) -> None:
        flat = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        bumpy = SurfaceSpec(
            height=bumpy_height_field(7, 0.0, 1.0, flat.height),
            friction_field=bumpy_friction_field(7, 0.0, 1.0, flat.friction_field),
            friction=flat.friction,
        )
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=1.5)
        a = simulate_ball(ball, flat)
        b = simulate_ball(ball, bumpy)
        assert a.rest_x_m == b.rest_x_m
        assert a.rest_y_m == b.rest_y_m
        assert a.time_s == b.time_s

    def test_seeded_bumpy_trajectory_is_reproducible(self) -> None:
        def build() -> SurfaceSpec:
            flat = SurfaceSpec.flat_uniform(stimp_ft=10.0)
            return SurfaceSpec(
                height=bumpy_height_field(21, 0.003, 0.8, flat.height),
                friction_field=bumpy_friction_field(21, 0.2, 0.8, flat.friction_field),
                friction=flat.friction,
            )

        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=1.5)
        a = simulate_ball(ball, build())
        b = simulate_ball(ball, build())
        assert a.rest_x_m == b.rest_x_m
        assert a.rest_y_m == b.rest_y_m
        # Bumps must actually perturb the putt vs the flat green.
        flat_result = simulate_ball(ball, SurfaceSpec.flat_uniform(stimp_ft=10.0))
        assert a.rest_x_m != flat_result.rest_x_m


class TestHoleCapture:
    def test_dying_putt_is_captured(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        mu_r = surface.friction.mu_roll0
        v0 = math.sqrt(2.0 * mu_r * _G * 2.1)  # rolls out ~2.1 m
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        result = simulate_ball(ball, surface, hole_x_m=2.0, hole_y_m=0.0)
        assert result.holed
        assert result.speed_at_hole_mps is not None
        assert result.speed_at_hole_mps <= capture_speed_mps()

    def test_charging_putt_lips_through(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        mu_r = surface.friction.mu_roll0
        v0 = math.sqrt(2.0 * mu_r * _G * 8.0)  # would roll out ~8 m
        ball = BallState(x_m=0.0, y_m=0.0, vx_mps=v0, spin_rad_s=v0 / _R)
        result = simulate_ball(ball, surface, hole_x_m=2.0, hole_y_m=0.0)
        assert not result.holed
        assert result.speed_at_hole_mps is not None
        assert result.speed_at_hole_mps > capture_speed_mps()
        assert result.rest_x_m > 2.0 + 0.054

    def test_capture_bound_value(self) -> None:
        # A dead-centre path has a full-hole-diameter travel budget:
        # 2 R_h sqrt(g / (2 r_ball)) ~= 1.64 m/s (Holmes full-chord
        # geometry, matching the AffineDrift article and P4 review).
        assert capture_speed_mps() == pytest.approx(
            2.0 * 0.054 * math.sqrt(_G / (2.0 * _R)), rel=1e-12
        )
        assert 1.5 < capture_speed_mps() < 1.75


class TestStrikeToRoll:
    def test_lofted_strike_rises_then_rolls_out(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        putter = PutterState(
            head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=8.0, speed_mps=3.0
        )
        result = simulate_strike(putter, surface)
        modes = [s.mode for s in result.samples]
        assert Mode.AIRBORNE in modes
        assert Mode.SLIDE in modes
        assert Mode.ROLL in modes
        assert result.final_mode is Mode.REST
        assert max(s.height_m for s in result.samples) > 0.0
        assert result.collision is not None
        assert result.collision.putter_dv_mps > 0.0

    def test_airborne_precedes_ground_modes(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        putter = PutterState(
            head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=8.0, speed_mps=3.0
        )
        result = simulate_strike(putter, surface)
        modes = [s.mode for s in result.samples]
        last_air = max(i for i, m in enumerate(modes) if m is Mode.AIRBORNE)
        first_ground = min(
            i for i, m in enumerate(modes) if m in (Mode.SLIDE, Mode.ROLL)
        )
        assert last_air < first_ground

    def test_delofted_strike_stays_on_the_turf(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        putter = PutterState(
            head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=-2.0, speed_mps=2.0
        )
        result = simulate_strike(putter, surface)
        assert all(s.height_m == 0.0 for s in result.samples)
