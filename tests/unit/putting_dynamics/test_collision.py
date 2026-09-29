"""Collision-model tests for putting_dynamics (#8345 P3).

Momentum and COR energy pins, putter-slowdown band, dynamic-loft
sweep monotonicity, and the hosel-position face-twist model.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from src.shared.python.putting_dynamics import (
    PutterState,
    effective_head_mass,
    energy_balance_error_j,
    strike,
    sweep_dynamic_loft,
)

pytestmark = pytest.mark.unit

_M_BALL = float(GOLF_BALL_MASS_KG)


def _putter(**overrides: float) -> PutterState:
    defaults: dict[str, float] = {
        "head_mass_kg": 0.35,
        "moi_kg_m2": 4.5e-4,
        "loft_deg": 3.0,
        "speed_mps": 2.0,
    }
    defaults.update(overrides)
    return PutterState(**defaults)


class TestTwoBodyImpulse:
    def test_momentum_conservation_along_the_putt_line(self) -> None:
        putter = _putter()
        report = strike(putter)
        delta = math.radians(putter.loft_deg)
        ball_px = _M_BALL * report.horizontal_speed_mps
        # Tangential impulse also carries x-momentum; reconstruct the
        # full x impulse from the report and pin head slowdown to it.
        impulse_t = (2.0 / 7.0) * _M_BALL * putter.speed_mps * math.sin(delta)
        impulse_x = report.impulse_n_s * math.cos(delta) + impulse_t * math.sin(delta)
        assert ball_px == pytest.approx(impulse_x, rel=1e-12)
        assert putter.head_mass_kg * report.putter_dv_mps == pytest.approx(
            impulse_x, rel=1e-12
        )

    def test_energy_loss_is_pinned_to_the_cor_formula(self) -> None:
        putter = _putter()
        report = strike(putter)
        delta = math.radians(putter.loft_deg)
        m_eff = effective_head_mass(putter, 0.0)
        m_red = m_eff * _M_BALL / (m_eff + _M_BALL)
        v_n = putter.speed_mps * math.cos(delta)
        u = putter.speed_mps * math.sin(delta)
        expected = 0.5 * m_red * v_n**2 * (1.0 - putter.cor**2) + _M_BALL * u**2 / 7.0
        assert report.kinetic_energy_loss_j == pytest.approx(expected, rel=1e-12)

    def test_energy_audit_closes_for_a_centered_strike(self) -> None:
        putter = _putter()
        report = strike(putter)
        ke_in = 0.5 * putter.head_mass_kg * putter.speed_mps**2
        assert energy_balance_error_j(report, putter) < 0.01 * ke_in

    def test_putter_slowdown_band_for_light_effective_head(self) -> None:
        # ~2 m/s putter, 0.15 kg effective mass vs the 45.93 g ball:
        # dv = J/M with J = (1 + e) m_red v -> ~0.83 m/s.
        report = strike(_putter(head_mass_kg=0.15))
        assert 0.6 <= report.putter_dv_mps <= 1.0

    def test_centered_effective_mass_equals_head_mass(self) -> None:
        putter = _putter()
        assert effective_head_mass(putter, 0.0) == pytest.approx(
            putter.head_mass_kg, rel=1e-12
        )

    def test_offset_strike_softens_the_blow(self) -> None:
        putter = _putter()
        assert effective_head_mass(putter, 0.03) < putter.head_mass_kg
        centered = strike(putter)
        toe = strike(putter, impact_toe_m=0.03)
        assert toe.ball_speed_mps < centered.ball_speed_mps

    def test_backspin_at_positive_loft(self) -> None:
        report = strike(_putter())
        assert report.spin_rad_s < 0.0
        u = 2.0 * math.sin(math.radians(3.0))
        expected = -(5.0 / 7.0) * u / float(GOLF_BALL_RADIUS_M)
        assert report.spin_rad_s == pytest.approx(expected, rel=1e-12)

    def test_tangential_impulse_is_consistent_with_backspin(self) -> None:
        """The face drags the rear contact point down and forward."""
        putter = _putter()
        report = strike(putter)
        delta = math.radians(putter.loft_deg)
        u = putter.speed_mps * math.sin(delta)
        normal_speed = report.impulse_n_s / _M_BALL
        tangential_speed = (2.0 / 7.0) * u
        assert report.horizontal_speed_mps == pytest.approx(
            normal_speed * math.cos(delta) + tangential_speed * math.sin(delta),
            rel=1e-12,
        )
        assert report.vertical_speed_mps == pytest.approx(
            normal_speed * math.sin(delta) - tangential_speed * math.cos(delta),
            rel=1e-12,
        )
        assert report.launch_angle_deg < putter.loft_deg


class TestLoftSweep:
    def test_sweep_covers_minus4_to_plus8(self) -> None:
        reports = sweep_dynamic_loft(_putter())
        assert reports[0].effective_loft_deg == pytest.approx(-4.0)
        assert reports[-1].effective_loft_deg == pytest.approx(8.0)

    def test_more_loft_launches_higher(self) -> None:
        reports = sweep_dynamic_loft(_putter())
        angles = [r.launch_angle_deg for r in reports]
        assert all(a < b for a, b in zip(angles, angles[1:], strict=False))

    def test_more_loft_means_more_initial_slide(self) -> None:
        reports = sweep_dynamic_loft(_putter())
        r = float(GOLF_BALL_RADIUS_M)
        slip = [x.horizontal_speed_mps - x.spin_rad_s * r for x in reports]
        assert all(a < b for a, b in zip(slip, slip[1:], strict=False))

    def test_negative_loft_drives_the_ball_downward(self) -> None:
        report = strike(_putter(loft_deg=-4.0))
        assert report.vertical_speed_mps < 0.0
        assert report.spin_rad_s > 0.0  # topspin off a delofted face


class TestHoselTwist:
    def test_centered_attachment_centered_strike_no_twist(self) -> None:
        report = strike(_putter())
        assert report.face_twist_rad_s == 0.0
        assert report.twist_moment_n_m_s == 0.0

    def test_heel_attachment_center_strike_twists_positively(self) -> None:
        # Anser-style heel hosel: a center strike lies toe-side of the
        # shaft axis, so the toe swings backward (positive twist).
        report = strike(_putter(hosel_toe_m=-0.05))
        assert report.face_twist_rad_s > 0.0

    def test_twist_is_antisymmetric_under_toe_mirror(self) -> None:
        putter = _putter(hosel_toe_m=0.03)
        mirrored = replace(putter, hosel_toe_m=-0.03)
        plus = strike(putter, impact_toe_m=0.01)
        minus = strike(mirrored, impact_toe_m=-0.01)
        assert plus.face_twist_rad_s == pytest.approx(
            -minus.face_twist_rad_s, rel=1e-12
        )

    def test_strike_through_the_shaft_axis_has_no_twist(self) -> None:
        report = strike(_putter(hosel_toe_m=0.02), impact_toe_m=0.02)
        assert report.face_twist_rad_s == pytest.approx(0.0, abs=1e-15)

    def test_wrench_moment_z_matches_twist_moment(self) -> None:
        report = strike(_putter(hosel_toe_m=-0.04), impact_toe_m=0.01)
        assert report.attachment_moment_n_m_s[2] == pytest.approx(
            report.twist_moment_n_m_s, rel=1e-12
        )

    def test_forward_hosel_offset_alone_adds_no_twist(self) -> None:
        report = strike(_putter(hosel_forward_m=0.02))
        assert report.face_twist_rad_s == 0.0
        # ... but it does appear in the wrench moment (pitch axis).
        assert report.attachment_moment_n_m_s[1] != 0.0

    def test_forward_hosel_offset_increases_shaft_axis_inertia(self) -> None:
        near = strike(_putter(hosel_toe_m=-0.04, hosel_forward_m=0.0))
        far = strike(_putter(hosel_toe_m=-0.04, hosel_forward_m=0.04))
        assert abs(far.face_twist_rad_s) < abs(near.face_twist_rad_s)
