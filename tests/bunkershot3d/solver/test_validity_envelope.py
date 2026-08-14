"""Tests for DRFT validity envelope enforcement (issue #8611).

The core requirement: out-of-envelope queries must refuse to return a plausible
number. RFT's stated limit is Fr < 0.4; we are at Fr ~ 25 for a bunker shot, so
honest envelope reporting is the single most important feature.
"""

import math

import numpy as np

from bunkershot3d.solver.envelope import (
    EnvelopeStatus,
    ValidityEnvelope,
    ValidityVerdict,
    compute_froude_number,
    compute_micro_inertial,
)


class TestFroudeNumber:
    """Test Froude number calculation: Fr = v / sqrt(g * L)."""

    def test_froude_at_slow_speed(self) -> None:
        """Fr at 1 m/s, L=0.1 m should be ~1.0."""
        fr = compute_froude_number(velocity_m_s=1.0, length_scale_m=0.1)
        expected = 1.0 / math.sqrt(9.81 * 0.1)
        assert np.isclose(fr, expected, rtol=1e-6)
        assert fr < 2.0

    def test_froude_at_bunker_speed(self) -> None:
        """Fr at 25 m/s, L=0.1 m should be ~25.2 (from research addendum)."""
        fr = compute_froude_number(velocity_m_s=25.0, length_scale_m=0.1)
        assert np.isclose(fr, 25.2, rtol=0.02)

    def test_froude_at_leading_edge_scale(self) -> None:
        """Fr at 25 m/s, L=5 mm should be ~112.9."""
        fr = compute_froude_number(velocity_m_s=25.0, length_scale_m=0.005)
        assert np.isclose(fr, 112.9, rtol=0.02)


class TestMicroInertial:
    """Test micro-inertial number I = v^2 * d^2 / (g * lambda^2)."""

    def test_micro_inertial_at_clubhead_scale(self) -> None:
        """I at v=25 m/s, d=0.5 mm, lambda=100 mm should be ~0.126."""
        i_num = compute_micro_inertial(
            velocity_m_s=25.0, grain_diameter_m=0.0005, intruder_scale_m=0.1
        )
        assert np.isclose(i_num, 0.126, rtol=0.1)


class TestEnvelopeStatus:
    """Test envelope status classification."""

    def test_inside_stated_envelope(self) -> None:
        """Fr < 0.4 is inside the stated RFT envelope."""
        env = ValidityEnvelope(froude=0.3, micro_inertial=0.05, depth_ratio=0.01)
        assert env.status == EnvelopeStatus.INSIDE_STATED

    def test_outside_stated_inside_dynamic(self) -> None:
        """Fr > 0.4 but < hard limit requires dynamic terms."""
        env = ValidityEnvelope(froude=1.0, micro_inertial=0.05, depth_ratio=0.01)
        assert env.status == EnvelopeStatus.REQUIRES_DYNAMIC

    def test_extrapolation_zone(self) -> None:
        """Fr >> stated limit is extrapolation."""
        env = ValidityEnvelope(froude=25.0, micro_inertial=0.1, depth_ratio=0.005)
        assert env.status == EnvelopeStatus.EXTRAPOLATION


class TestValidityVerdict:
    """Test the complete validity verdict."""

    def test_verdict_refuses_without_dynamic_at_high_fr(self) -> None:
        """At Fr > 1, quasi-static RFT must refuse."""
        verdict = ValidityVerdict.evaluate(
            froude=25.0,
            micro_inertial=0.1,
            depth_ratio=0.005,
            dynamic_terms_active=False,
        )
        assert verdict.should_refuse
        assert "dynamic" in verdict.reason.lower()

    def test_verdict_allows_dynamic_at_high_fr(self) -> None:
        """At Fr > 1, DRFT (with dynamic terms) should proceed with warning."""
        verdict = ValidityVerdict.evaluate(
            froude=25.0,
            micro_inertial=0.1,
            depth_ratio=0.005,
            dynamic_terms_active=True,
        )
        assert not verdict.should_refuse
        assert verdict.is_extrapolation

    def test_verdict_clean_at_low_fr(self) -> None:
        """At Fr < 0.4, RFT is in its stated envelope."""
        verdict = ValidityVerdict.evaluate(
            froude=0.3,
            micro_inertial=0.05,
            depth_ratio=0.02,
            dynamic_terms_active=False,
        )
        assert not verdict.should_refuse
        assert not verdict.is_extrapolation

    def test_verdict_reports_all_dimensionless_groups(self) -> None:
        """Verdict must carry Fr, I, d/L for provenance."""
        verdict = ValidityVerdict.evaluate(
            froude=10.0,
            micro_inertial=0.5,
            depth_ratio=0.01,
            dynamic_terms_active=True,
        )
        assert verdict.envelope.froude == 10.0
        assert verdict.envelope.micro_inertial == 0.5
        assert verdict.envelope.depth_ratio == 0.01


class TestSmokeTestForce:
    """The research addendum smoke test: ~1550 N for a 20x80 mm sole at 25 m/s."""

    def test_smoke_test_order_of_magnitude(self) -> None:
        """Force should be order 1000 N, not 10 or 10000."""
        # This is a placeholder - the actual calculation is in the solver
        # Here we just verify the envelope calculation doesn't blow up
        fr = compute_froude_number(velocity_m_s=25.0, length_scale_m=0.08)
        assert 10 < fr < 50  # Reasonable range
