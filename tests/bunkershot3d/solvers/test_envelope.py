"""The validity envelope (issue #8611, research addendum section 1).

ADR-0032 makes the envelope the defining feature of the F0 tier, so
these tests check the two things that matter: that the dimensionless
groups reproduce the published table exactly, and that a query outside
the envelope produces a refusal or a loudly flagged verdict rather than
a bare plausible number.
"""

from __future__ import annotations

import pytest

from bunkershot3d.solvers import (
    MAX_VALIDATED_SPEED_M_S,
    MIN_CONTINUUM_SIZE_RATIO,
    RFT_FROUDE_LIMIT,
    RFT_INERTIAL_NUMBER_LIMIT,
    STANDING_CAVEATS,
    Caveat,
    EnvelopeStatus,
    OutOfEnvelopeError,
    RefusalPolicy,
    SolverInputError,
    ValidityVerdict,
    dimensionless_groups,
    evaluate_envelope,
    worst_of,
)

pytestmark = pytest.mark.unit

_CLUBHEAD_SCALES_M = {
    "clubhead": 0.100,
    "sole width": 0.030,
    "leading edge": 0.005,
}


class TestDimensionlessGroups:
    """The addendum's own table, recomputed."""

    @pytest.mark.parametrize(
        ("length_m", "froude", "inertial", "grain_ratio"),
        [
            (0.100, 25.2, 0.126, 0.005),
            (0.030, 46.1, 0.768, 0.017),
            (0.005, 112.9, 11.3, 0.100),
        ],
    )
    def test_reproduces_the_published_envelope_table(
        self, length_m: float, froude: float, inertial: float, grain_ratio: float
    ) -> None:
        groups = dimensionless_groups(
            speed_m_s=25.0,
            feature_length_m=length_m,
            grain_diameter_m=0.5e-3,
            element_size_m=2.0e-3,
        )
        assert groups.froude == pytest.approx(froude, rel=2e-3)
        assert groups.micro_inertial_number == pytest.approx(inertial, rel=2e-3)
        assert groups.grain_size_ratio == pytest.approx(grain_ratio, rel=2e-2)

    def test_micro_inertial_number_is_froude_times_the_grain_ratio(self) -> None:
        groups = dimensionless_groups(
            speed_m_s=17.0,
            feature_length_m=0.042,
            grain_diameter_m=0.33e-3,
            element_size_m=1.5e-3,
        )
        assert groups.micro_inertial_number == pytest.approx(
            groups.froude * groups.grain_size_ratio, rel=1e-14
        )

    def test_askari_kamrin_element_number_reproduces_the_quoted_four(self) -> None:
        # lambda = 2 mm, d = 0.5 mm, v = 25 m/s gives I_G ~ 4.0.
        groups = dimensionless_groups(
            speed_m_s=25.0,
            feature_length_m=0.1,
            grain_diameter_m=0.5e-3,
            element_size_m=2.0e-3,
        )
        assert groups.macro_inertial_number == pytest.approx(4.0, rel=2e-2)

    def test_refining_the_mesh_makes_the_element_number_worse(self) -> None:
        coarse = dimensionless_groups(
            speed_m_s=25.0,
            feature_length_m=0.1,
            grain_diameter_m=0.5e-3,
            element_size_m=4.0e-3,
        )
        fine = dimensionless_groups(
            speed_m_s=25.0,
            feature_length_m=0.1,
            grain_diameter_m=0.5e-3,
            element_size_m=1.0e-3,
        )
        assert fine.macro_inertial_number > coarse.macro_inertial_number

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"speed_m_s": float("nan")},
            {"speed_m_s": -1.0},
            {"grain_diameter_m": 0.0},
            {"element_size_m": -1.0},
            {"feature_length_m": 0.0},
        ],
    )
    def test_rejects_unusable_inputs(self, kwargs: dict[str, float]) -> None:
        arguments: dict[str, float] = {
            "speed_m_s": 10.0,
            "feature_length_m": 0.1,
            "grain_diameter_m": 0.5e-3,
            "element_size_m": 2.0e-3,
        }
        arguments.update(kwargs)
        with pytest.raises(SolverInputError):
            dimensionless_groups(**arguments)


class TestVerdictAtBunkerSpeed:
    """25 m/s: everything the addendum warns about, at once."""

    def _verdict(self, **overrides: object) -> ValidityVerdict:
        arguments: dict[str, object] = {
            "speed_m_s": 25.0,
            "feature_lengths_m": _CLUBHEAD_SCALES_M,
            "grain_diameter_m": 0.33e-3,
            "element_size_m": 2.0e-3,
            "dynamic_terms_active": True,
            "submerged_depth_m": 0.040,
        }
        arguments.update(overrides)
        return evaluate_envelope(**arguments)  # type: ignore[arg-type]

    def test_is_flagged_beyond_validation_not_merely_extrapolated(self) -> None:
        verdict = self._verdict()
        assert verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert not verdict.is_within_stated_envelope

    def test_governing_scale_is_the_smallest_feature(self) -> None:
        verdict = self._verdict()
        assert verdict.governing.scale.name == "leading edge"
        # The smallest feature fails hardest: the clubhead scale alone
        # would report I = 0.1-ish and look almost respectable.
        assert (
            verdict.groups[0].micro_inertial_number
            > verdict.groups[-1].micro_inertial_number
        )

    def test_carries_every_standing_caveat(self) -> None:
        verdict = self._verdict()
        for caveat in STANDING_CAVEATS:
            assert caveat in verdict.caveats

    def test_names_the_froude_speed_and_grain_rate_failures(self) -> None:
        verdict = self._verdict()
        assert Caveat.SUPERCRITICAL_FROUDE in verdict.caveats
        assert Caveat.GRAIN_RATE_EFFECTS in verdict.caveats
        assert Caveat.BEYOND_PUBLISHED_SPEED in verdict.caveats
        assert Caveat.UNCALIBRATED_STRUCTURAL_CORRECTION in verdict.caveats

    def test_summary_quotes_the_numbers_and_the_reasons(self) -> None:
        text = self._verdict().summary()
        assert "BEYOND_VALIDATION" in text
        assert "leading edge" in text
        assert str(RFT_FROUDE_LIMIT) in text
        assert str(MAX_VALIDATED_SPEED_M_S) in text

    def test_each_finding_is_stated_once(self) -> None:
        reasons = self._verdict().reasons
        assert len(reasons) == len(set(reasons))

    def test_is_not_a_refusal_while_the_dynamic_terms_are_on(self) -> None:
        verdict = self._verdict()
        assert not verdict.is_refusal
        verdict.require_usable(RefusalPolicy.STRICT)


class TestRefusals:
    """The cases where no number may be reported."""

    def test_quasi_static_above_the_froude_ceiling_is_refused(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=25.0,
            feature_lengths_m=_CLUBHEAD_SCALES_M,
            grain_diameter_m=0.33e-3,
            element_size_m=2.0e-3,
            dynamic_terms_active=False,
        )
        assert verdict.status is EnvelopeStatus.REFUSED
        assert any("dynamic terms switched off" in reason for reason in verdict.reasons)

    def test_strict_policy_turns_a_refusal_into_an_exception(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=25.0,
            feature_lengths_m=_CLUBHEAD_SCALES_M,
            grain_diameter_m=0.33e-3,
            element_size_m=2.0e-3,
            dynamic_terms_active=False,
        )
        with pytest.raises(OutOfEnvelopeError) as excinfo:
            verdict.require_usable(RefusalPolicy.STRICT)
        assert excinfo.value.verdict is verdict

    def test_reporting_policy_returns_the_refusal_as_a_value(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=25.0,
            feature_lengths_m=_CLUBHEAD_SCALES_M,
            grain_diameter_m=0.33e-3,
            element_size_m=2.0e-3,
            dynamic_terms_active=False,
        )
        verdict.require_usable(RefusalPolicy.REPORT)
        assert verdict.is_refusal

    def test_a_feature_only_a_few_grains_wide_is_refused(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=0.2,
            feature_lengths_m={"leading edge": 0.002},
            grain_diameter_m=1.0e-3,
            element_size_m=1.0e-3,
            dynamic_terms_active=True,
        )
        assert verdict.status is EnvelopeStatus.REFUSED
        assert verdict.governing.continuum_size_ratio < MIN_CONTINUUM_SIZE_RATIO

    def test_the_reference_case_is_not_refused_on_grain_count(self) -> None:
        # The addendum's own 5 mm / 0.5 mm reference case must survive:
        # a limit that refuses the reference case is describing the rule,
        # not the physics.
        verdict = evaluate_envelope(
            speed_m_s=0.2,
            feature_lengths_m={"leading edge": 0.005},
            grain_diameter_m=0.5e-3,
            element_size_m=2.0e-3,
            dynamic_terms_active=True,
        )
        assert verdict.status is not EnvelopeStatus.REFUSED
        assert Caveat.MARGINAL_CONTINUUM in verdict.caveats


class TestQuasiStaticQueriesStayInsideTheEnvelope:
    """A slow laboratory intrusion is the one case that passes cleanly."""

    def test_a_slow_deep_intrusion_is_within_the_stated_envelope(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=0.05,
            feature_lengths_m={"plate": 0.05},
            grain_diameter_m=0.3e-3,
            element_size_m=5.0e-3,
            dynamic_terms_active=True,
            submerged_depth_m=0.05,
            structural_correction_calibrated=True,
        )
        assert verdict.status is EnvelopeStatus.WITHIN
        assert verdict.governing.froude < RFT_FROUDE_LIMIT
        assert verdict.governing.micro_inertial_number < RFT_INERTIAL_NUMBER_LIMIT
        # Even a clean verdict keeps the standing caveats.
        assert Caveat.TRANSIENT_RESPONSE in verdict.caveats

    def test_a_barely_submerged_body_is_flagged_shallow(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=0.05,
            feature_lengths_m={"plate": 0.05},
            grain_diameter_m=0.3e-3,
            element_size_m=5.0e-3,
            dynamic_terms_active=True,
            submerged_depth_m=0.2e-3,
        )
        assert Caveat.SHALLOW_INTRUSION in verdict.caveats


class TestVerdictConstruction:
    """A verdict cannot be built without something to judge."""

    def test_requires_at_least_one_feature_scale(self) -> None:
        with pytest.raises(SolverInputError, match="at least one feature scale"):
            evaluate_envelope(
                speed_m_s=1.0,
                feature_lengths_m={},
                grain_diameter_m=0.3e-3,
                element_size_m=1e-3,
                dynamic_terms_active=True,
            )

    def test_rejects_an_out_of_range_governing_index(self) -> None:
        verdict = evaluate_envelope(
            speed_m_s=1.0,
            feature_lengths_m={"plate": 0.05},
            grain_diameter_m=0.3e-3,
            element_size_m=1e-3,
            dynamic_terms_active=True,
        )
        with pytest.raises(SolverInputError):
            ValidityVerdict(
                status=verdict.status, groups=verdict.groups, governing_index=7
            )


class TestCombiningVerdicts:
    """A shot is only as answerable as its worst step."""

    def _at(self, speed: float, *, dynamic: bool = True) -> ValidityVerdict:
        return evaluate_envelope(
            speed_m_s=speed,
            feature_lengths_m={"plate": 0.05},
            grain_diameter_m=0.3e-3,
            element_size_m=5.0e-3,
            dynamic_terms_active=dynamic,
        )

    def test_one_refused_step_refuses_the_whole_trace(self) -> None:
        combined = worst_of(
            [self._at(0.05), self._at(0.05), self._at(25.0, dynamic=False)]
        )
        assert combined.status is EnvelopeStatus.REFUSED

    def test_caveats_are_the_union_across_steps(self) -> None:
        combined = worst_of([self._at(0.05), self._at(25.0)])
        assert Caveat.BEYOND_PUBLISHED_SPEED in combined.caveats
        assert Caveat.TRANSIENT_RESPONSE in combined.caveats

    def test_combined_caveats_are_deduplicated(self) -> None:
        combined = worst_of([self._at(25.0)] * 5)
        assert len(combined.caveats) == len(set(combined.caveats))
        assert len(combined.reasons) == len(set(combined.reasons))

    def test_refuses_to_combine_nothing(self) -> None:
        with pytest.raises(SolverInputError):
            worst_of([])
